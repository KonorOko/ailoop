use std::time::Duration;

use ailoop_core::{ProviderError, ProviderErrorKind, RetryClassification, Retryable};
use reqwest::StatusCode;

/// Discriminated category of an HTTP-level Azure OpenAI API error,
/// derived from the `error.code` field in the JSON envelope (Azure
/// also returns `error.type` on some endpoints — `code` is the more
/// stable surface).
///
/// Azure's code taxonomy is less stable than Anthropic's: the API
/// versions evolve and casing can drift. The `Other(String)` variant
/// preserves whatever was returned so callers (logs, metrics, future
/// `RetryingModel<M>`) never lose information.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AzureOpenAIApiErrorKind {
    /// Per-resource quota exceeded. Transient; honours `Retry-After`
    /// (and Azure's vendor-specific `retry-after-ms` header).
    RateLimit,
    /// Malformed body, unknown deployment, validation failure.
    /// Permanent — retrying without changes produces the same error.
    InvalidRequest,
    /// `context_length_exceeded` — the prompt (plus the requested
    /// completion tokens) does not fit the deployment's context
    /// window. Azure returns it as a 400 with `type:
    /// "invalid_request_error"`; it is split out of
    /// [`InvalidRequest`](Self::InvalidRequest) so callers can react
    /// by compacting the history. Permanent for
    /// [`RetryingModel`](ailoop_core::RetryingModel): resending the
    /// same prompt fails the same way.
    ContextOverflow,
    /// Azure content-safety filter blocked the request or response.
    /// Permanent — retrying produces the same block.
    ContentFilter,
    /// Missing or invalid API key / token. Permanent.
    Authentication,
    /// Caller authenticated successfully but lacks access. Permanent.
    Permission,
    /// Resource (deployment id, model name) not found. Permanent.
    NotFound,
    /// Specific case of [`NotFound`](Self::NotFound) where the error
    /// payload identifies the failing resource as the deployment. Kept
    /// distinct so callers can surface a clearer message (deployments
    /// are a common configuration error).
    DeploymentNotFound,
    /// Generic upstream 5xx. Treated as transient.
    ServerError,
    /// Forward-compatibility variant for `error.code` strings the
    /// adapter does not yet have a typed variant for. Treated
    /// conservatively as transient.
    Other(String),
}

impl AzureOpenAIApiErrorKind {
    /// Map Azure's `error.code` strings to typed variants. Azure mixes
    /// snake_case (`invalid_request_error`) with PascalCase
    /// (`DeploymentNotFound`); both forms are matched. Unknown codes
    /// land in `Other(s)` verbatim.
    pub fn from_error_code(s: &str) -> Self {
        match s {
            "rate_limit_exceeded" | "429" => Self::RateLimit,
            "invalid_request_error" | "BadRequest" => Self::InvalidRequest,
            "context_length_exceeded" => Self::ContextOverflow,
            "content_filter" => Self::ContentFilter,
            "invalid_api_key" | "Unauthorized" => Self::Authentication,
            "PermissionDenied" => Self::Permission,
            "NotFound" => Self::NotFound,
            "DeploymentNotFound" => Self::DeploymentNotFound,
            "server_error" | "InternalServerError" => Self::ServerError,
            other => Self::Other(other.to_string()),
        }
    }

    /// The provider-agnostic [`ProviderErrorKind`] for this kind.
    /// [`DeploymentNotFound`](Self::DeploymentNotFound) and
    /// [`NotFound`](Self::NotFound) are both
    /// [`ModelNotFound`](ProviderErrorKind::ModelNotFound), since a
    /// deployment is how Azure serves a model.
    pub fn provider_kind(&self) -> ProviderErrorKind {
        match self {
            Self::RateLimit => ProviderErrorKind::RateLimited,
            Self::InvalidRequest => ProviderErrorKind::InvalidRequest,
            Self::ContextOverflow => ProviderErrorKind::ContextOverflow,
            Self::ContentFilter => ProviderErrorKind::ContentFilter,
            Self::Authentication => ProviderErrorKind::Auth,
            Self::Permission => ProviderErrorKind::Permission,
            Self::NotFound | Self::DeploymentNotFound => ProviderErrorKind::ModelNotFound,
            Self::ServerError => ProviderErrorKind::Unavailable,
            Self::Other(_) => ProviderErrorKind::Other,
        }
    }
}

/// Failure surface of [`AzureOpenAIChatModel::chat_stream`](crate::AzureOpenAIChatModel)
/// and the surrounding HTTP / SSE plumbing.
///
/// Wrapped by the façade as
/// [`EngineError::Model`](https://docs.rs/ailoop) when it surfaces
/// during a run. Implements [`Retryable`] so
/// [`RetryingModel`](ailoop_core::RetryingModel) can drive backoff
/// off the variant.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AzureOpenAIError {
    /// Transport-level failure from `reqwest` (DNS, TLS, connection
    /// reset). Treated as transient by [`Retryable`].
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Typed 4xx/5xx response: the body parsed as Azure's documented
    /// error envelope (`{"error":{"code":..,"message":..}}`) and
    /// `Retry-After` / `retry-after-ms` were inspected. `retry_after`
    /// is `None` when both headers were missing or unparseable.
    #[error("Azure OpenAI API error ({status}, {kind:?}): {message}")]
    #[non_exhaustive]
    Api {
        /// HTTP status code returned by the API.
        status: StatusCode,
        /// Typed category derived from `error.code`.
        kind: AzureOpenAIApiErrorKind,
        /// Human-readable message from the error envelope.
        message: String,
        /// Parsed `Retry-After` (or vendor `retry-after-ms`), when
        /// present and parseable. HTTP-date form returns `None`.
        retry_after: Option<Duration>,
    },

    /// Fallback 4xx/5xx response: the body did not parse as the expected
    /// error envelope (e.g. an Azure Front Door HTML page). The raw
    /// body is preserved so callers can still surface it.
    #[error("API returned status {status}: {body}")]
    Status {
        /// HTTP status code returned by the upstream.
        status: StatusCode,
        /// Raw response body, preserved verbatim.
        body: String,
    },

    /// SSE framing error from `eventsource-stream` (chunked transport
    /// failure, unparseable event boundaries). Treated as permanent.
    #[error("SSE parse error: {0}")]
    Sse(#[from] eventsource_stream::EventStreamError<reqwest::Error>),

    /// JSON deserialization of an event payload failed. Permanent.
    #[error("malformed event payload: {0}")]
    Json(#[from] serde_json::Error),

    /// Mid-stream error event delivered over SSE: the service failed
    /// after the response started and sent `{"error":{...}}` in place
    /// of a chunk. No HTTP headers are available at this layer, so
    /// `retry_after` is intentionally absent; `kind` is derived from
    /// the event's `code` (or `type` when `code` is missing) with the
    /// same mapping as [`Api`](Self::Api).
    #[error("Azure OpenAI error event ({kind:?}): {message}")]
    #[non_exhaustive]
    Provider {
        /// Typed category derived from the event payload.
        kind: AzureOpenAIApiErrorKind,
        /// Human-readable message from the event payload.
        message: String,
    },

    /// Configuration error surfaced from `from_env` and similar constructors:
    /// missing endpoint, mutually exclusive secrets both set, etc.
    #[error("missing required configuration: {0}")]
    Config(String),

    /// A request carried content the Chat Completions wire model cannot
    /// represent: a [`ailoop_core::UserBlock::Document`], an image
    /// inside a tool result, or a [`ailoop_core::Source::FileId`] on an
    /// image block. Surfaced at request-build time before any HTTP call
    /// is made.
    ///
    /// To downgrade unsupported content automatically, install a
    /// [`ChatMiddleware`](ailoop_core::ChatMiddleware) that rewrites the
    /// request in `on_chat_request` — the adapter intentionally does
    /// not invent fallbacks. `kind` is a stable short label naming the
    /// shape that could not be encoded.
    #[error("unsupported content for Chat Completions: {kind}")]
    UnsupportedContent {
        /// Stable short label for the shape that could not be encoded
        /// (`"document"`, `"tool_result_image"`, `"image_file_id"`).
        kind: &'static str,
    },
}

/// Map an Azure-typed `AzureOpenAIApiErrorKind` to a retry decision. Azure's code
/// taxonomy is less stable than Anthropic's, so `Other(_)` is treated
/// conservatively as transient — better an extra retry than to strand a
/// request when the API ships a new code we haven't typed yet.
fn classify_kind(
    kind: &AzureOpenAIApiErrorKind,
    retry_after: Option<Duration>,
) -> RetryClassification {
    match kind {
        AzureOpenAIApiErrorKind::RateLimit
        | AzureOpenAIApiErrorKind::ServerError
        | AzureOpenAIApiErrorKind::Other(_) => RetryClassification::Transient { retry_after },
        AzureOpenAIApiErrorKind::Authentication
        | AzureOpenAIApiErrorKind::Permission
        | AzureOpenAIApiErrorKind::InvalidRequest
        | AzureOpenAIApiErrorKind::ContextOverflow
        | AzureOpenAIApiErrorKind::NotFound
        | AzureOpenAIApiErrorKind::DeploymentNotFound
        | AzureOpenAIApiErrorKind::ContentFilter => RetryClassification::Permanent,
    }
}

impl Retryable for AzureOpenAIError {
    fn retry_classification(&self) -> RetryClassification {
        match self {
            AzureOpenAIError::Api {
                kind, retry_after, ..
            } => classify_kind(kind, *retry_after),
            AzureOpenAIError::Status { status, .. } => {
                if status.is_server_error() {
                    RetryClassification::Transient { retry_after: None }
                } else {
                    RetryClassification::Permanent
                }
            }
            AzureOpenAIError::Provider { kind, .. } => classify_kind(kind, None),
            AzureOpenAIError::Http(_) => RetryClassification::Transient { retry_after: None },
            // Parsing failures are deterministic.
            AzureOpenAIError::Sse(_)
            | AzureOpenAIError::Json(_)
            | AzureOpenAIError::Config(_)
            | AzureOpenAIError::UnsupportedContent { .. } => RetryClassification::Permanent,
        }
    }
}

/// Kind for a response whose body is not Azure's error envelope, read
/// from the status alone.
fn kind_from_status(status: StatusCode) -> ProviderErrorKind {
    match status.as_u16() {
        401 => ProviderErrorKind::Auth,
        403 => ProviderErrorKind::Permission,
        404 => ProviderErrorKind::ModelNotFound,
        413 => ProviderErrorKind::RequestTooLarge,
        429 => ProviderErrorKind::RateLimited,
        500..=599 => ProviderErrorKind::Unavailable,
        _ => ProviderErrorKind::Other,
    }
}

impl ProviderError for AzureOpenAIError {
    /// - [`Api`](AzureOpenAIError::Api) and
    ///   [`Provider`](AzureOpenAIError::Provider): from their
    ///   [`AzureOpenAIApiErrorKind`] (see
    ///   [`AzureOpenAIApiErrorKind::provider_kind`]), whether it arrived
    ///   as an HTTP error envelope or as a mid-stream error event.
    /// - [`Status`](AzureOpenAIError::Status): from the HTTP status.
    /// - [`Http`](AzureOpenAIError::Http) and an SSE transport failure:
    ///   [`Network`](ProviderErrorKind::Network).
    /// - Any other SSE failure, [`Json`](AzureOpenAIError::Json),
    ///   [`Config`](AzureOpenAIError::Config) and
    ///   [`UnsupportedContent`](AzureOpenAIError::UnsupportedContent):
    ///   [`Other`](ProviderErrorKind::Other).
    fn kind(&self) -> ProviderErrorKind {
        match self {
            AzureOpenAIError::Api { kind, .. } | AzureOpenAIError::Provider { kind, .. } => {
                kind.provider_kind()
            }
            AzureOpenAIError::Status { status, .. } => kind_from_status(*status),
            AzureOpenAIError::Http(_)
            | AzureOpenAIError::Sse(eventsource_stream::EventStreamError::Transport(_)) => {
                ProviderErrorKind::Network
            }
            AzureOpenAIError::Sse(_)
            | AzureOpenAIError::Json(_)
            | AzureOpenAIError::Config(_)
            | AzureOpenAIError::UnsupportedContent { .. } => ProviderErrorKind::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ailoop_core::{ChatRequest, CompletionModel, RetryConfig, RetryingModel, StreamChunk};
    use futures::stream::BoxStream;

    use super::*;

    #[test]
    fn rate_limit_with_retry_after_is_transient() {
        let err = AzureOpenAIError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: AzureOpenAIApiErrorKind::RateLimit,
            message: "throttled".into(),
            retry_after: Some(Duration::from_millis(750)),
        };
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient {
                retry_after: Some(Duration::from_millis(750))
            },
        );
    }

    #[test]
    fn server_error_is_transient() {
        let err = AzureOpenAIError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: AzureOpenAIApiErrorKind::ServerError,
            message: "boom".into(),
            retry_after: None,
        };
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient { retry_after: None },
        );
    }

    #[test]
    fn authentication_is_permanent() {
        let err = AzureOpenAIError::Api {
            status: StatusCode::UNAUTHORIZED,
            kind: AzureOpenAIApiErrorKind::Authentication,
            message: "bad key".into(),
            retry_after: None,
        };
        assert_eq!(err.retry_classification(), RetryClassification::Permanent);
    }

    #[test]
    fn deployment_not_found_is_permanent() {
        let err = AzureOpenAIError::Api {
            status: StatusCode::NOT_FOUND,
            kind: AzureOpenAIApiErrorKind::DeploymentNotFound,
            message: "no such deployment".into(),
            retry_after: None,
        };
        assert_eq!(err.retry_classification(), RetryClassification::Permanent);
    }

    #[test]
    fn context_overflow_is_permanent() {
        let err = AzureOpenAIError::Api {
            status: StatusCode::BAD_REQUEST,
            kind: AzureOpenAIApiErrorKind::ContextOverflow,
            message: "Your input exceeds the context window of this model.".into(),
            retry_after: None,
        };
        assert_eq!(err.retry_classification(), RetryClassification::Permanent);
    }

    #[test]
    fn provider_error_flags_only_context_overflow() {
        let overflow = AzureOpenAIError::Api {
            status: StatusCode::BAD_REQUEST,
            kind: AzureOpenAIApiErrorKind::ContextOverflow,
            message: "too long".into(),
            retry_after: None,
        };
        let other = AzureOpenAIError::Api {
            status: StatusCode::BAD_REQUEST,
            kind: AzureOpenAIApiErrorKind::InvalidRequest,
            message: "bad field".into(),
            retry_after: None,
        };
        let event = AzureOpenAIError::Provider {
            kind: AzureOpenAIApiErrorKind::ContextOverflow,
            message: "too long".into(),
        };
        assert!(overflow.is_context_overflow());
        assert!(event.is_context_overflow());
        assert!(!other.is_context_overflow());
    }

    #[test]
    fn provider_event_is_classified_by_kind() {
        let server = AzureOpenAIError::Provider {
            kind: AzureOpenAIApiErrorKind::ServerError,
            message: "boom".into(),
        };
        assert_eq!(
            server.retry_classification(),
            RetryClassification::Transient { retry_after: None },
        );
        let filtered = AzureOpenAIError::Provider {
            kind: AzureOpenAIApiErrorKind::ContentFilter,
            message: "blocked".into(),
        };
        assert_eq!(
            filtered.retry_classification(),
            RetryClassification::Permanent
        );
    }

    #[test]
    fn unknown_kind_is_conservatively_transient() {
        let err = AzureOpenAIError::Api {
            status: StatusCode::BAD_GATEWAY,
            kind: AzureOpenAIApiErrorKind::Other("WeirdNewCode".into()),
            message: "?".into(),
            retry_after: None,
        };
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient { retry_after: None },
        );
    }

    /// Minimal model whose `chat_stream` setup always fails with the
    /// error built by `make`, counting calls so tests can observe
    /// whether [`RetryingModel`](ailoop_core::RetryingModel) reissued it.
    struct FailingModel {
        make: fn() -> AzureOpenAIError,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl CompletionModel for FailingModel {
        type Error = AzureOpenAIError;

        fn name(&self) -> &str {
            "failing"
        }

        fn model(&self) -> &str {
            "failing"
        }

        async fn chat_stream(
            &self,
            _req: ChatRequest,
        ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err((self.make)())
        }
    }

    async fn calls_through_retrying_model(make: fn() -> AzureOpenAIError) -> usize {
        let mut config = RetryConfig::default();
        config.base_delay = Duration::from_millis(1);
        config.max_delay = Duration::from_millis(1);
        config.jitter = false;
        let model = RetryingModel::with_config(
            FailingModel {
                make,
                calls: AtomicUsize::new(0),
            },
            config,
        );
        assert!(
            model
                .chat_stream(ChatRequest::new(vec![], 0))
                .await
                .is_err(),
            "setup must fail",
        );
        model.inner().calls.load(Ordering::SeqCst)
    }

    #[tokio::test]
    async fn retrying_model_does_not_retry_context_overflow() {
        assert_eq!(
            calls_through_retrying_model(|| AzureOpenAIError::Api {
                status: StatusCode::BAD_REQUEST,
                kind: AzureOpenAIApiErrorKind::ContextOverflow,
                message: "too long".into(),
                retry_after: None,
            })
            .await,
            1
        );
        // Control: a transient error on the same harness is retried up
        // to `max_attempts`, so the assertion above is not vacuous.
        assert_eq!(
            calls_through_retrying_model(|| AzureOpenAIError::Api {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                kind: AzureOpenAIApiErrorKind::ServerError,
                message: "boom".into(),
                retry_after: None,
            })
            .await,
            3
        );
    }

    #[test]
    fn api_kinds_map_to_provider_kinds() {
        let cases = [
            (
                AzureOpenAIApiErrorKind::RateLimit,
                ProviderErrorKind::RateLimited,
            ),
            (
                AzureOpenAIApiErrorKind::InvalidRequest,
                ProviderErrorKind::InvalidRequest,
            ),
            (
                AzureOpenAIApiErrorKind::ContextOverflow,
                ProviderErrorKind::ContextOverflow,
            ),
            (
                AzureOpenAIApiErrorKind::ContentFilter,
                ProviderErrorKind::ContentFilter,
            ),
            (
                AzureOpenAIApiErrorKind::Authentication,
                ProviderErrorKind::Auth,
            ),
            (
                AzureOpenAIApiErrorKind::Permission,
                ProviderErrorKind::Permission,
            ),
            (
                AzureOpenAIApiErrorKind::NotFound,
                ProviderErrorKind::ModelNotFound,
            ),
            (
                AzureOpenAIApiErrorKind::DeploymentNotFound,
                ProviderErrorKind::ModelNotFound,
            ),
            (
                AzureOpenAIApiErrorKind::ServerError,
                ProviderErrorKind::Unavailable,
            ),
            (
                AzureOpenAIApiErrorKind::Other("future".into()),
                ProviderErrorKind::Other,
            ),
        ];
        for (api, expected) in cases {
            let http = AzureOpenAIError::Api {
                status: StatusCode::BAD_REQUEST,
                kind: api.clone(),
                message: String::new(),
                retry_after: None,
            };
            let event = AzureOpenAIError::Provider {
                kind: api.clone(),
                message: String::new(),
            };
            assert_eq!(http.kind(), expected, "{api:?}");
            assert_eq!(event.kind(), expected, "{api:?}");
        }
    }

    #[test]
    fn status_fallback_kind_comes_from_the_status() {
        let cases = [
            (401, ProviderErrorKind::Auth),
            (403, ProviderErrorKind::Permission),
            (404, ProviderErrorKind::ModelNotFound),
            (413, ProviderErrorKind::RequestTooLarge),
            (429, ProviderErrorKind::RateLimited),
            (503, ProviderErrorKind::Unavailable),
            (400, ProviderErrorKind::Other),
        ];
        for (status, expected) in cases {
            let err = AzureOpenAIError::Status {
                status: StatusCode::from_u16(status).unwrap(),
                body: "<html/>".into(),
            };
            assert_eq!(err.kind(), expected, "{status}");
        }
    }

    #[test]
    fn local_failures_are_other() {
        let json = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        assert_eq!(
            AzureOpenAIError::Json(json).kind(),
            ProviderErrorKind::Other
        );
        let utf8 = String::from_utf8(vec![0xff]).unwrap_err();
        let sse = AzureOpenAIError::Sse(eventsource_stream::EventStreamError::Utf8(utf8));
        assert_eq!(sse.kind(), ProviderErrorKind::Other);
        let unsupported = AzureOpenAIError::UnsupportedContent { kind: "document" };
        assert_eq!(unsupported.kind(), ProviderErrorKind::Other);
    }
}
