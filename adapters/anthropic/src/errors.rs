use std::time::Duration;

use ailoop_core::{ProviderError, RetryClassification, Retryable};
use reqwest::StatusCode;

/// Discriminated category of an HTTP-level Anthropic API error, derived
/// from the `error.type` field in the JSON envelope. The `Other` variant
/// preserves any forward-compatible error types Anthropic may add later.
///
/// Consumers (for example a future `RetryingModel<M>`) can pattern-match
/// `Overloaded` and `RateLimit` to drive backoff, and ignore the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnthropicApiErrorKind {
    /// `overloaded_error` — Anthropic is at capacity. Transient;
    /// honours `Retry-After` when present.
    Overloaded,
    /// `rate_limit_error` — the caller exceeded a quota. Transient;
    /// honours `Retry-After`.
    RateLimit,
    /// `invalid_request_error` — malformed body, unknown model,
    /// validation failure. Permanent — retrying without changes
    /// produces the same error.
    InvalidRequest,
    /// `invalid_request_error` whose message reports that the prompt
    /// does not fit the model's context window (`"prompt is too
    /// long"`, HTTP 400). Split out of
    /// [`InvalidRequest`](Self::InvalidRequest) so callers can react
    /// by compacting the history. Permanent for
    /// [`RetryingModel`](ailoop_core::RetryingModel): resending the
    /// same prompt fails the same way; recovery needs the history,
    /// which lives above the model.
    ///
    /// Only produced by [`from_error`](Self::from_error), which sees
    /// the message; [`from_error_type`](Self::from_error_type) alone
    /// cannot tell it apart from `InvalidRequest`.
    ContextOverflow,
    /// `authentication_error` — missing or invalid API key.
    /// Permanent.
    Authentication,
    /// `permission_error` — the API key is valid but lacks access to
    /// the requested resource. Permanent.
    Permission,
    /// `not_found_error` — model id or resource does not exist.
    /// Permanent.
    NotFound,
    /// `request_too_large` — request body exceeded provider limits.
    /// Permanent unless the caller compacts and retries explicitly.
    RequestTooLarge,
    /// `api_error` — generic server-side failure with no further
    /// classification. Treated as transient.
    Api,
    /// Forward-compatibility variant for any future `error.type`
    /// strings. Treated conservatively as transient so unknown
    /// variants don't strand requests on a retryable code.
    Other(String),
}

impl AnthropicApiErrorKind {
    /// Map Anthropic's documented `error.type` strings to typed variants.
    /// Unknown types are captured as `Other(s)` so callers can still log
    /// them and so we never silently drop information.
    pub fn from_error_type(s: &str) -> Self {
        match s {
            "overloaded_error" => Self::Overloaded,
            "rate_limit_error" => Self::RateLimit,
            "invalid_request_error" => Self::InvalidRequest,
            "authentication_error" => Self::Authentication,
            "permission_error" => Self::Permission,
            "not_found_error" => Self::NotFound,
            "request_too_large" => Self::RequestTooLarge,
            "api_error" => Self::Api,
            other => Self::Other(other.to_string()),
        }
    }

    /// Like [`from_error_type`](Self::from_error_type), but also reads
    /// the message to tell a context-window overflow apart from other
    /// validation failures. Anthropic reports both as
    /// `invalid_request_error`; the overflow is recognised by its
    /// documented `"prompt is too long"` message (matched
    /// case-insensitively, since the API appends token counts).
    pub fn from_error(error_type: &str, message: &str) -> Self {
        match Self::from_error_type(error_type) {
            Self::InvalidRequest if is_prompt_too_long(message) => Self::ContextOverflow,
            kind => kind,
        }
    }
}

fn is_prompt_too_long(message: &str) -> bool {
    message.to_ascii_lowercase().contains("prompt is too long")
}

/// Failure surface of [`AnthropicChatModel::chat_stream`](crate::AnthropicChatModel)
/// and the surrounding HTTP / SSE plumbing.
///
/// Wrapped by the façade as
/// [`EngineError::Model`](https://docs.rs/ailoop) when it surfaces
/// during a run. Implements [`Retryable`] so
/// [`RetryingModel`](ailoop_core::RetryingModel) can drive backoff
/// off the variant.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AnthropicError {
    /// Transport-level failure from `reqwest` (DNS, TLS, connection
    /// reset). Treated as transient by [`Retryable`].
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Typed 4xx/5xx response: the body parsed as Anthropic's documented
    /// error envelope (`{"type":"error","error":{"type":..,"message":..}}`)
    /// and `Retry-After` was inspected. `retry_after` is `None` when the
    /// header was missing or unparseable.
    #[error("Anthropic API error ({status}, {kind:?}): {message}")]
    Api {
        /// HTTP status code returned by the API.
        status: StatusCode,
        /// Typed category derived from `error.type`.
        kind: AnthropicApiErrorKind,
        /// Human-readable message from the error envelope.
        message: String,
        /// Parsed `Retry-After` header, when present and parseable as
        /// integer seconds. HTTP-date form returns `None`.
        retry_after: Option<Duration>,
    },

    /// Fallback 4xx/5xx response: the body did not parse as the expected
    /// error envelope (e.g. an upstream proxy returned HTML). The raw
    /// body is preserved so callers can still surface it.
    #[error("API returned status {status}: {body}")]
    Status {
        /// HTTP status code returned by the upstream.
        status: StatusCode,
        /// Raw response body, preserved verbatim.
        body: String,
    },

    /// SSE framing error from `eventsource-stream` (chunked transport
    /// failure, unparseable event boundaries). Treated as permanent
    /// — retrying produces the same parse failure.
    #[error("SSE parse error: {0}")]
    Sse(#[from] eventsource_stream::EventStreamError<reqwest::Error>),

    /// JSON deserialization of an event payload failed. Permanent.
    #[error("malformed event payload: {0}")]
    Json(#[from] serde_json::Error),

    /// Mid-stream error event delivered over SSE. No HTTP headers are
    /// available at this layer, so `retry_after` is intentionally absent;
    /// the typed `kind` lets callers (e.g. `RetryingModel<M>`) match on
    /// `AnthropicApiErrorKind::Overloaded` without parsing strings.
    #[error("Anthropic error event ({kind:?}): {message}")]
    Provider {
        /// Typed category derived from the event payload.
        kind: AnthropicApiErrorKind,
        /// Human-readable message from the event payload.
        message: String,
    },
}

/// Map an Anthropic-typed `AnthropicApiErrorKind` to a retry decision. Used both
/// for HTTP-envelope errors (where `retry_after` may be `Some`) and for
/// SSE `Provider` errors (where it is always `None`).
fn classify_kind(
    kind: &AnthropicApiErrorKind,
    retry_after: Option<Duration>,
) -> RetryClassification {
    match kind {
        // Overloaded / rate limit / generic api_error are the canonical
        // transient signals on Anthropic. `Other(_)` is conservative —
        // unknown kinds default to transient so we don't strand a request
        // on a future error type that's actually retryable.
        AnthropicApiErrorKind::Overloaded
        | AnthropicApiErrorKind::RateLimit
        | AnthropicApiErrorKind::Api
        | AnthropicApiErrorKind::Other(_) => RetryClassification::Transient { retry_after },
        AnthropicApiErrorKind::Authentication
        | AnthropicApiErrorKind::Permission
        | AnthropicApiErrorKind::InvalidRequest
        | AnthropicApiErrorKind::ContextOverflow
        | AnthropicApiErrorKind::NotFound
        | AnthropicApiErrorKind::RequestTooLarge => RetryClassification::Permanent,
    }
}

impl Retryable for AnthropicError {
    fn retry_classification(&self) -> RetryClassification {
        match self {
            AnthropicError::Api {
                kind, retry_after, ..
            } => classify_kind(kind, *retry_after),
            AnthropicError::Provider { kind, .. } => classify_kind(kind, None),
            AnthropicError::Status { status, .. } => {
                if status.is_server_error() {
                    RetryClassification::Transient { retry_after: None }
                } else {
                    RetryClassification::Permanent
                }
            }
            AnthropicError::Http(_) => RetryClassification::Transient { retry_after: None },
            // Parse failures are deterministic — retrying won't change the bytes.
            AnthropicError::Sse(_) | AnthropicError::Json(_) => RetryClassification::Permanent,
        }
    }
}

impl ProviderError for AnthropicError {
    /// `true` for [`AnthropicApiErrorKind::ContextOverflow`], whether it
    /// arrived as an HTTP error envelope or as a mid-stream error event.
    fn is_context_overflow(&self) -> bool {
        matches!(
            self,
            AnthropicError::Api {
                kind: AnthropicApiErrorKind::ContextOverflow,
                ..
            } | AnthropicError::Provider {
                kind: AnthropicApiErrorKind::ContextOverflow,
                ..
            }
        )
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
        let err = AnthropicError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: AnthropicApiErrorKind::RateLimit,
            message: "slow down".into(),
            retry_after: Some(Duration::from_secs(2)),
        };
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient {
                retry_after: Some(Duration::from_secs(2))
            },
        );
    }

    #[test]
    fn overloaded_provider_event_is_transient_without_retry_after() {
        let err = AnthropicError::Provider {
            kind: AnthropicApiErrorKind::Overloaded,
            message: "overloaded".into(),
        };
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient { retry_after: None },
        );
    }

    #[test]
    fn authentication_is_permanent() {
        let err = AnthropicError::Api {
            status: StatusCode::UNAUTHORIZED,
            kind: AnthropicApiErrorKind::Authentication,
            message: "bad key".into(),
            retry_after: None,
        };
        assert_eq!(err.retry_classification(), RetryClassification::Permanent);
    }

    #[test]
    fn unknown_kind_is_conservatively_transient() {
        let err = AnthropicError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: AnthropicApiErrorKind::Other("future_kind".into()),
            message: "?".into(),
            retry_after: None,
        };
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient { retry_after: None },
        );
    }

    #[test]
    fn context_overflow_is_permanent() {
        let err = AnthropicError::Api {
            status: StatusCode::BAD_REQUEST,
            kind: AnthropicApiErrorKind::ContextOverflow,
            message: "prompt is too long: 1048577 tokens > 1000000 maximum".into(),
            retry_after: None,
        };
        assert_eq!(err.retry_classification(), RetryClassification::Permanent);
    }

    #[test]
    fn provider_error_flags_only_context_overflow() {
        let http = AnthropicError::Api {
            status: StatusCode::BAD_REQUEST,
            kind: AnthropicApiErrorKind::ContextOverflow,
            message: "prompt is too long".into(),
            retry_after: None,
        };
        let event = AnthropicError::Provider {
            kind: AnthropicApiErrorKind::ContextOverflow,
            message: "prompt is too long".into(),
        };
        let other = AnthropicError::Api {
            status: StatusCode::BAD_REQUEST,
            kind: AnthropicApiErrorKind::InvalidRequest,
            message: "bad field".into(),
            retry_after: None,
        };
        assert!(http.is_context_overflow());
        assert!(event.is_context_overflow());
        assert!(!other.is_context_overflow());
    }

    #[test]
    fn from_error_splits_overflow_from_other_invalid_requests() {
        assert_eq!(
            AnthropicApiErrorKind::from_error("invalid_request_error", "prompt is too long"),
            AnthropicApiErrorKind::ContextOverflow,
        );
        assert_eq!(
            AnthropicApiErrorKind::from_error(
                "invalid_request_error",
                "Prompt is too long: 1048577 tokens > 1000000 maximum",
            ),
            AnthropicApiErrorKind::ContextOverflow,
        );
        assert_eq!(
            AnthropicApiErrorKind::from_error(
                "invalid_request_error",
                "adaptive thinking is not supported on this model",
            ),
            AnthropicApiErrorKind::InvalidRequest,
        );
        // The phrase only means overflow on an invalid_request_error.
        assert_eq!(
            AnthropicApiErrorKind::from_error("api_error", "prompt is too long"),
            AnthropicApiErrorKind::Api,
        );
    }

    #[test]
    fn status_fallback_splits_on_server_vs_client() {
        let server = AnthropicError::Status {
            status: StatusCode::BAD_GATEWAY,
            body: "<html/>".into(),
        };
        assert_eq!(
            server.retry_classification(),
            RetryClassification::Transient { retry_after: None },
        );
        let client = AnthropicError::Status {
            status: StatusCode::BAD_REQUEST,
            body: "<html/>".into(),
        };
        assert_eq!(
            client.retry_classification(),
            RetryClassification::Permanent
        );
    }

    /// Minimal model whose `chat_stream` setup always fails with the
    /// error built by `make`, counting calls so tests can observe
    /// whether [`RetryingModel`](ailoop_core::RetryingModel) reissued it.
    struct FailingModel {
        make: fn() -> AnthropicError,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl CompletionModel for FailingModel {
        type Error = AnthropicError;

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

    async fn calls_through_retrying_model(make: fn() -> AnthropicError) -> usize {
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
            calls_through_retrying_model(|| AnthropicError::Api {
                status: StatusCode::BAD_REQUEST,
                kind: AnthropicApiErrorKind::ContextOverflow,
                message: "prompt is too long".into(),
                retry_after: None,
            })
            .await,
            1
        );
        // Control: a transient error on the same harness is retried up
        // to `max_attempts`, so the assertion above is not vacuous.
        assert_eq!(
            calls_through_retrying_model(|| AnthropicError::Api {
                status: StatusCode::from_u16(529).unwrap(),
                kind: AnthropicApiErrorKind::Overloaded,
                message: "busy".into(),
                retry_after: None,
            })
            .await,
            3
        );
    }
}
