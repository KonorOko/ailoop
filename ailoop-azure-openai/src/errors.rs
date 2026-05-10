use std::time::Duration;

use ailoop_core::{RetryClassification, Retryable};
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
            "content_filter" => Self::ContentFilter,
            "invalid_api_key" | "Unauthorized" => Self::Authentication,
            "PermissionDenied" => Self::Permission,
            "NotFound" => Self::NotFound,
            "DeploymentNotFound" => Self::DeploymentNotFound,
            "server_error" | "InternalServerError" => Self::ServerError,
            other => Self::Other(other.to_string()),
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

    /// Mid-stream error event delivered over SSE. Carries the raw
    /// `error.type` / message because Azure's SSE events do not expose
    /// the same `error.code` shape as HTTP-envelope errors. Treated as
    /// permanent — there is no typed signal to drive a retry decision.
    #[error("Azure OpenAI error event: {error_type}: {message}")]
    Provider {
        /// Raw `error.type` string from the event payload.
        error_type: String,
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
            AzureOpenAIError::Http(_) => RetryClassification::Transient { retry_after: None },
            // Parsing failures are deterministic. Mid-stream `Provider`
            // events on Azure are rare and we don't have a typed `kind`
            // to drive a smart decision — treat as permanent rather than
            // looping on something that's almost certainly an API bug.
            AzureOpenAIError::Sse(_)
            | AzureOpenAIError::Json(_)
            | AzureOpenAIError::Provider { .. }
            | AzureOpenAIError::Config(_)
            | AzureOpenAIError::UnsupportedContent { .. } => RetryClassification::Permanent,
        }
    }
}

#[cfg(test)]
mod tests {
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
}
