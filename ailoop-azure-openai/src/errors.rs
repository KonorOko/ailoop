use std::time::Duration;

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
pub enum ApiErrorKind {
    RateLimit,
    InvalidRequest,
    ContentFilter,
    Authentication,
    Permission,
    NotFound,
    DeploymentNotFound,
    ServerError,
    Other(String),
}

impl ApiErrorKind {
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

#[derive(Debug, thiserror::Error)]
pub enum AzureOpenAIError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Typed 4xx/5xx response: the body parsed as Azure's documented
    /// error envelope (`{"error":{"code":..,"message":..}}`) and
    /// `Retry-After` / `retry-after-ms` were inspected. `retry_after`
    /// is `None` when both headers were missing or unparseable.
    #[error("Azure OpenAI API error ({status}, {kind:?}): {message}")]
    Api {
        status: StatusCode,
        kind: ApiErrorKind,
        message: String,
        retry_after: Option<Duration>,
    },

    /// Fallback 4xx/5xx response: the body did not parse as the expected
    /// error envelope (e.g. an Azure Front Door HTML page). The raw
    /// body is preserved so callers can still surface it.
    #[error("API returned status {status}: {body}")]
    Status { status: StatusCode, body: String },

    #[error("SSE parse error: {0}")]
    Sse(#[from] eventsource_stream::EventStreamError<reqwest::Error>),

    #[error("malformed event payload: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Azure OpenAI error event: {error_type}: {message}")]
    Provider { error_type: String, message: String },

    /// Configuration error surfaced from `from_env` and similar constructors:
    /// missing endpoint, mutually exclusive secrets both set, etc.
    #[error("missing required configuration: {0}")]
    Config(String),
}
