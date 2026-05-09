use std::time::Duration;

use reqwest::StatusCode;

/// Discriminated category of an HTTP-level Anthropic API error, derived
/// from the `error.type` field in the JSON envelope. The `Other` variant
/// preserves any forward-compatible error types Anthropic may add later.
///
/// Consumers (for example a future `RetryingModel<M>`) can pattern-match
/// `Overloaded` and `RateLimit` to drive backoff, and ignore the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiErrorKind {
    Overloaded,
    RateLimit,
    InvalidRequest,
    Authentication,
    Permission,
    NotFound,
    RequestTooLarge,
    Api,
    Other(String),
}

impl ApiErrorKind {
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
}

#[derive(Debug, thiserror::Error)]
pub enum AnthropicError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Typed 4xx/5xx response: the body parsed as Anthropic's documented
    /// error envelope (`{"type":"error","error":{"type":..,"message":..}}`)
    /// and `Retry-After` was inspected. `retry_after` is `None` when the
    /// header was missing or unparseable.
    #[error("Anthropic API error ({status}, {kind:?}): {message}")]
    Api {
        status: StatusCode,
        kind: ApiErrorKind,
        message: String,
        retry_after: Option<Duration>,
    },

    /// Fallback 4xx/5xx response: the body did not parse as the expected
    /// error envelope (e.g. an upstream proxy returned HTML). The raw
    /// body is preserved so callers can still surface it.
    #[error("API returned status {status}: {body}")]
    Status { status: StatusCode, body: String },

    #[error("SSE parse error: {0}")]
    Sse(#[from] eventsource_stream::EventStreamError<reqwest::Error>),

    #[error("malformed event payload: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Anthropic error event: {error_type}: {message}")]
    Provider { error_type: String, message: String },
}
