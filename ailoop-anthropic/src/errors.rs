use std::time::Duration;

use ailoop_core::{RetryClassification, Retryable};
use reqwest::StatusCode;

/// Discriminated category of an HTTP-level Anthropic API error, derived
/// from the `error.type` field in the JSON envelope. The `Other` variant
/// preserves any forward-compatible error types Anthropic may add later.
///
/// Consumers (for example a future `RetryingModel<M>`) can pattern-match
/// `Overloaded` and `RateLimit` to drive backoff, and ignore the rest.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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

    /// Mid-stream error event delivered over SSE. No HTTP headers are
    /// available at this layer, so `retry_after` is intentionally absent;
    /// the typed `kind` lets callers (e.g. `RetryingModel<M>`) match on
    /// `ApiErrorKind::Overloaded` without parsing strings.
    #[error("Anthropic error event ({kind:?}): {message}")]
    Provider { kind: ApiErrorKind, message: String },
}

/// Map an Anthropic-typed `ApiErrorKind` to a retry decision. Used both
/// for HTTP-envelope errors (where `retry_after` may be `Some`) and for
/// SSE `Provider` errors (where it is always `None`).
fn classify_kind(kind: &ApiErrorKind, retry_after: Option<Duration>) -> RetryClassification {
    match kind {
        // Overloaded / rate limit / generic api_error are the canonical
        // transient signals on Anthropic. `Other(_)` is conservative —
        // unknown kinds default to transient so we don't strand a request
        // on a future error type that's actually retryable.
        ApiErrorKind::Overloaded
        | ApiErrorKind::RateLimit
        | ApiErrorKind::Api
        | ApiErrorKind::Other(_) => RetryClassification::Transient { retry_after },
        ApiErrorKind::Authentication
        | ApiErrorKind::Permission
        | ApiErrorKind::InvalidRequest
        | ApiErrorKind::NotFound
        | ApiErrorKind::RequestTooLarge => RetryClassification::Permanent,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limit_with_retry_after_is_transient() {
        let err = AnthropicError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: ApiErrorKind::RateLimit,
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
            kind: ApiErrorKind::Overloaded,
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
            kind: ApiErrorKind::Authentication,
            message: "bad key".into(),
            retry_after: None,
        };
        assert_eq!(err.retry_classification(), RetryClassification::Permanent);
    }

    #[test]
    fn unknown_kind_is_conservatively_transient() {
        let err = AnthropicError::Api {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: ApiErrorKind::Other("future_kind".into()),
            message: "?".into(),
            retry_after: None,
        };
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient { retry_after: None },
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
}
