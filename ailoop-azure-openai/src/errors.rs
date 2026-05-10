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

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
        kind: AzureOpenAIApiErrorKind,
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
            | AzureOpenAIError::Config(_) => RetryClassification::Permanent,
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
