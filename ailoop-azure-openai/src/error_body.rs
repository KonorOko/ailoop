use std::time::Duration;

use reqwest::header::HeaderMap;
use serde::Deserialize;

use crate::errors::{AzureOpenAIApiErrorKind, AzureOpenAIError};

/// Top-level shape of Azure OpenAI's HTTP error response:
/// `{"error":{"code":"...","message":"...","type":"...","param":null}}`.
/// `code` is the most stable surface; `type` and `param` are not always
/// present.
#[derive(Debug, Deserialize)]
pub(crate) struct AzureApiErrorBody {
    pub error: AzureApiError,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AzureApiError {
    /// Optional because some Azure surfaces (e.g. older API versions)
    /// omit `code` and only ship `message`.
    #[serde(default)]
    pub code: Option<String>,
    pub message: String,
}

/// Parse the retry headers Azure may emit. `retry-after-ms` is checked
/// first because it carries a finer-grained value when present; we fall
/// back to the standard `Retry-After` integer-seconds form. HTTP-date
/// form is not yet supported (would require `httpdate`/`chrono` for a
/// path that does not occur in practice).
pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    if let Some(ms) = headers
        .get("retry-after-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
    {
        return Some(Duration::from_millis(ms));
    }
    headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Convert a non-success HTTP response (status + body + headers) into
/// the most specific `AzureOpenAIError` variant we can. Falls back to
/// `Status` when the body does not parse as the expected envelope.
pub(crate) fn classify_http_error(
    status: reqwest::StatusCode,
    body: String,
    retry_after: Option<Duration>,
) -> AzureOpenAIError {
    match serde_json::from_str::<AzureApiErrorBody>(&body) {
        Ok(parsed) => AzureOpenAIError::Api {
            status,
            kind: AzureOpenAIApiErrorKind::from_error_code(
                parsed.error.code.as_deref().unwrap_or(""),
            ),
            message: parsed.error.message,
            retry_after,
        },
        Err(_) => AzureOpenAIError::Status { status, body },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};
    use reqwest::StatusCode;

    #[test]
    fn parses_retry_after_ms_with_higher_priority() {
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("3"));
        h.insert("retry-after-ms", HeaderValue::from_static("1500"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn falls_back_to_retry_after_seconds() {
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("7"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(7)));
    }

    #[test]
    fn missing_retry_headers_returns_none() {
        let h = HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn classifies_rate_limit_with_retry_after_ms() {
        let body =
            r#"{"error":{"code":"rate_limit_exceeded","message":"slow down","type":"requests"}}"#
                .to_string();
        let err = classify_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            body,
            Some(Duration::from_millis(2500)),
        );
        match err {
            AzureOpenAIError::Api {
                status,
                kind,
                message,
                retry_after,
            } => {
                assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
                assert_eq!(kind, AzureOpenAIApiErrorKind::RateLimit);
                assert_eq!(message, "slow down");
                assert_eq!(retry_after, Some(Duration::from_millis(2500)));
            }
            other => panic!("expected Api variant, got {other:?}"),
        }
    }

    #[test]
    fn classifies_deployment_not_found() {
        let body =
            r#"{"error":{"code":"DeploymentNotFound","message":"no such deployment"}}"#.to_string();
        let err = classify_http_error(StatusCode::NOT_FOUND, body, None);
        assert!(matches!(
            err,
            AzureOpenAIError::Api {
                kind: AzureOpenAIApiErrorKind::DeploymentNotFound,
                ..
            }
        ));
    }

    #[test]
    fn classifies_content_filter() {
        let body = r#"{"error":{"code":"content_filter","message":"blocked"}}"#.to_string();
        let err = classify_http_error(StatusCode::BAD_REQUEST, body, None);
        assert!(matches!(
            err,
            AzureOpenAIError::Api {
                kind: AzureOpenAIApiErrorKind::ContentFilter,
                ..
            }
        ));
    }

    #[test]
    fn unknown_error_code_is_captured_as_other() {
        let body = r#"{"error":{"code":"FutureCode","message":"x"}}"#.to_string();
        let err = classify_http_error(StatusCode::IM_A_TEAPOT, body, None);
        match err {
            AzureOpenAIError::Api { kind, .. } => {
                assert_eq!(kind, AzureOpenAIApiErrorKind::Other("FutureCode".into()));
            }
            other => panic!("expected Api variant, got {other:?}"),
        }
    }

    #[test]
    fn missing_code_field_is_captured_as_empty_other() {
        // Older Azure API versions sometimes omit `code` entirely.
        let body = r#"{"error":{"message":"some message"}}"#.to_string();
        let err = classify_http_error(StatusCode::BAD_REQUEST, body, None);
        match err {
            AzureOpenAIError::Api { kind, message, .. } => {
                assert_eq!(kind, AzureOpenAIApiErrorKind::Other(String::new()));
                assert_eq!(message, "some message");
            }
            other => panic!("expected Api variant, got {other:?}"),
        }
    }

    #[test]
    fn non_json_body_falls_back_to_status() {
        let body = "<html>Azure Front Door 502</html>".to_string();
        let err = classify_http_error(StatusCode::BAD_GATEWAY, body.clone(), None);
        match err {
            AzureOpenAIError::Status {
                status,
                body: returned,
            } => {
                assert_eq!(status, StatusCode::BAD_GATEWAY);
                assert_eq!(returned, body);
            }
            other => panic!("expected Status fallback, got {other:?}"),
        }
    }
}
