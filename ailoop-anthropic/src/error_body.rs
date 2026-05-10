use std::time::Duration;

use reqwest::header::HeaderMap;
use serde::Deserialize;

use crate::errors::{AnthropicApiErrorKind, AnthropicError};

/// Top-level shape of Anthropic's HTTP error response:
/// `{"type":"error","error":{"type":"...","message":"..."}}`.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicApiErrorBody {
    pub error: AnthropicApiError,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicApiError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Parse the `Retry-After` header as integer seconds. Anthropic returns
/// a delay-seconds value in practice; HTTP-date form is left as a TODO
/// — adding `httpdate`/`chrono` for a path that does not occur in the
/// wild would be premature.
pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get("retry-after")?.to_str().ok()?;
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

/// Convert a non-success HTTP response (status + body + headers) into
/// the most specific `AnthropicError` variant we can. Falls back to
/// `Status` when the body does not parse as the expected envelope, so
/// callers always receive the raw body for unknown shapes (HTML from
/// upstream proxies, etc.).
pub(crate) fn classify_http_error(
    status: reqwest::StatusCode,
    body: String,
    retry_after: Option<Duration>,
) -> AnthropicError {
    match serde_json::from_str::<AnthropicApiErrorBody>(&body) {
        Ok(parsed) => AnthropicError::Api {
            status,
            kind: AnthropicApiErrorKind::from_error_type(&parsed.error.error_type),
            message: parsed.error.message,
            retry_after,
        },
        Err(_) => AnthropicError::Status { status, body },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};
    use reqwest::StatusCode;

    #[test]
    fn parses_retry_after_seconds() {
        let mut h = HeaderMap::new();
        h.insert("retry-after", HeaderValue::from_static("30"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(30)));
    }

    #[test]
    fn missing_retry_after_returns_none() {
        let h = HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn http_date_retry_after_is_not_yet_supported() {
        // Documents current behavior: the helper falls back to `None`
        // for HTTP-date form (`Wed, 21 Oct 2015 07:28:00 GMT`). Replace
        // this test if we add date parsing later.
        let mut h = HeaderMap::new();
        h.insert(
            "retry-after",
            HeaderValue::from_static("Wed, 21 Oct 2015 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn classifies_rate_limit_with_retry_after() {
        let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#
            .to_string();
        let err = classify_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            body,
            Some(Duration::from_secs(15)),
        );
        match err {
            AnthropicError::Api {
                status,
                kind,
                message,
                retry_after,
            } => {
                assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
                assert_eq!(kind, AnthropicApiErrorKind::RateLimit);
                assert_eq!(message, "slow down");
                assert_eq!(retry_after, Some(Duration::from_secs(15)));
            }
            other => panic!("expected Api variant, got {other:?}"),
        }
    }

    #[test]
    fn classifies_overloaded() {
        let body =
            r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#.to_string();
        let err = classify_http_error(StatusCode::from_u16(529).unwrap(), body, None);
        assert!(matches!(
            err,
            AnthropicError::Api {
                kind: AnthropicApiErrorKind::Overloaded,
                ..
            }
        ));
    }

    #[test]
    fn unknown_error_type_is_captured_as_other() {
        let body =
            r#"{"type":"error","error":{"type":"future_error_kind","message":"x"}}"#.to_string();
        let err = classify_http_error(StatusCode::IM_A_TEAPOT, body, None);
        match err {
            AnthropicError::Api { kind, .. } => {
                assert_eq!(
                    kind,
                    AnthropicApiErrorKind::Other("future_error_kind".into())
                );
            }
            other => panic!("expected Api variant, got {other:?}"),
        }
    }

    #[test]
    fn non_json_body_falls_back_to_status() {
        let body = "<html>nginx 502 bad gateway</html>".to_string();
        let err = classify_http_error(StatusCode::BAD_GATEWAY, body.clone(), None);
        match err {
            AnthropicError::Status {
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
