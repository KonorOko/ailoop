use std::time::Duration;

use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::errors::{OpenAIApiErrorKind, OpenAIError};

/// What the request carried, so a vague "not supported" message is only
/// read as a missing capability when the request used it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RequestFeatures {
    pub images: bool,
    pub tools: bool,
}

/// Fields pulled out of an error body. Servers use several shapes:
///
/// - OpenAI, Ollama, LM Studio, vLLM: `{"error":{"message","type","code","param"}}`,
///   `code` a string or `null`.
/// - OpenRouter: the same, with a numeric `code`.
/// - llama.cpp and some LM Studio paths: `{"error":"message"}`.
/// - Others: `{"message":...}` or `{"detail":...}`.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ErrorFields {
    pub message: Option<String>,
    pub code: Option<String>,
    pub error_type: Option<String>,
}

pub(crate) fn parse_error_fields(value: &Value) -> ErrorFields {
    let scalar = |v: Option<&Value>| match v {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Number(n)) => Some(n.to_string()),
        _ => None,
    };
    match value.get("error") {
        Some(Value::Object(err)) => ErrorFields {
            message: scalar(err.get("message")),
            code: scalar(err.get("code")),
            error_type: scalar(err.get("type")),
        },
        Some(Value::String(s)) => ErrorFields {
            message: Some(s.clone()),
            ..Default::default()
        },
        _ => ErrorFields {
            message: scalar(value.get("message")).or_else(|| scalar(value.get("detail"))),
            ..Default::default()
        },
    }
}

/// Classify from what is known: the status when there is one (an HTTP
/// response), then code, type and message.
pub(crate) fn classify(
    status: Option<StatusCode>,
    fields: &ErrorFields,
    features: RequestFeatures,
) -> OpenAIApiErrorKind {
    let code = fields.code.as_deref().unwrap_or("");
    let error_type = fields.error_type.as_deref().unwrap_or("");
    let message = fields.message.as_deref().unwrap_or("").to_ascii_lowercase();
    let has = |token: &str| code == token || error_type == token;

    // The account is out of credit, whatever the status says.
    if has("insufficient_quota") || status == Some(StatusCode::PAYMENT_REQUIRED) {
        return OpenAIApiErrorKind::QuotaExceeded;
    }
    if has("context_length_exceeded") || mentions_context_overflow(&message) {
        return OpenAIApiErrorKind::ContextOverflow;
    }
    // Capability messages arrive as 400, 404 (OpenRouter's "No endpoints
    // found that support ...") or 500 depending on the server.
    if features.images && mentions_unsupported_images(&message) {
        return OpenAIApiErrorKind::UnsupportedImages;
    }
    if features.tools && mentions_unsupported_tools(&message) {
        return OpenAIApiErrorKind::UnsupportedTools;
    }

    match status {
        Some(StatusCode::UNAUTHORIZED) => return OpenAIApiErrorKind::Authentication,
        Some(StatusCode::FORBIDDEN) => return OpenAIApiErrorKind::Permission,
        Some(StatusCode::TOO_MANY_REQUESTS) => return OpenAIApiErrorKind::RateLimit,
        Some(StatusCode::PAYLOAD_TOO_LARGE) => return OpenAIApiErrorKind::RequestTooLarge,
        Some(StatusCode::NOT_FOUND) => {
            // Azure: `DeploymentNotFound`; on Azure the deployment is
            // the model.
            return if has("model_not_found")
                || has("DeploymentNotFound")
                || message.contains("model")
                || message.contains("deployment")
            {
                OpenAIApiErrorKind::ModelNotFound
            } else {
                OpenAIApiErrorKind::NotFound
            };
        }
        Some(s) if s.is_server_error() => return OpenAIApiErrorKind::ServerError,
        _ => {}
    }

    // No decisive status (a stream event, or a 400/422): use the code.
    // Azure mixes OpenAI's snake_case codes with PascalCase ones.
    match code {
        "invalid_api_key" | "401" | "Unauthorized" => OpenAIApiErrorKind::Authentication,
        "403" | "PermissionDenied" => OpenAIApiErrorKind::Permission,
        "rate_limit_exceeded" | "429" => OpenAIApiErrorKind::RateLimit,
        "model_not_found" | "DeploymentNotFound" => OpenAIApiErrorKind::ModelNotFound,
        "content_filter" | "content_policy_violation" | "ResponsibleAIPolicyViolation" => {
            OpenAIApiErrorKind::ContentFilter
        }
        "413" | "request_too_large" => OpenAIApiErrorKind::RequestTooLarge,
        "server_error" | "InternalServerError" | "500" | "502" | "503" => {
            OpenAIApiErrorKind::ServerError
        }
        "invalid_request_error" | "BadRequest" => OpenAIApiErrorKind::InvalidRequest,
        _ => match error_type {
            "server_error" => OpenAIApiErrorKind::ServerError,
            "authentication_error" => OpenAIApiErrorKind::Authentication,
            _ if matches!(
                status,
                Some(StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY)
            ) =>
            {
                OpenAIApiErrorKind::InvalidRequest
            }
            _ => OpenAIApiErrorKind::Other(
                fields
                    .code
                    .clone()
                    .or_else(|| fields.error_type.clone())
                    .unwrap_or_default(),
            ),
        },
    }
}

fn mentions_context_overflow(message: &str) -> bool {
    // OpenAI: "maximum context length is N tokens"; llama.cpp: "exceeds
    // the available context size"; LM Studio: "greater than the context
    // length"; vLLM: "longer than the model's context length".
    message.contains("maximum context length")
        || message.contains("context size")
        || message.contains("greater than the context length")
        || message.contains("longer than the model's context length")
        || message.contains("context window")
}

fn says_unsupported(message: &str) -> bool {
    message.contains("not support")
        || message.contains("unsupported")
        || message.contains("only supported")
        || message.contains("no endpoints found that support")
        || message.contains("not a multimodal")
        || message.contains("not enabled")
}

fn mentions_unsupported_images(message: &str) -> bool {
    (message.contains("image") || message.contains("vision") || message.contains("multimodal"))
        && says_unsupported(message)
}

fn mentions_unsupported_tools(message: &str) -> bool {
    (message.contains("tool") || message.contains("function call")) && says_unsupported(message)
}

/// Build the error for a non-success response.
pub(crate) fn classify_http_error(
    status: StatusCode,
    headers: &HeaderMap,
    body: String,
    features: RequestFeatures,
) -> OpenAIError {
    let fields = serde_json::from_str::<Value>(&body)
        .map(|v| parse_error_fields(&v))
        .unwrap_or_default();
    let kind = classify(Some(status), &fields, features);
    let retry_after = parse_retry_after(headers);
    let message = fields.message.clone().unwrap_or_else(|| {
        let trimmed = body.trim();
        if trimmed.is_empty() {
            status.canonical_reason().unwrap_or("").to_string()
        } else {
            trimmed.chars().take(2000).collect()
        }
    });
    OpenAIError::Api {
        status,
        kind,
        message,
        code: fields.code.or(fields.error_type),
        retry_after,
    }
}

/// The delay the server asked for, from the first header that has one:
///
/// 1. `retry-after-ms` (milliseconds; Azure and some proxies).
/// 2. `retry-after` in seconds. The HTTP-date form is not parsed.
/// 3. OpenAI's `x-ratelimit-reset-requests` / `x-ratelimit-reset-tokens`
///    (durations like `1s`, `6m0s`, `20ms`). When one of the matching
///    `x-ratelimit-remaining-*` headers is `0`, that limit's reset is
///    used; otherwise the longer of the two.
pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let get = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
    };

    if let Some(ms) = get("retry-after-ms").and_then(|s| s.parse::<f64>().ok())
        && ms.is_finite()
        && ms >= 0.0
    {
        return Some(Duration::from_secs_f64(ms / 1000.0));
    }
    if let Some(secs) = get("retry-after").and_then(|s| s.parse::<f64>().ok())
        && secs.is_finite()
        && secs >= 0.0
    {
        return Some(Duration::from_secs_f64(secs));
    }

    let requests = get("x-ratelimit-reset-requests").and_then(parse_go_duration);
    let tokens = get("x-ratelimit-reset-tokens").and_then(parse_go_duration);
    let exhausted = |name: &str| get(name) == Some("0");
    match (requests, tokens) {
        (Some(r), _) if exhausted("x-ratelimit-remaining-requests") => Some(r),
        (_, Some(t)) if exhausted("x-ratelimit-remaining-tokens") => Some(t),
        (Some(r), Some(t)) => Some(r.max(t)),
        (r, t) => r.or(t),
    }
}

/// Parse a Go-style duration as OpenAI sends it: `1s`, `6m0s`, `1.5s`,
/// `20ms`, `1h2m3s`.
pub(crate) fn parse_go_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut total = 0f64;
    let mut rest = s;
    while !rest.is_empty() {
        let num_len = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        if num_len == 0 {
            return None;
        }
        let value: f64 = rest[..num_len].parse().ok()?;
        rest = &rest[num_len..];
        let unit_len = rest
            .find(|c: char| c.is_ascii_digit() || c == '.')
            .unwrap_or(rest.len());
        let factor = match &rest[..unit_len] {
            "h" => 3600.0,
            "m" => 60.0,
            "s" => 1.0,
            "ms" => 0.001,
            "us" | "µs" => 0.000_001,
            "ns" => 0.000_000_001,
            _ => return None,
        };
        rest = &rest[unit_len..];
        total += value * factor;
    }
    Some(Duration::from_secs_f64(total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;
    use serde_json::json;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_static(v));
        }
        h
    }

    const ALL: RequestFeatures = RequestFeatures {
        images: true,
        tools: true,
    };

    fn kind_of(status: u16, body: &str, features: RequestFeatures) -> OpenAIApiErrorKind {
        match classify_http_error(
            StatusCode::from_u16(status).unwrap(),
            &HeaderMap::new(),
            body.to_string(),
            features,
        ) {
            OpenAIError::Api { kind, .. } => kind,
            other => panic!("expected Api, got {other:?}"),
        }
    }

    #[test]
    fn parses_go_durations() {
        assert_eq!(parse_go_duration("1s"), Some(Duration::from_secs(1)));
        assert_eq!(parse_go_duration("6m0s"), Some(Duration::from_secs(360)));
        assert_eq!(parse_go_duration("20ms"), Some(Duration::from_millis(20)));
        assert_eq!(
            parse_go_duration("1h2m3.5s"),
            Some(Duration::from_secs_f64(3723.5))
        );
        assert_eq!(parse_go_duration("soon"), None);
        assert_eq!(parse_go_duration("5"), None);
    }

    #[test]
    fn retry_headers_in_priority_order() {
        assert_eq!(
            parse_retry_after(&headers(&[
                ("retry-after-ms", "1500"),
                ("retry-after", "3")
            ])),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(
            parse_retry_after(&headers(&[("retry-after", "7")])),
            Some(Duration::from_secs(7))
        );
        // HTTP-date form is not parsed.
        assert_eq!(
            parse_retry_after(&headers(&[(
                "retry-after",
                "Wed, 21 Oct 2015 07:28:00 GMT"
            )])),
            None
        );
        assert_eq!(parse_retry_after(&HeaderMap::new()), None);
    }

    #[test]
    fn openai_reset_headers_follow_the_exhausted_limit() {
        let h = headers(&[
            ("x-ratelimit-remaining-requests", "0"),
            ("x-ratelimit-remaining-tokens", "1200"),
            ("x-ratelimit-reset-requests", "2s"),
            ("x-ratelimit-reset-tokens", "6m0s"),
        ]);
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(2)));
        let h = headers(&[
            ("x-ratelimit-remaining-tokens", "0"),
            ("x-ratelimit-reset-requests", "2s"),
            ("x-ratelimit-reset-tokens", "1m30s"),
        ]);
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(90)));
        let h = headers(&[
            ("x-ratelimit-reset-requests", "2s"),
            ("x-ratelimit-reset-tokens", "1m"),
        ]);
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(60)));
    }

    #[test]
    fn error_body_shapes() {
        assert_eq!(
            parse_error_fields(&json!({"error":{"message":"m","type":"t","code":"c"}})),
            ErrorFields {
                message: Some("m".into()),
                code: Some("c".into()),
                error_type: Some("t".into()),
            }
        );
        // OpenRouter: numeric code.
        assert_eq!(
            parse_error_fields(&json!({"error":{"code":401,"message":"No auth"}})).code,
            Some("401".into())
        );
        assert_eq!(
            parse_error_fields(&json!({"error":"Model not loaded"})).message,
            Some("Model not loaded".into())
        );
        assert_eq!(
            parse_error_fields(&json!({"detail":"Not Found"})).message,
            Some("Not Found".into())
        );
    }

    #[test]
    fn status_decides_auth_and_rate_limit() {
        let openai_401 = r#"{"error":{"message":"Incorrect API key provided: sk-...","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}"#;
        assert_eq!(
            kind_of(401, openai_401, ALL),
            OpenAIApiErrorKind::Authentication
        );
        let openrouter_401 = r#"{"error":{"code":401,"message":"No auth credentials found"}}"#;
        assert_eq!(
            kind_of(401, openrouter_401, ALL),
            OpenAIApiErrorKind::Authentication
        );
        assert_eq!(kind_of(403, "", ALL), OpenAIApiErrorKind::Permission);
        let limit = r#"{"error":{"message":"Rate limit reached for gpt-4o","type":"requests","code":"rate_limit_exceeded"}}"#;
        assert_eq!(kind_of(429, limit, ALL), OpenAIApiErrorKind::RateLimit);
        assert_eq!(
            kind_of(429, "<html>slow down</html>", ALL),
            OpenAIApiErrorKind::RateLimit
        );
    }

    #[test]
    fn quota_is_not_a_rate_limit() {
        let quota = r#"{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","param":null,"code":"insufficient_quota"}}"#;
        assert_eq!(kind_of(429, quota, ALL), OpenAIApiErrorKind::QuotaExceeded);
        let credits = r#"{"error":{"code":402,"message":"Insufficient credits"}}"#;
        assert_eq!(
            kind_of(402, credits, ALL),
            OpenAIApiErrorKind::QuotaExceeded
        );
    }

    #[test]
    fn capability_messages_from_each_server() {
        let ollama_tools = r#"{"error":{"message":"registry.ollama.ai/library/gemma2:2b does not support tools","type":"api_error","param":null,"code":null}}"#;
        assert_eq!(
            kind_of(400, ollama_tools, ALL),
            OpenAIApiErrorKind::UnsupportedTools
        );
        let openrouter_images =
            r#"{"error":{"message":"No endpoints found that support image input","code":404}}"#;
        assert_eq!(
            kind_of(404, openrouter_images, ALL),
            OpenAIApiErrorKind::UnsupportedImages
        );
        let openai_images = r#"{"error":{"message":"Invalid content type. image_url is only supported by certain models.","type":"invalid_request_error","param":"messages.[0].content.[1].type","code":null}}"#;
        assert_eq!(
            kind_of(400, openai_images, ALL),
            OpenAIApiErrorKind::UnsupportedImages
        );
        let vllm_images = r#"{"object":"error","message":"gpt2 is not a multimodal model","type":"BadRequestError","code":400}"#;
        assert_eq!(
            kind_of(400, vllm_images, ALL),
            OpenAIApiErrorKind::UnsupportedImages
        );
    }

    #[test]
    fn capability_messages_need_the_feature_in_the_request() {
        let ollama_tools =
            r#"{"error":{"message":"llama2 does not support tools","type":"api_error"}}"#;
        assert_eq!(
            kind_of(400, ollama_tools, RequestFeatures::default()),
            OpenAIApiErrorKind::InvalidRequest
        );
    }

    #[test]
    fn model_not_found_and_context_overflow() {
        let ollama = r#"{"error":{"message":"model \"llama9\" not found, try pulling it first","type":"api_error","param":null,"code":null}}"#;
        assert_eq!(kind_of(404, ollama, ALL), OpenAIApiErrorKind::ModelNotFound);
        let openai = r#"{"error":{"message":"The model `gpt-9` does not exist or you do not have access to it.","type":"invalid_request_error","param":null,"code":"model_not_found"}}"#;
        assert_eq!(kind_of(404, openai, ALL), OpenAIApiErrorKind::ModelNotFound);
        assert_eq!(
            kind_of(404, r#"{"detail":"Not Found"}"#, ALL),
            OpenAIApiErrorKind::NotFound
        );
        let overflow = r#"{"error":{"message":"This model's maximum context length is 128000 tokens.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}"#;
        assert_eq!(
            kind_of(400, overflow, ALL),
            OpenAIApiErrorKind::ContextOverflow
        );
        let llama_cpp = r#"{"error":{"code":400,"message":"the request exceeds the available context size, try increasing it","type":"exceed_context_size_error"}}"#;
        assert_eq!(
            kind_of(400, llama_cpp, ALL),
            OpenAIApiErrorKind::ContextOverflow
        );
    }

    #[test]
    fn server_errors_and_fallbacks() {
        assert_eq!(
            kind_of(503, "<html>overloaded</html>", ALL),
            OpenAIApiErrorKind::ServerError
        );
        assert_eq!(
            kind_of(400, r#"{"error":{"message":"bad field"}}"#, ALL),
            OpenAIApiErrorKind::InvalidRequest
        );
        let err = classify_http_error(
            StatusCode::BAD_GATEWAY,
            &HeaderMap::new(),
            "upstream down".into(),
            ALL,
        );
        match err {
            OpenAIError::Api { message, .. } => assert_eq!(message, "upstream down"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn rate_limit_error_carries_retry_delay() {
        let err = classify_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after", "30")]),
            "{}".into(),
            ALL,
        );
        match err {
            OpenAIError::Api { retry_after, .. } => {
                assert_eq!(retry_after, Some(Duration::from_secs(30)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn request_too_large_from_status() {
        assert_eq!(
            kind_of(413, "<html>too big</html>", ALL),
            OpenAIApiErrorKind::RequestTooLarge
        );
    }

    // Azure OpenAI bodies, ported from the former `ailoop-azure-openai`.

    #[test]
    fn azure_rate_limit_with_retry_after_ms() {
        let body =
            r#"{"error":{"code":"rate_limit_exceeded","message":"slow down","type":"requests"}}"#;
        let err = classify_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after-ms", "2500"), ("retry-after", "3")]),
            body.into(),
            ALL,
        );
        match err {
            OpenAIError::Api {
                status,
                kind,
                message,
                retry_after,
                ..
            } => {
                assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
                assert_eq!(kind, OpenAIApiErrorKind::RateLimit);
                assert_eq!(message, "slow down");
                assert_eq!(retry_after, Some(Duration::from_millis(2500)));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn azure_deployment_not_found_is_model_not_found() {
        let body = r#"{"error":{"code":"DeploymentNotFound","message":"The API deployment for this resource does not exist."}}"#;
        assert_eq!(kind_of(404, body, ALL), OpenAIApiErrorKind::ModelNotFound);
        let terse = r#"{"error":{"code":"DeploymentNotFound","message":"no such thing"}}"#;
        assert_eq!(kind_of(404, terse, ALL), OpenAIApiErrorKind::ModelNotFound);
    }

    #[test]
    fn azure_content_filter() {
        let body = r#"{"error":{"message":"The response was filtered due to the prompt triggering Azure OpenAI's content management policy.","type":null,"param":"prompt","code":"content_filter","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation"}}}"#;
        assert_eq!(kind_of(400, body, ALL), OpenAIApiErrorKind::ContentFilter);
        let simple = r#"{"error":{"code":"content_filter","message":"blocked"}}"#;
        assert_eq!(kind_of(400, simple, ALL), OpenAIApiErrorKind::ContentFilter);
    }

    /// Bodies Azure returns when the input exceeds the deployment's
    /// context window, as reported on Microsoft Q&A
    /// (<https://learn.microsoft.com/en-us/answers/questions/2280883/azure-openai-model-gpt-4-1-context-window-exceeded>)
    /// and in <https://github.com/Azure/azure-sdk-for-python/issues/40986>.
    #[test]
    fn azure_context_length_exceeded() {
        let input_exceeds = r#"{"error":{"message":"Your input exceeds the context window of this model. Please adjust your input and try again.","type":"invalid_request_error","param":"input","code":"context_length_exceeded"}}"#;
        let max_context = r#"{"error":{"message":"This model's maximum context length is 128000 tokens. However, you requested 1124171 tokens (124171 in the messages, 1000000 in the completion). Please reduce the length of the messages or completion.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}"#;
        for body in [input_exceeds, max_context] {
            assert_eq!(kind_of(400, body, ALL), OpenAIApiErrorKind::ContextOverflow);
        }
    }

    #[test]
    fn azure_codes_without_a_decisive_status() {
        let invalid = r#"{"error":{"message":"Unrecognized request argument supplied: foo","type":"invalid_request_error","param":null,"code":"invalid_request_error"}}"#;
        assert_eq!(
            kind_of(400, invalid, ALL),
            OpenAIApiErrorKind::InvalidRequest
        );
        let future = r#"{"error":{"code":"FutureCode","message":"x"}}"#;
        assert_eq!(
            kind_of(418, future, ALL),
            OpenAIApiErrorKind::Other("FutureCode".into())
        );
        // Older Azure API versions omit `code`.
        let no_code = r#"{"error":{"message":"some message"}}"#;
        assert_eq!(
            kind_of(418, no_code, ALL),
            OpenAIApiErrorKind::Other(String::new())
        );
    }

    #[test]
    fn azure_front_door_html_keeps_status_and_body() {
        let err = classify_http_error(
            StatusCode::BAD_GATEWAY,
            &HeaderMap::new(),
            "<html>Azure Front Door 502</html>".into(),
            ALL,
        );
        match err {
            OpenAIError::Api {
                status,
                kind,
                message,
                ..
            } => {
                assert_eq!(status, StatusCode::BAD_GATEWAY);
                assert_eq!(kind, OpenAIApiErrorKind::ServerError);
                assert_eq!(message, "<html>Azure Front Door 502</html>");
            }
            other => panic!("{other:?}"),
        }
    }
}
