use std::time::Duration;

use ailoop_core::{Capability, ProviderError, ProviderErrorKind, RetryClassification, Retryable};
use reqwest::StatusCode;

/// Typed category of an error the server reported, from the HTTP
/// status first and then from the body's `error.code`, `error.type` and
/// message.
///
/// Compatible servers disagree on the error body: OpenAI sends string
/// codes, OpenRouter numeric ones, Ollama and LM Studio often only a
/// message. The status decides the broad category so that every server
/// lands somewhere sensible; the body refines it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpenAIApiErrorKind {
    /// 401: missing, invalid or revoked API key. Permanent.
    Authentication,
    /// 403: the key is valid but may not use this model or endpoint.
    /// Permanent.
    Permission,
    /// 429 rate limit: too many requests or tokens per minute.
    /// Transient; the error carries the delay the server asked for.
    RateLimit,
    /// The account has no quota or credit left: OpenAI's 429
    /// `insufficient_quota`, OpenRouter's 402. Permanent: waiting does
    /// not help, the user has to add credit or change plan.
    QuotaExceeded,
    /// The model does not accept image input. Recognized from the
    /// message, and only when the request carried an image.
    UnsupportedImages,
    /// The model does not accept tools. Recognized from the message,
    /// and only when the request carried tools.
    UnsupportedTools,
    /// The prompt does not fit the model's context window
    /// (`context_length_exceeded` and the messages local servers use).
    /// Permanent; compact the history and resend.
    ContextOverflow,
    /// 404 naming the model: not installed locally, misspelled, or not
    /// available to this key. Azure's `DeploymentNotFound` lands here
    /// too, since on Azure the deployment is the model.
    ModelNotFound,
    /// Other 404.
    NotFound,
    /// The provider's moderation blocked the request. Permanent.
    ContentFilter,
    /// 413: the request body is too large. Permanent.
    RequestTooLarge,
    /// Other 400/422: a malformed or invalid request. Permanent.
    InvalidRequest,
    /// 5xx, or a 503/529 overload. Transient.
    ServerError,
    /// Anything else, with the code or type the server sent (empty when
    /// it sent none).
    Other(String),
}

/// Failure surface of [`OpenAIChatModel`](crate::OpenAIChatModel) and
/// [`OpenAIClient::list_models`](crate::OpenAIClient::list_models).
///
/// Use [`ProviderError::kind`] to decide what to show; match the
/// variants when you need the details. [`Retryable`] gives the retry
/// decision and the delay the server asked for.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OpenAIError {
    /// No connection was made: DNS failure, connection refused, or the
    /// connect timeout passed. `local` tells a local server that is not
    /// running apart from a missing network.
    #[error("could not reach {url}: {source}")]
    #[non_exhaustive]
    Unreachable {
        /// The URL of the failed request.
        url: String,
        /// Whether the URL points at this machine or the local network
        /// (see [`OpenAIClient::is_local`](crate::OpenAIClient::is_local)).
        local: bool,
        /// The transport error.
        #[source]
        source: reqwest::Error,
    },

    /// Another transport failure from `reqwest` before the response
    /// arrived (TLS, a reset during the request).
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// The server answered with a 4xx/5xx status.
    #[error("API error ({status}, {kind:?}): {message}")]
    #[non_exhaustive]
    Api {
        /// HTTP status.
        status: StatusCode,
        /// Typed category.
        kind: OpenAIApiErrorKind,
        /// The server's message, or the raw body when it sent no JSON
        /// error envelope.
        message: String,
        /// `error.code` (or `error.type`) as the server sent it, numbers
        /// included, for logs.
        code: Option<String>,
        /// How long the server asked to wait before retrying, from
        /// `retry-after-ms`, `retry-after` or OpenAI's
        /// `x-ratelimit-reset-*` headers.
        retry_after: Option<Duration>,
    },

    /// The server sent an error event in the middle of the stream
    /// (`data: {"error":{...}}`). No headers at this point, so no retry
    /// delay.
    #[error("error event in stream ({kind:?}): {message}")]
    #[non_exhaustive]
    Provider {
        /// Typed category, from the event's code, type and message.
        kind: OpenAIApiErrorKind,
        /// The event's message.
        message: String,
    },

    /// The response stopped before it was complete: the connection
    /// dropped mid-body, or the stream ended without a `finish_reason`.
    #[error("response interrupted: {message}")]
    #[non_exhaustive]
    Interrupted {
        /// What was observed.
        message: String,
        /// The transport error, when there was one.
        #[source]
        source: Option<reqwest::Error>,
    },

    /// The stream was not valid Server-Sent Events (bad UTF-8 or
    /// framing). Permanent.
    #[error("SSE parse error: {0}")]
    Sse(String),

    /// A JSON payload did not have the expected shape. Permanent.
    #[error("malformed payload: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid client configuration (missing key for OpenAI, invalid
    /// header, missing `AZURE_OPENAI_*` variable), or a
    /// [`TokenProvider`](crate::TokenProvider) that could not fetch a
    /// token.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// The request carried content Chat Completions cannot represent: a
    /// [`ailoop_core::UserBlock::Document`], an image inside a tool
    /// result, or an image given by `Source::FileId`. Raised before any
    /// HTTP call. `kind` is `"document"`, `"tool_result_image"`,
    /// `"image_file_id"` or `"image_source"`.
    #[error("unsupported content for Chat Completions: {kind}")]
    UnsupportedContent {
        /// Short stable label of the shape that could not be encoded.
        kind: &'static str,
    },
}

fn provider_kind(kind: &OpenAIApiErrorKind) -> ProviderErrorKind {
    match kind {
        OpenAIApiErrorKind::Authentication => ProviderErrorKind::Auth,
        OpenAIApiErrorKind::Permission => ProviderErrorKind::Permission,
        OpenAIApiErrorKind::RateLimit => ProviderErrorKind::RateLimited,
        OpenAIApiErrorKind::QuotaExceeded => ProviderErrorKind::QuotaExhausted,
        OpenAIApiErrorKind::UnsupportedImages => ProviderErrorKind::Unsupported(Capability::Images),
        OpenAIApiErrorKind::UnsupportedTools => ProviderErrorKind::Unsupported(Capability::Tools),
        OpenAIApiErrorKind::ContextOverflow => ProviderErrorKind::ContextOverflow,
        OpenAIApiErrorKind::ModelNotFound => ProviderErrorKind::ModelNotFound,
        OpenAIApiErrorKind::ContentFilter => ProviderErrorKind::ContentFilter,
        OpenAIApiErrorKind::RequestTooLarge => ProviderErrorKind::RequestTooLarge,
        OpenAIApiErrorKind::InvalidRequest => ProviderErrorKind::InvalidRequest,
        OpenAIApiErrorKind::ServerError => ProviderErrorKind::Unavailable,
        OpenAIApiErrorKind::NotFound | OpenAIApiErrorKind::Other(_) => ProviderErrorKind::Other,
    }
}

fn classify_kind(
    kind: &OpenAIApiErrorKind,
    status: Option<StatusCode>,
    retry_after: Option<Duration>,
) -> RetryClassification {
    match kind {
        OpenAIApiErrorKind::RateLimit | OpenAIApiErrorKind::ServerError => {
            RetryClassification::Transient { retry_after }
        }
        // An unknown code is retried only when the status says the
        // server, not the request, is at fault. Without a status (an
        // event mid-stream) it is retried, like Azure's adapter does.
        OpenAIApiErrorKind::Other(_) => match status {
            Some(s)
                if !(s.is_server_error()
                    || s == StatusCode::REQUEST_TIMEOUT
                    || s == StatusCode::CONFLICT) =>
            {
                RetryClassification::Permanent
            }
            _ => RetryClassification::Transient { retry_after },
        },
        OpenAIApiErrorKind::Authentication
        | OpenAIApiErrorKind::Permission
        | OpenAIApiErrorKind::QuotaExceeded
        | OpenAIApiErrorKind::UnsupportedImages
        | OpenAIApiErrorKind::UnsupportedTools
        | OpenAIApiErrorKind::ContextOverflow
        | OpenAIApiErrorKind::ModelNotFound
        | OpenAIApiErrorKind::NotFound
        | OpenAIApiErrorKind::ContentFilter
        | OpenAIApiErrorKind::RequestTooLarge
        | OpenAIApiErrorKind::InvalidRequest => RetryClassification::Permanent,
    }
}

impl Retryable for OpenAIError {
    fn retry_classification(&self) -> RetryClassification {
        match self {
            // A local server that refuses the connection is not running;
            // retrying only delays the message. A remote one may be a
            // network blip.
            OpenAIError::Unreachable { local: true, .. } => RetryClassification::Permanent,
            OpenAIError::Unreachable { local: false, .. } | OpenAIError::Http(_) => {
                RetryClassification::Transient { retry_after: None }
            }
            OpenAIError::Api {
                status,
                kind,
                retry_after,
                ..
            } => classify_kind(kind, Some(*status), *retry_after),
            OpenAIError::Provider { kind, .. } => classify_kind(kind, None, None),
            OpenAIError::Interrupted { .. } => RetryClassification::Transient { retry_after: None },
            OpenAIError::Sse(_)
            | OpenAIError::Json(_)
            | OpenAIError::Config(_)
            | OpenAIError::UnsupportedContent { .. } => RetryClassification::Permanent,
        }
    }
}

impl ProviderError for OpenAIError {
    /// - [`Unreachable`](OpenAIError::Unreachable) and
    ///   [`Http`](OpenAIError::Http) are [`Network`](ProviderErrorKind::Network).
    ///   Tell a local server that is not running from a missing network
    ///   with [`OpenAIClient::is_local`](crate::OpenAIClient::is_local).
    /// - [`Api`](OpenAIError::Api) and [`Provider`](OpenAIError::Provider)
    ///   follow their [`OpenAIApiErrorKind`]: `ServerError` is
    ///   [`Unavailable`](ProviderErrorKind::Unavailable), `NotFound` and
    ///   `Other` are [`Other`](ProviderErrorKind::Other), the rest map to
    ///   the variant of the same meaning.
    /// - [`Interrupted`](OpenAIError::Interrupted) is
    ///   [`Interrupted`](ProviderErrorKind::Interrupted).
    /// - [`UnsupportedContent`](OpenAIError::UnsupportedContent) is
    ///   [`InvalidRequest`](ProviderErrorKind::InvalidRequest).
    /// - [`Sse`](OpenAIError::Sse), [`Json`](OpenAIError::Json) and
    ///   [`Config`](OpenAIError::Config) are [`Other`](ProviderErrorKind::Other).
    fn kind(&self) -> ProviderErrorKind {
        match self {
            OpenAIError::Unreachable { .. } | OpenAIError::Http(_) => ProviderErrorKind::Network,
            OpenAIError::Api { kind, .. } | OpenAIError::Provider { kind, .. } => {
                provider_kind(kind)
            }
            OpenAIError::Interrupted { .. } => ProviderErrorKind::Interrupted,
            OpenAIError::UnsupportedContent { .. } => ProviderErrorKind::InvalidRequest,
            OpenAIError::Sse(_) | OpenAIError::Json(_) | OpenAIError::Config(_) => {
                ProviderErrorKind::Other
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api(status: StatusCode, kind: OpenAIApiErrorKind) -> OpenAIError {
        OpenAIError::Api {
            status,
            kind,
            message: "m".into(),
            code: None,
            retry_after: None,
        }
    }

    #[test]
    fn rate_limit_is_transient_and_quota_is_permanent() {
        let mut limited = api(StatusCode::TOO_MANY_REQUESTS, OpenAIApiErrorKind::RateLimit);
        if let OpenAIError::Api { retry_after, .. } = &mut limited {
            *retry_after = Some(Duration::from_secs(2));
        }
        assert_eq!(
            limited.retry_classification(),
            RetryClassification::Transient {
                retry_after: Some(Duration::from_secs(2))
            }
        );
        let quota = api(
            StatusCode::TOO_MANY_REQUESTS,
            OpenAIApiErrorKind::QuotaExceeded,
        );
        assert_eq!(quota.retry_classification(), RetryClassification::Permanent);
        assert_eq!(quota.kind(), ProviderErrorKind::QuotaExhausted);
    }

    #[test]
    fn unknown_kind_retries_only_on_server_status() {
        let client_side = api(
            StatusCode::IM_A_TEAPOT,
            OpenAIApiErrorKind::Other("x".into()),
        );
        assert_eq!(
            client_side.retry_classification(),
            RetryClassification::Permanent
        );
        let server_side = api(
            StatusCode::BAD_GATEWAY,
            OpenAIApiErrorKind::Other("x".into()),
        );
        assert!(matches!(
            server_side.retry_classification(),
            RetryClassification::Transient { .. }
        ));
    }

    #[test]
    fn kinds_follow_the_api_kind() {
        for (api_kind, expected) in [
            (OpenAIApiErrorKind::Authentication, ProviderErrorKind::Auth),
            (
                OpenAIApiErrorKind::Permission,
                ProviderErrorKind::Permission,
            ),
            (
                OpenAIApiErrorKind::RateLimit,
                ProviderErrorKind::RateLimited,
            ),
            (
                OpenAIApiErrorKind::UnsupportedTools,
                ProviderErrorKind::Unsupported(Capability::Tools),
            ),
            (
                OpenAIApiErrorKind::UnsupportedImages,
                ProviderErrorKind::Unsupported(Capability::Images),
            ),
            (
                OpenAIApiErrorKind::ContentFilter,
                ProviderErrorKind::ContentFilter,
            ),
            (
                OpenAIApiErrorKind::ModelNotFound,
                ProviderErrorKind::ModelNotFound,
            ),
            (
                OpenAIApiErrorKind::RequestTooLarge,
                ProviderErrorKind::RequestTooLarge,
            ),
            (
                OpenAIApiErrorKind::InvalidRequest,
                ProviderErrorKind::InvalidRequest,
            ),
            (
                OpenAIApiErrorKind::ServerError,
                ProviderErrorKind::Unavailable,
            ),
            (OpenAIApiErrorKind::NotFound, ProviderErrorKind::Other),
            (
                OpenAIApiErrorKind::Other("x".into()),
                ProviderErrorKind::Other,
            ),
        ] {
            assert_eq!(
                api(StatusCode::BAD_REQUEST, api_kind.clone()).kind(),
                expected,
                "{api_kind:?}"
            );
            let event = OpenAIError::Provider {
                kind: api_kind.clone(),
                message: "m".into(),
            };
            assert_eq!(event.kind(), expected, "{api_kind:?}");
        }
        assert!(
            api(StatusCode::BAD_REQUEST, OpenAIApiErrorKind::ContextOverflow).is_context_overflow()
        );
        assert!(
            !api(StatusCode::BAD_REQUEST, OpenAIApiErrorKind::InvalidRequest).is_context_overflow()
        );
        let interrupted = OpenAIError::Interrupted {
            message: "m".into(),
            source: None,
        };
        assert_eq!(interrupted.kind(), ProviderErrorKind::Interrupted);
        assert_eq!(
            OpenAIError::UnsupportedContent { kind: "document" }.kind(),
            ProviderErrorKind::InvalidRequest
        );
        assert_eq!(
            OpenAIError::Config("x".into()).kind(),
            ProviderErrorKind::Other
        );
    }
}
