use std::time::{Duration, SystemTime};

use ailoop_core::{ProviderError, RetryClassification, Retryable};
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
    /// available to this key.
    ModelNotFound,
    /// Other 404.
    NotFound,
    /// The provider's moderation blocked the request. Permanent.
    ContentFilter,
    /// Other 400/422: a malformed or invalid request. Permanent.
    InvalidRequest,
    /// 5xx, or a 503/529 overload. Transient.
    ServerError,
    /// Anything else, with the code or type the server sent (empty when
    /// it sent none).
    Other(String),
}

/// One thing the model may not support, for
/// [`ErrorCategory::ModelUnsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Capability {
    /// Image input.
    Images,
    /// Tool (function) calling.
    Tools,
}

/// What went wrong, in the terms an app shows to its user. Built by
/// [`OpenAIError::category`]; one value per message the app needs.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorCategory {
    /// A remote server could not be reached: no network, DNS failure,
    /// connection refused or timed out.
    Offline {
        /// The URL that failed.
        url: String,
    },
    /// A server on this machine or the local network did not accept the
    /// connection: Ollama or LM Studio is not running, or listens
    /// elsewhere.
    LocalServiceUnreachable {
        /// The URL that failed.
        url: String,
    },
    /// The server rejected the key (401) or its permissions (403).
    AuthRejected {
        /// 401 or 403.
        status: StatusCode,
    },
    /// The provider's rate limit (429). Retry after the delay when
    /// there is one.
    RateLimited {
        /// How long the server asked to wait, from `retry-after-ms`,
        /// `retry-after` or OpenAI's `x-ratelimit-reset-*` headers.
        retry_after: Option<Duration>,
        /// `retry_after` added to the time the response arrived.
        retry_at: Option<SystemTime>,
    },
    /// No quota or credit left on the account.
    QuotaExhausted,
    /// The model does not accept images or tools.
    ModelUnsupported(Capability),
    /// The model does not exist on this server.
    ModelNotFound,
    /// The conversation no longer fits the model's context window.
    ContextOverflow,
    /// The response stopped before it was complete: the connection
    /// dropped, or the stream ended without saying why the model
    /// stopped. Text received so far is incomplete.
    Interrupted,
    /// Anything else. The error's `Display` describes it.
    Other,
}

/// Failure surface of [`OpenAIChatModel`](crate::OpenAIChatModel) and
/// [`OpenAIClient::list_models`](crate::OpenAIClient::list_models).
///
/// Use [`category`](Self::category) to decide what to show; match the
/// variants when you need the details.
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
        /// How long the server asked to wait before retrying.
        retry_after: Option<Duration>,
        /// `retry_after` added to the time the response arrived.
        retry_at: Option<SystemTime>,
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
    /// header).
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

impl OpenAIError {
    /// Classify the error for display. See [`ErrorCategory`].
    pub fn category(&self) -> ErrorCategory {
        match self {
            OpenAIError::Unreachable {
                url, local: true, ..
            } => ErrorCategory::LocalServiceUnreachable { url: url.clone() },
            OpenAIError::Unreachable { url, .. } => ErrorCategory::Offline { url: url.clone() },
            OpenAIError::Api {
                status,
                kind,
                retry_after,
                retry_at,
                ..
            } => kind_category(kind, Some(*status), *retry_after, *retry_at),
            OpenAIError::Provider { kind, .. } => kind_category(kind, None, None, None),
            OpenAIError::Interrupted { .. } => ErrorCategory::Interrupted,
            OpenAIError::Http(_)
            | OpenAIError::Sse(_)
            | OpenAIError::Json(_)
            | OpenAIError::Config(_)
            | OpenAIError::UnsupportedContent { .. } => ErrorCategory::Other,
        }
    }

    /// The server's retry delay, for a rate limit that came with one.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            OpenAIError::Api { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

fn kind_category(
    kind: &OpenAIApiErrorKind,
    status: Option<StatusCode>,
    retry_after: Option<Duration>,
    retry_at: Option<SystemTime>,
) -> ErrorCategory {
    match kind {
        OpenAIApiErrorKind::Authentication => ErrorCategory::AuthRejected {
            status: status.unwrap_or(StatusCode::UNAUTHORIZED),
        },
        OpenAIApiErrorKind::Permission => ErrorCategory::AuthRejected {
            status: status.unwrap_or(StatusCode::FORBIDDEN),
        },
        OpenAIApiErrorKind::RateLimit => ErrorCategory::RateLimited {
            retry_after,
            retry_at,
        },
        OpenAIApiErrorKind::QuotaExceeded => ErrorCategory::QuotaExhausted,
        OpenAIApiErrorKind::UnsupportedImages => {
            ErrorCategory::ModelUnsupported(Capability::Images)
        }
        OpenAIApiErrorKind::UnsupportedTools => ErrorCategory::ModelUnsupported(Capability::Tools),
        OpenAIApiErrorKind::ContextOverflow => ErrorCategory::ContextOverflow,
        OpenAIApiErrorKind::ModelNotFound => ErrorCategory::ModelNotFound,
        OpenAIApiErrorKind::NotFound
        | OpenAIApiErrorKind::ContentFilter
        | OpenAIApiErrorKind::InvalidRequest
        | OpenAIApiErrorKind::ServerError
        | OpenAIApiErrorKind::Other(_) => ErrorCategory::Other,
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
    /// `true` for [`OpenAIApiErrorKind::ContextOverflow`], from the HTTP
    /// response or from an event mid-stream.
    fn is_context_overflow(&self) -> bool {
        matches!(
            self,
            OpenAIError::Api {
                kind: OpenAIApiErrorKind::ContextOverflow,
                ..
            } | OpenAIError::Provider {
                kind: OpenAIApiErrorKind::ContextOverflow,
                ..
            }
        )
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
            retry_at: None,
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
        assert_eq!(quota.category(), ErrorCategory::QuotaExhausted);
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
    fn categories_cover_auth_and_capabilities() {
        assert_eq!(
            api(StatusCode::FORBIDDEN, OpenAIApiErrorKind::Permission).category(),
            ErrorCategory::AuthRejected {
                status: StatusCode::FORBIDDEN
            }
        );
        assert_eq!(
            api(
                StatusCode::BAD_REQUEST,
                OpenAIApiErrorKind::UnsupportedTools
            )
            .category(),
            ErrorCategory::ModelUnsupported(Capability::Tools)
        );
        assert!(
            api(StatusCode::BAD_REQUEST, OpenAIApiErrorKind::ContextOverflow).is_context_overflow()
        );
    }
}
