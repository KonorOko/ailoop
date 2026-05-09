use reqwest::StatusCode;

#[derive(Debug, thiserror::Error)]
pub enum AzureOpenAIError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// 4xx / 5xx
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
