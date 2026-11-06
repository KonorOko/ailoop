use crate::client::AnthropicClient;
use crate::error_body::{classify_http_error, parse_retry_after};
use crate::errors::AnthropicError;
use crate::request::build_body;
use crate::stream::process_response;
use crate::thinking::{Shape, ThinkingMode};

use ailoop_core::{ChatRequest, CompletionModel, StreamChunk};
use futures::stream::BoxStream;

/// [`CompletionModel`] implementation bound to a specific Anthropic
/// model id (e.g. `"claude-sonnet-4-6"`). Constructed via
/// [`AnthropicClient::model`] or
/// [`CompletionClient::completion_model`](ailoop_core::CompletionClient::completion_model).
///
/// Honours every per-block [`CacheControl`](ailoop_core::CacheControl)
/// supplied on the [`ChatRequest`] (system / messages / tools), emits
/// the TTL-broken-down `cache_creation_5m_tokens` /
/// `cache_creation_1h_tokens` counters on
/// [`StreamChunk::TurnFinished`], and round-trips reasoning blocks via
/// [`AssistantBlock::Reasoning`](ailoop_core::AssistantBlock::Reasoning).
///
/// [`CompletionClient`]: ailoop_core::CompletionClient
/// [`StreamChunk::TurnFinished`]: ailoop_core::StreamChunk::TurnFinished
#[derive(Clone)]
pub struct AnthropicChatModel {
    client: AnthropicClient,
    model: String,
    thinking_mode: Option<ThinkingMode>,
}

impl AnthropicChatModel {
    /// Bind `client` to a specific `model` id. Prefer
    /// [`AnthropicClient::model`] for the one-client-one-model case
    /// — this constructor is for adapter authors composing their own
    /// pipelines.
    pub fn new(client: AnthropicClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            thinking_mode: None,
        }
    }

    /// Force the wire shape used for
    /// [`ChatRequest::reasoning_effort`](ailoop_core::ChatRequest::reasoning_effort).
    ///
    /// Without this, the shape comes from the model id: Claude 4.5 and
    /// earlier (`claude-haiku-4-5`, `claude-sonnet-4-5`,
    /// `claude-opus-4-5`, `claude-opus-4-1`, `claude-*-4-0`) and the
    /// whole `claude-3-*` family use [`ThinkingMode::Budget`]; every
    /// other id, including ones the adapter does not know, uses
    /// [`ThinkingMode::Adaptive`]. Set it for an id the list cannot
    /// read, such as an alias behind a proxy.
    pub fn thinking_mode(mut self, mode: ThinkingMode) -> Self {
        self.thinking_mode = Some(mode);
        self
    }
}

#[async_trait::async_trait]
impl CompletionModel for AnthropicChatModel {
    type Error = AnthropicError;

    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, AnthropicError>>, AnthropicError> {
        let shape = Shape::resolve(&self.model, self.thinking_mode);
        let body = build_body(&self.model, &req, shape);

        let mut req_builder = self
            .client
            .http_client
            .post(&self.client.base_url)
            .header("x-api-key", &self.client.api_key)
            .header("anthropic-version", &self.client.anthropic_version)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");

        if !self.client.beta_features.is_empty() {
            req_builder = req_builder.header("anthropic-beta", self.client.beta_features.join(","));
        }

        let response = req_builder.json(&body).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers());
            let body = response.text().await.unwrap_or_default();
            return Err(classify_http_error(status, body, retry_after));
        }

        Ok(process_response(response))
    }
}
