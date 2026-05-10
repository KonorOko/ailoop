use crate::client::AnthropicClient;
use crate::error_body::{classify_http_error, parse_retry_after};
use crate::errors::AnthropicError;
use crate::request::build_body;
use crate::stream::process_response;

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
pub struct AnthropicModel {
    client: AnthropicClient,
    model: String,
}

impl AnthropicModel {
    /// Bind `client` to a specific `model` id. Prefer
    /// [`AnthropicClient::model`] for the one-client-one-model case
    /// — this constructor is for adapter authors composing their own
    /// pipelines.
    pub fn new(client: AnthropicClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

#[async_trait::async_trait]
impl CompletionModel for AnthropicModel {
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
        let body = build_body(&self.model, &req);

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
