use crate::client::AnthropicClient;
use crate::errors::AnthropicError;
use crate::request::build_body;
use crate::stream::process_response;

use ailoop_core::{ChatRequest, CompletionModel, StreamChunk};
use futures::stream::BoxStream;

#[derive(Clone)]
pub struct AnthropicModel {
    client: AnthropicClient,
    model: String,
}

impl AnthropicModel {
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

        let response = self
            .client
            .http_client
            .post(&self.client.base_url)
            .header("x-api-key", &self.client.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AnthropicError::Status { status, body });
        }

        Ok(process_response(response))
    }
}
