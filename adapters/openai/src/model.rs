use ailoop_core::{ChatRequest, CompletionModel, StreamChunk};
use futures::stream::BoxStream;

use crate::client::OpenAIClient;
use crate::error_body::classify_http_error;
use crate::errors::OpenAIError;
use crate::request::{BodyOptions, build_body, request_features};
use crate::stream::process_response;

/// Chat Completions model bound to one model id. Built with
/// [`OpenAIClient::model`] or
/// [`CompletionClient::completion_model`](ailoop_core::CompletionClient::completion_model).
///
/// Reasoning the model streams (`reasoning` / `reasoning_content`, sent
/// by Ollama, LM Studio, OpenRouter and DeepSeek) arrives as
/// [`StreamChunk::ReasoningDelta`]; reasoning blocks are dropped when the
/// history is sent back, because Chat Completions has no slot for them.
#[derive(Clone, Debug)]
pub struct OpenAIChatModel {
    client: OpenAIClient,
    model: String,
}

impl OpenAIChatModel {
    /// Bind `client` to `model`.
    pub fn new(client: OpenAIClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// The client this model sends through.
    pub fn client(&self) -> &OpenAIClient {
        &self.client
    }
}

#[async_trait::async_trait]
impl CompletionModel for OpenAIChatModel {
    type Error = OpenAIError;

    /// `"azure-openai"` for a client built with
    /// [`OpenAIClient::azure`], `"openai"` otherwise.
    fn name(&self) -> &str {
        if self.client.is_azure() {
            "azure-openai"
        } else {
            "openai"
        }
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, OpenAIError>>, OpenAIError> {
        let url = self.client.endpoint("chat/completions");
        let options = BodyOptions {
            max_tokens_field: self.client.max_tokens_field,
            stream_usage: self.client.stream_usage,
        };
        let body = build_body(&self.model, &req, &options)?;
        let features = request_features(&req);

        let builder = self
            .client
            .authorize(self.client.http_client.post(&url))
            .await?
            .header("accept", "text/event-stream")
            .json(&body);
        let response = builder
            .send()
            .await
            .map_err(|e| self.client.send_error(&url, e))?;

        if !response.status().is_success() {
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.text().await.unwrap_or_default();
            return Err(classify_http_error(status, &headers, body, features));
        }

        Ok(process_response(response, features))
    }
}
