use ailoop_core::{ChatRequest, CompletionModel, StreamChunk};
use futures::stream::BoxStream;

use crate::client::{AzureOpenAIAuth, AzureOpenAIClient};
use crate::error_body::{classify_http_error, parse_retry_after};
use crate::errors::AzureOpenAIError;
use crate::request::build_body;
use crate::stream::process_response;

/// Chat Completions model bound to a specific deployment. Constructed via
/// `AzureOpenAIClient::model(...)` or `CompletionClient::completion_model(...)`.
#[derive(Clone)]
pub struct AzureOpenAIChatModel {
    client: AzureOpenAIClient,
    deployment: String,
}

impl AzureOpenAIChatModel {
    pub fn new(client: AzureOpenAIClient, deployment: impl Into<String>) -> Self {
        Self {
            client,
            deployment: deployment.into(),
        }
    }
}

#[async_trait::async_trait]
impl CompletionModel for AzureOpenAIChatModel {
    type Error = AzureOpenAIError;

    fn name(&self) -> &str {
        "azure-openai"
    }

    fn model(&self) -> &str {
        &self.deployment
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, AzureOpenAIError>>, AzureOpenAIError> {
        let url = format!(
            "{}/openai/v1/chat/completions",
            self.client.endpoint.trim_end_matches('/'),
        );

        let body = build_body(&self.deployment, &req);

        let mut req_builder = self
            .client
            .http_client
            .post(&url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");

        match &self.client.auth {
            AzureOpenAIAuth::ApiKey(k) => {
                req_builder = req_builder.header("api-key", k);
            }
            AzureOpenAIAuth::Token(t) => {
                req_builder = req_builder.header("Authorization", format!("Bearer {t}"));
            }
            AzureOpenAIAuth::Provider(provider) => {
                let token = provider.token().await?;
                req_builder = req_builder.header("Authorization", format!("Bearer {token}"));
            }
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
