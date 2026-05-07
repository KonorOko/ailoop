use futures::stream::BoxStream;

use crate::{request::ChatRequest, stream::StreamChunk};

pub trait CompletionClient {
    type Model: CompletionModel;

    fn completion_model(&self, model_name: impl Into<String>) -> Self::Model;
}

#[async_trait::async_trait]
pub trait CompletionModel {
    type Error: std::error::Error + Send + Sync + 'static;

    fn name(&self) -> &str;
    fn model(&self) -> &str;

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error>;
}
