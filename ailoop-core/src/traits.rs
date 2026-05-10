use futures::stream::BoxStream;

use crate::{request::ChatRequest, stream::StreamChunk};

/// Factory for [`CompletionModel`] handles bound to a provider
/// account / endpoint.
///
/// A `CompletionClient` typically holds the HTTP client, credentials,
/// and base URL; calling [`Self::completion_model`] fixes a specific
/// model identifier so it can be passed to the engine. Adapters with
/// multiple model families (e.g. Anthropic's chat models, Azure's Chat
/// Completions deployments) implement one client and produce one model
/// type.
pub trait CompletionClient {
    /// Concrete `CompletionModel` this client produces.
    type Model: CompletionModel;

    /// Build a model handle for `model_name` (Anthropic model id,
    /// Azure deployment name, etc.). Implementations clone whatever
    /// state the model needs out of the client.
    fn completion_model(&self, model_name: impl Into<String>) -> Self::Model;
}

/// Provider-agnostic streaming completion contract.
///
/// Implementors map [`ChatRequest`] to the provider's wire format,
/// open the streaming response, and translate per-chunk events into
/// [`StreamChunk`]s. Implementations must be `Send + Sync` because
/// [`RetryingModel`](crate::RetryingModel) and the engine hold them
/// across `await` boundaries; the trait does not declare these
/// super-bounds yet — every use site adds them — but new
/// implementations should satisfy them.
#[async_trait::async_trait]
pub trait CompletionModel {
    /// Error type for the model's transport / setup failures.
    /// Implement [`crate::Retryable`] on it to make the model
    /// composable with [`RetryingModel`](crate::RetryingModel).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Adapter family name used in telemetry, e.g. `"anthropic"` or
    /// `"azure-openai"`. Stable across model ids — changing model
    /// name does not change this.
    fn name(&self) -> &str;
    /// Model identifier this handle is bound to: Anthropic model id
    /// (`"claude-sonnet-4-5"`), Azure deployment name, etc. The
    /// engine surfaces this in tracing and the JSON tracer envelope.
    fn model(&self) -> &str;

    /// Open a streaming completion. Returns a stream of
    /// `Result<StreamChunk, Self::Error>` so per-chunk transport
    /// errors can be surfaced mid-stream. Setup-time failures (auth,
    /// schema, transport open) return the outer `Err`.
    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error>;
}
