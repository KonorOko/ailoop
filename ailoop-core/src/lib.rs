//! Foundation crate for the `ailoop` workspace: shared vocabulary used
//! by the `ailoop` façade and by every adapter / middleware crate.
//!
//! Most application code does not depend on `ailoop-core` directly —
//! it depends on `ailoop`, which re-exports the types you need to
//! build a `Conversation` and write a [`ChatMiddleware`]. Reach for
//! `ailoop-core` when:
//!
//! - implementing a new [`CompletionModel`] / [`CompletionClient`]
//!   adapter for a provider HTTP API;
//! - writing a generic middleware that should be reusable across
//!   façades, examples, or test harnesses;
//! - building reliability or observability infrastructure on top of
//!   the shared event vocabulary.
//!
//! ## Layout
//!
//! - [`Message`], [`UserBlock`], [`AssistantBlock`],
//!   [`ToolResultContent`], [`SystemPrompt`] — the wire model
//!   exchanged between the engine and the provider.
//! - [`ChatRequest`], [`ToolDefinition`], [`ToolChoice`], [`ToolTag`]
//!   — the per-turn request shape adapters lower to the provider.
//! - [`StreamChunk`], [`FinishReason`], [`Usage`] — the event stream
//!   the engine emits for every run.
//! - [`ChatMiddleware`], [`HookAction`], [`ToolDecision`] — the
//!   extension point for transforms, gating, and observation.
//! - [`CompletionClient`], [`CompletionModel`] — the contract every
//!   provider adapter implements.
//! - [`RetryingModel`], [`Retryable`], [`RetryConfig`] — backoff
//!   decorator that wraps any [`CompletionModel`].
//! - [`BoxedModel`], [`BoxedError`] — a [`CompletionModel`] with its
//!   type erased, for picking a provider at runtime.
//! - [`ProviderError`] — provider-agnostic classification of model
//!   errors ([`ProviderErrorKind`]) for callers above the model.

#![deny(missing_docs)]

mod background;
mod boxed;
mod config;
mod control;
mod ids;
mod message;
mod middleware;
mod provider_error;
mod request;
mod retry;
mod stream;
mod suspension;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod tokenizer;
mod traits;

pub use background::{BackgroundConfig, BackgroundToolResult};
pub use boxed::{BoxedError, BoxedModel};
pub use config::{DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_TOKENS, RunConfig};
pub use control::{BackgroundEnd, Drained, Inbound, RunControl};
pub use ids::{RunId, StepId};
pub use message::{
    AssistantBlock, CacheControl, Message, Source, SystemBlock, SystemPrompt, ToolResultBlock,
    ToolResultContent, UserBlock,
};
pub use middleware::{
    ChatMiddleware, ContinueDecision, HookAction, RunErrorInfo, RunFinishedInfo, RunStartInfo,
    StepInfo, ToolCallInfo, ToolDecision, TurnEndInfo,
};
pub use provider_error::{Capability, ProviderError, ProviderErrorKind};
pub use request::{ChatRequest, ReasoningEffort, ToolChoice, ToolDefinition, ToolTag};
pub use retry::{RetryClassification, RetryConfig, Retryable, RetryingModel};
pub use stream::{AbortReason, FinishReason, StreamChunk, ToolProgress, Usage};
pub use suspension::{CallState, Resume, ResumeDecision, ResumeError, SuspendedCall, Suspension};
pub use tokenizer::{CharTokenizer, Tokenizer};
pub use traits::{CompletionClient, CompletionModel};

pub use tokio_util::sync::CancellationToken;
