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

#![deny(missing_docs)]

pub mod config;
pub mod ids;
pub mod message;
pub mod middleware;
pub mod request;
pub mod retry;
pub mod stream;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod tokenizer;
mod traits;

pub use config::RunConfig;
pub use ids::{RunId, StepId};
pub use message::{
    AssistantBlock, CacheControl, Message, Source, SystemBlock, SystemPrompt, ToolResultBlock,
    ToolResultContent, UserBlock,
};
pub use middleware::{ChatMiddleware, HookAction, ToolDecision};
pub use request::{ChatRequest, ToolChoice, ToolDefinition, ToolTag};
pub use retry::{RetryClassification, RetryConfig, Retryable, RetryingModel};
pub use stream::{FinishReason, StreamChunk, Usage};
pub use tokenizer::{CharTokenizer, Tokenizer};
pub use traits::{CompletionClient, CompletionModel};

pub use tokio_util::sync::CancellationToken;
