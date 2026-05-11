//! High-level façade for building an LLM agent loop. Most application
//! code only depends on this crate — it re-exports the vocabulary from
//! [`ailoop_core`] (messages, stream chunks, hooks) and from the side
//! crates (`ailoop-history`, `ailoop-tools`, `ailoop-prompts`) you need
//! to wire a [`Conversation`] together.
//!
//! ## Happy path
//!
//! ```no_run
//! # async fn run<M: ailoop::CompletionModel + Send + Sync>(model: M)
//! # -> Result<(), Box<dyn std::error::Error>> {
//! let mut chat = ailoop::Conversation::builder(model)
//!     .system_prompt("You are a helpful assistant.")
//!     .build()?;
//!
//! let outcome = chat.run("What is the speed of light?").await?;
//! println!("{}", outcome.final_text.unwrap_or_default());
//! # Ok(()) }
//! ```
//!
//! [`Conversation::run`] is the one-shot helper for CLI flows and
//! notebooks; [`Conversation::stream`] yields one [`StreamChunk`] at a
//! time when you want to render tokens, observe tool calls, or thread
//! events through middleware as they happen.
//!
//! ## Mini-index
//!
//! - [`Conversation`] — the agent loop. Construct via
//!   [`Conversation::builder`].
//! - [`ConversationBuilder`] — builder pattern. Register tools with
//!   [`tool`](ConversationBuilder::tool) /
//!   [`tool_dyn`](ConversationBuilder::tool_dyn), gate them with
//!   [`with_capabilities`](ConversationBuilder::with_capabilities) /
//!   [`with_approval`](ConversationBuilder::with_approval), and layer
//!   per-request defaults with [`temperature`](ConversationBuilder::temperature),
//!   [`max_tokens`](ConversationBuilder::max_tokens), and friends.
//! - [`RunOutcome`] — what [`Conversation::run`] returns. Aborts surface
//!   here as [`FinishReason::Aborted`], not as `Err`.
//! - Built-in middlewares: [`AntiLoop`] (loop-detection abort),
//!   [`MaxToolCalls`] (flat cap on total tool invocations per run),
//!   [`Sanitize`] (caller-supplied text rewrites at the model
//!   boundary), [`ApprovalMiddleware`] (human-in-the-loop gating),
//!   [`JsonTracer`] (NDJSON event sink). With the `tracing` feature,
//!   `TracingMiddleware` routes the same events through the
//!   `tracing` crate.
//! - [`SubAgentTool`] — wrap a [`Conversation`] as a [`ToolDyn`] so a
//!   parent agent can delegate to it.
//! - [`advanced::run_chat`] — escape hatch for engine-level access
//!   without a [`History`] in the loop.

#![deny(missing_docs)]

mod anti_loop;
mod conversation;
mod engine;
mod errors;
mod json_tracer;
mod max_tool_calls;
mod middleware;
mod sanitize;
mod sub_agent;
#[cfg(feature = "tracing")]
mod tracing_middleware;

pub use ailoop_core::{
    AssistantBlock, CancellationToken, CharTokenizer, ChatMiddleware, ChatRequest,
    CompletionClient, CompletionModel, FinishReason, HookAction, Message, ReasoningEffort,
    RetryClassification, RetryConfig, Retryable, RetryingModel, RunConfig, RunId, StepId,
    StreamChunk, Tokenizer, ToolChoice, ToolDecision, ToolDefinition, ToolResultContent, ToolTag,
    Usage, UserBlock,
};
pub use ailoop_derive::{ToolJsonType, ailoop_tool};
pub use ailoop_history::{
    CompactionError, CompactionStrategy, ConversationSnapshot, FromMessagesError, History,
    HistoryBuilder, HistoryStore, InMemoryHistoryStore, JsonFileHistoryStore,
    JsonFileHistoryStoreError, SummarizeStrategy, TruncateStrategy,
};
pub use ailoop_prompts::{Prompt, PromptBuilder, PromptSection};
// Note: `ToolJsonType` is also re-exported above from `ailoop_derive` as
// the derive macro of the same name. The two live in different
// namespaces (one is a trait, one is a macro), so both can be brought
// into scope by `use ailoop::*;` without conflict.
pub use ailoop_tools::{
    TimeoutTool, Tool, ToolActivation, ToolActivationError, ToolContext, ToolDyn, ToolJsonType,
    ToolRegistry, errors::ToolRegistryError,
};
pub use anti_loop::{AntiLoop, TextPredicate};
pub use conversation::{
    Conversation, ConversationBuilder, DEFAULT_HISTORY_MAX_TOKENS, RunOutcome, RunStream,
};
pub use errors::{BuildError, EngineError};
pub use max_tool_calls::MaxToolCalls;

/// Lower-level entry points outside the [`Conversation`] happy path.
///
/// Most callers should use [`Conversation::builder`] — it wires history
/// management, system-prompt assembly, and per-request defaults. Reach
/// into this module only when you need to drive the engine without a
/// [`History`] in the loop (e.g. one-shot calls with a fixed
/// message slice and a pre-built [`ToolRegistry`]).
pub mod advanced {
    pub use crate::engine::run_chat;
}
pub use json_tracer::JsonTracer;
pub use middleware::ApprovalMiddleware;
pub use sanitize::{Sanitize, TextRewriter, ToolArgsRewriter, ToolResultRewriter};
pub use sub_agent::SubAgentTool;
#[cfg(feature = "tracing")]
pub use tracing_middleware::TracingMiddleware;
