//! High-level façade for building an LLM agent loop. Most application
//! code only depends on this crate — it re-exports the vocabulary from
//! [`ailoop_core`] (messages, stream chunks, hooks) and from the side
//! crates (`ailoop-history`, `ailoop-tools`, `ailoop-prompts`) you need
//! to wire a [`Conversation`] together.
//!
//! ## Happy path
//!
//! ```no_run
//! # async fn run<M>(model: M) -> Result<(), Box<dyn std::error::Error>>
//! # where M: ailoop::CompletionModel, M::Error: ailoop::ProviderError {
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
//!   [`capabilities`](ConversationBuilder::capabilities) /
//!   [`approval`](ConversationBuilder::approval), and layer
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
//!
//! ## Writing a middleware
//!
//! [`ChatMiddleware`] (like [`CompactionStrategy`] and
//! [`CompletionModel`]) is an `async_trait` trait. The macro is
//! re-exported as [`macro@async_trait`], so an implementation needs no
//! `async-trait` dependency of its own:
//!
//! ```
//! use ailoop::{ChatMiddleware, ContinueDecision, TurnEndInfo};
//!
//! struct LogTurns;
//!
//! #[ailoop::async_trait]
//! impl ChatMiddleware for LogTurns {
//!     async fn on_turn_end(&self, turn: &TurnEndInfo<'_>) -> ContinueDecision {
//!         println!("turn ended: {:?}", turn.reason);
//!         ContinueDecision::Stop
//!     }
//! }
//! ```

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
    AbortReason, AssistantBlock, CacheControl, CancellationToken, CharTokenizer, ChatMiddleware,
    ChatRequest, CompletionClient, CompletionModel, ContinueDecision, DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS, FinishReason, HookAction, Message, ProviderError, ReasoningEffort,
    RetryClassification, RetryConfig, Retryable, RetryingModel, RunConfig, RunErrorInfo,
    RunFinishedInfo, RunId, RunStartInfo, Source, StepId, StepInfo, StreamChunk, SystemBlock,
    SystemPrompt, Tokenizer, ToolCallInfo, ToolChoice, ToolDecision, ToolDefinition,
    ToolResultBlock, ToolResultContent, ToolTag, TurnEndInfo, Usage, UserBlock,
};
pub use ailoop_derive::{ToolJsonType, ailoop_tool};
pub use ailoop_history::{
    CompactionError, CompactionOutput, CompactionStats, CompactionStrategy, ConversationSnapshot,
    DEFAULT_SUMMARIZER_PROMPT, FromMessagesError, History, HistoryBuilder, HistoryStore,
    InMemoryHistoryStore, JsonFileHistoryStore, JsonFileHistoryStoreError, SummarizeStrategy,
    TruncateStrategy,
};
pub use ailoop_prompts::{Prompt, PromptBuilder, PromptSection};
/// Attribute macro for implementing the crate's async traits
/// ([`ChatMiddleware`], [`CompactionStrategy`], [`CompletionModel`], …).
///
/// Re-exported from the `async-trait` crate so implementations can write
/// `#[ailoop::async_trait]` without adding (and version-matching) that
/// dependency themselves.
pub use async_trait::async_trait;
// Note: `ToolJsonType` is also re-exported above from `ailoop_derive` as
// the derive macro of the same name. The two live in different
// namespaces (one is a trait, one is a macro), so both can be brought
// into scope by `use ailoop::*;` without conflict.
pub use ailoop_tools::{
    TimeoutTool, Tool, ToolActivation, ToolActivationError, ToolContext, ToolDyn, ToolJsonType,
    ToolRegistry, ToolRegistryError, UsageSink,
};
pub use anti_loop::{AntiLoop, TextPredicate};
pub use conversation::{
    Conversation, ConversationBuilder, DEFAULT_HISTORY_MAX_TOKENS, RunOptions, RunOutcome,
    RunStream,
};
pub use errors::{BuildError, EngineError, RunError, RunErrorParts};
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

/// Test doubles for exercising your own middlewares, tools and
/// conversation code without a real provider.
///
/// [`ScriptedModel`](testing::ScriptedModel) replays a queue of scripted
/// turns (chunks, setup errors, mid-stream errors) as a
/// [`CompletionModel`]. Enabled by the `testing` feature; add it under
/// `[dev-dependencies]` so it stays out of release builds:
///
/// ```toml
/// [dev-dependencies]
/// ailoop = { version = "1", features = ["testing"] }
/// ```
#[cfg(feature = "testing")]
pub mod testing {
    pub use ailoop_core::testing::*;
}
pub use json_tracer::JsonTracer;
pub use middleware::{ApprovalMiddleware, ApprovalRequest};
pub use sanitize::{Sanitize, TextRewriter, ToolArgsRewriter, ToolResultRewriter};
pub use sub_agent::{DEFAULT_WRAP_UP_INSTRUCTION, SubAgentConfig, SubAgentTool, WrapUp};
#[cfg(feature = "tracing")]
pub use tracing_middleware::TracingMiddleware;
