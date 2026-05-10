mod anti_loop;
mod conversation;
mod engine;
mod errors;
mod json_tracer;
mod middleware;
mod sanitize;
mod sub_agent;
#[cfg(feature = "tracing")]
mod tracing_middleware;

pub use ailoop_context::{
    CompactionError, CompactionStrategy, ContextManager, ContextManagerBuilder,
    ConversationSnapshot, FromMessagesError, HistoryStore, InMemoryHistoryStore,
    JsonFileHistoryStore, JsonFileHistoryStoreError, SummarizeStrategy, TruncateStrategy,
};
pub use ailoop_core::{
    AssistantBlock, CancellationToken, CharTokenizer, ChatMiddleware, ChatRequest,
    CompletionClient, CompletionModel, FinishReason, HookAction, Message, RetryClassification,
    RetryConfig, Retryable, RetryingModel, RunConfig, RunId, StepId, StreamChunk, Tokenizer,
    ToolChoice, ToolDecision, ToolDefinition, ToolResultContent, ToolTag, Usage, UserBlock,
};
pub use ailoop_derive::{ToolJsonType, ailoop_tool};
pub use ailoop_prompts::{Prompt, PromptBuilder, PromptSection};
pub use ailoop_tools::{Tool, ToolDyn, ToolJsonType, ToolRegistry, errors::ToolRegistryError};
pub use anti_loop::{AntiLoop, TextPredicate};
pub use conversation::{Conversation, ConversationBuilder, RunOutcome, RunStream};
pub use engine::run_chat;
pub use errors::{BuildError, EngineError};
pub use json_tracer::JsonTracer;
pub use middleware::ApprovalMiddleware;
pub use sanitize::{Sanitize, TextRewriter, ToolArgsRewriter, ToolResultRewriter};
pub use sub_agent::SubAgentTool;
#[cfg(feature = "tracing")]
pub use tracing_middleware::TracingMiddleware;
