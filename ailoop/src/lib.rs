mod conversation;
mod engine;
mod errors;
mod middleware;
mod sub_agent;
#[cfg(feature = "tracing")]
mod tracing_middleware;

pub use ailoop_context::{
    ContextManager, ConversationSnapshot, HistoryStore, InMemoryHistoryStore, JsonFileHistoryStore,
    JsonFileHistoryStoreError,
};
pub use ailoop_core::{
    AssistantBlock, ChatMiddleware, Message, RetryClassification, RetryConfig, Retryable,
    RetryingModel, RunConfig, StreamChunk, ToolChoice, ToolDecision, ToolDefinition,
    ToolResultContent, ToolTag, Usage,
};
pub use ailoop_derive::{ToolJsonType, ailoop_tool};
pub use ailoop_prompts::{Prompt, PromptSection};
pub use ailoop_tools::{Tool, ToolDyn, ToolJsonType, ToolRegistry};
pub use conversation::{Conversation, ConversationBuilder, RunOutcome};
pub use engine::run_chat;
pub use middleware::ApprovalMiddleware;
pub use sub_agent::SubAgentTool;
#[cfg(feature = "tracing")]
pub use tracing_middleware::TracingMiddleware;
