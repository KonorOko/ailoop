mod conversation;
mod engine;
mod errors;
mod middleware;
#[cfg(feature = "tracing")]
mod tracing_middleware;

pub use ailoop_context::ContextManager;
pub use ailoop_core::{
    AssistantBlock, ChatMiddleware, Message, RunConfig, StreamChunk, ToolDecision, ToolDefinition,
    ToolResultContent, ToolTag, Usage,
};
pub use ailoop_derive::ailoop_tool;
pub use ailoop_prompts::{Prompt, PromptSection};
pub use ailoop_tools::{Tool, ToolRegistry};
pub use conversation::{Conversation, RunOutcome};
pub use engine::run_chat;
pub use middleware::ApprovalMiddleware;
#[cfg(feature = "tracing")]
pub use tracing_middleware::TracingMiddleware;
