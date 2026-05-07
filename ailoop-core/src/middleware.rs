use crate::{ChatRequest, FinishReason, Message, StreamChunk, ToolResultContent, Usage};
use serde_json::Value;

use crate::RunConfig;

#[async_trait::async_trait]
pub trait ChatMiddleware: Send + Sync {
    // chat
    async fn on_run_start(&self, messages: &[Message], config: &RunConfig) -> HookAction {
        HookAction::Continue
    }
    async fn on_chat_request(&self, req: &mut ChatRequest) {}
    async fn on_chunk(&self, chunk: &StreamChunk) {}
    async fn on_run_finished(
        &self,
        reason: &FinishReason,
        usage: &Usage,
        new_messages: &[Message],
    ) {
    }
    async fn on_run_error(&self, err: &dyn std::error::Error) {}

    // tools
    async fn on_before_tool_call(&self, name: &str, args: &Value) -> ToolDecision {
        ToolDecision::Continue
    }
    async fn on_after_tool_call(&self, name: &str, args: &Value, result: &ToolResultContent) {}
}

pub enum HookAction {
    Continue,
    Terminate { reason: String },
}

pub enum ToolDecision {
    Continue,
    Skip { reason: String },
    Terminate { reason: String },
}
