use crate::{ChatRequest, FinishReason, Message, RunId, StepId, StreamChunk, ToolResultContent, Usage};
use serde_json::Value;

use crate::RunConfig;

#[async_trait::async_trait]
#[allow(unused_variables)]
pub trait ChatMiddleware: Send + Sync {
    // chat
    async fn on_run_start(
        &self,
        run_id: &RunId,
        messages: &[Message],
        config: &RunConfig,
    ) -> HookAction {
        HookAction::Continue
    }
    async fn on_chat_request(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        req: &mut ChatRequest,
    ) {
    }
    async fn on_chunk(&self, chunk: &StreamChunk) {}
    async fn on_run_finished(
        &self,
        run_id: &RunId,
        reason: &FinishReason,
        usage: &Usage,
        new_messages: &[Message],
    ) {
    }
    async fn on_run_error(
        &self,
        run_id: &RunId,
        err: &(dyn std::error::Error + Send + Sync),
    ) {
    }

    // tools
    async fn on_before_tool_call(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &Value,
    ) -> ToolDecision {
        ToolDecision::Continue
    }
    async fn on_after_tool_call(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &Value,
        result: &ToolResultContent,
    ) {
    }
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
