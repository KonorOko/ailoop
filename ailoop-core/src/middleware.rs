use crate::{
    ChatRequest, FinishReason, Message, RunId, StepId, StreamChunk, ToolResultContent, Usage,
};
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
    async fn on_chat_request(&self, run_id: &RunId, step_id: &StepId, req: &mut ChatRequest) {}
    async fn on_chunk(&self, chunk: &StreamChunk) {}
    /// Mutating counterpart to [`on_chunk`]. Engines invoke every
    /// middleware's `on_chunk_mut` (in registration order) **before** any
    /// `on_chunk`, so transformers run as a phase ahead of observers and
    /// every observer sees the same fully-mutated chunk. The mutated
    /// chunk is also what the engine itself uses to build assistant
    /// history and what the stream consumer ultimately receives.
    async fn on_chunk_mut(&self, chunk: &mut StreamChunk) {}
    async fn on_run_finished(
        &self,
        run_id: &RunId,
        reason: &FinishReason,
        usage: &Usage,
        new_messages: &[Message],
    ) {
    }
    async fn on_run_error(&self, run_id: &RunId, err: &(dyn std::error::Error + Send + Sync)) {}

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
    /// Mutating counterpart to [`on_before_tool_call`]. Engines invoke
    /// every middleware's `on_before_tool_call_mut` (in registration
    /// order) **before** any `on_before_tool_call`, so input transforms
    /// (sanitization, redaction, defaulting) run as a phase ahead of
    /// gating decisions. Gating still belongs in `on_before_tool_call`;
    /// this hook only rewrites `args`.
    async fn on_before_tool_call_mut(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &mut Value,
    ) {
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
    /// Mutating counterpart to [`on_after_tool_call`]. Engines invoke
    /// every middleware's `on_after_tool_call_mut` (in registration
    /// order) **before** any `on_after_tool_call`, so output transforms
    /// (PII scrubbing, truncation-with-marker) run as a phase ahead of
    /// observers. The mutated `result` is what the model sees on the
    /// next turn and what the engine emits in `StreamChunk::ToolResult`.
    async fn on_after_tool_call_mut(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &Value,
        result: &mut ToolResultContent,
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
