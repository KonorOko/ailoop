//! [`ChatMiddleware`] extension point and its decision enums
//! ([`HookAction`], [`ToolDecision`]).

use crate::{
    ChatRequest, FinishReason, Message, RunId, StepId, StreamChunk, ToolResultContent, Usage,
};
use serde_json::Value;

use crate::RunConfig;

/// Extension point invoked by the engine at every lifecycle event of a
/// run.
///
/// Middlewares run in the registration order of
/// [`crate::RunConfig::middlewares`]. For every hook with a `_mut`
/// counterpart, the engine fires every middleware's mutating variant
/// first (in registration order), then every read-only variant — so
/// transformers always run as a phase ahead of observers, and every
/// observer sees the same fully-mutated input.
///
/// All hooks have default no-op implementations; only override the ones
/// you need. Implementors must be `Send + Sync` because the engine
/// holds them behind `Arc<dyn ChatMiddleware>`.
#[async_trait::async_trait]
#[allow(unused_variables)]
pub trait ChatMiddleware: Send + Sync {
    // chat
    /// Fired once per run before any provider call. Return
    /// [`HookAction::Terminate`] to abort early; the engine surfaces
    /// the reason as [`crate::FinishReason::Aborted`] and still fires
    /// [`Self::on_run_finished`] so observability is consistent.
    async fn on_run_started(
        &self,
        run_id: &RunId,
        messages: &[Message],
        config: &RunConfig,
    ) -> HookAction {
        HookAction::Continue
    }
    /// Fired once per step, after the engine has assembled the
    /// per-turn [`ChatRequest`] but before sending it. Mutate `req` to
    /// inject defaults, switch model parameters per-turn, or strip
    /// fields. The façade's per-builder defaults are wired through an
    /// internal middleware that runs ahead of any user-supplied one.
    async fn on_chat_request(&self, run_id: &RunId, step_id: &StepId, req: &mut ChatRequest) {}
    /// Fired for every [`StreamChunk`] the engine emits, including
    /// chunks the engine itself synthesizes
    /// (`RunStarted`/`StepStarted`/`StepFinished`/`ToolResult`/
    /// `RunFinished`/`HistoryCompacted`). For mutation, override
    /// [`Self::on_chunk_mut`] instead.
    async fn on_chunk(&self, chunk: &StreamChunk) {}
    /// Mutating counterpart to [`Self::on_chunk`]. Engines invoke every
    /// middleware's `on_chunk_mut` (in registration order) **before** any
    /// `on_chunk`, so transformers run as a phase ahead of observers and
    /// every observer sees the same fully-mutated chunk. The mutated
    /// chunk is also what the engine itself uses to build assistant
    /// history and what the stream consumer ultimately receives.
    async fn on_chunk_mut(&self, chunk: &mut StreamChunk) {}
    /// Fired once per run after the engine emits its
    /// [`StreamChunk::RunFinished`]. Always fires — including aborted
    /// runs and runs terminated by middleware — so observers see a
    /// consistent close. `new_messages` covers everything the engine
    /// added to history this run; partial tool results are preserved
    /// when the run was aborted mid-step.
    ///
    /// [`StreamChunk::RunFinished`]: crate::StreamChunk::RunFinished
    async fn on_run_finished(
        &self,
        run_id: &RunId,
        reason: &FinishReason,
        usage: &Usage,
        new_messages: &[Message],
    ) {
    }
    /// Fired when a run terminates with a transport / setup-time
    /// error from the provider (i.e. an `Err` returned to the caller).
    /// Aborts via [`HookAction::Terminate`] /
    /// [`ToolDecision::Terminate`] / `RunConfig.cancellation` /
    /// `RunConfig.timeout` go through [`Self::on_run_finished`]
    /// instead — they are not errors.
    async fn on_run_error(&self, run_id: &RunId, err: &(dyn std::error::Error + Send + Sync)) {}

    // tools
    /// Fired before the engine invokes a tool. Return
    /// [`ToolDecision::Skip`] to feed a synthesized error result back
    /// to the model without running the tool, or
    /// [`ToolDecision::Terminate`] to abort the run. This is the
    /// gating hook; for input rewriting, use
    /// [`Self::on_before_tool_call_mut`].
    async fn on_before_tool_call(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &Value,
    ) -> ToolDecision {
        ToolDecision::Continue
    }
    /// Mutating counterpart to [`Self::on_before_tool_call`]. Engines invoke
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
    /// Fired after the engine has executed a tool but before the
    /// result is yielded to the stream consumer or recorded in
    /// history. Read-only; for output rewriting use
    /// [`Self::on_after_tool_call_mut`].
    async fn on_after_tool_call(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &Value,
        result: &ToolResultContent,
    ) {
    }
    /// Mutating counterpart to [`Self::on_after_tool_call`]. Engines invoke
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

/// Decision returned from
/// [`ChatMiddleware::on_run_started`] to optionally short-circuit a
/// run before any provider call.
#[non_exhaustive]
pub enum HookAction {
    /// Default: let the run proceed.
    Continue,
    /// Abort the run before the first provider call. The engine
    /// surfaces the reason as
    /// [`crate::FinishReason::Aborted`]`(reason)` and still fires
    /// [`ChatMiddleware::on_run_finished`].
    Terminate {
        /// Human-readable reason; threaded through `FinishReason::Aborted`.
        reason: String,
    },
}

/// Decision returned from
/// [`ChatMiddleware::on_before_tool_call`] to optionally bypass or
/// abort a tool invocation.
#[non_exhaustive]
pub enum ToolDecision {
    /// Default: execute the tool.
    Continue,
    /// Skip execution. The engine synthesizes an `is_error: true`
    /// [`crate::ToolResultContent`] carrying `reason` and feeds it
    /// back to the model so the loop can continue. Use when a single
    /// tool call should be denied (rate limit, policy violation) but
    /// the run as a whole should keep going.
    Skip {
        /// Human-readable reason; surfaces in the synthesized error
        /// `tool_result` the model receives.
        reason: String,
    },
    /// Abort the run before executing this tool. The engine surfaces
    /// the reason as [`crate::FinishReason::Aborted`]`(reason)` and
    /// fires [`ChatMiddleware::on_run_finished`]; partial tool
    /// results from earlier tool calls in the same step are preserved
    /// in `new_messages`.
    Terminate {
        /// Human-readable reason; threaded through `FinishReason::Aborted`.
        reason: String,
    },
}
