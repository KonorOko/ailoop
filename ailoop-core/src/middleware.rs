//! [`ChatMiddleware`] extension point and its decision enums
//! ([`HookAction`], [`ToolDecision`], [`ContinueDecision`]).

use crate::{
    ChatRequest, FinishReason, Message, RunId, StepId, StreamChunk, ToolResultContent, Usage,
    UserBlock,
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
///
/// # Tool calls within a step
///
/// When the model requests several tools in one turn, the tool hooks
/// ([`Self::on_before_tool_call_mut`], [`Self::on_before_tool_call`],
/// [`Self::on_after_tool_call_mut`], [`Self::on_after_tool_call`]) and
/// the [`StreamChunk::ToolResult`] chunk fire once per call. The
/// engine guarantees:
///
/// - **Order within one call.** Every middleware's
///   `on_before_tool_call_mut`, then the `on_before_tool_call` gates,
///   then the tool, then every `on_after_tool_call_mut`, then every
///   `on_after_tool_call`, then the `ToolResult` chunk.
/// - **Order in history.** The tool results recorded in history (and in
///   `new_messages`) follow the order of the model's tool calls, no
///   matter in which order the calls ran.
/// - **Step boundary.** Every call of a step has finished before
///   [`Self::on_turn_end`] or the next step's [`Self::on_chat_request`]
///   fires.
/// - **Rejected calls.** A call the engine refuses without running it
///   (arguments that are not a JSON object, or a tool that is not in
///   the run's active set) fires none of the tool hooks, only its
///   `ToolResult` chunk carrying the error sent back to the model.
///
/// The order **between** calls of the same step is not guaranteed.
/// Today the engine runs them one at a time, in the model's order, but
/// a later release may run them concurrently, so the hooks of different
/// calls can interleave or overlap. A middleware that must keep working
/// then:
///
/// - does not assume an `on_after_tool_call` belongs to the most recent
///   `on_before_tool_call`, or that the calls before it in the model's
///   order have already run;
/// - keys per-call state by `(run_id, call_id)` from the
///   [`ToolCallInfo`] every tool hook receives, instead of keeping a
///   single "current call" slot. Two calls to the same tool with the
///   same arguments in one step are told apart only by their
///   `call_id`, which also matches the `id` / `call_id` on
///   [`StreamChunk::ToolCallFinished`] and [`StreamChunk::ToolResult`];
/// - keeps per-run state keyed by [`RunId`] (and per-step state by
///   [`StepId`]), guards all of it with a lock, since hooks take
///   `&self`, and drops a run's entries in [`Self::on_run_finished`] /
///   [`Self::on_run_error`].
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
    ///
    /// The façade's system-prompt assembly runs *after* user
    /// middlewares, so `req.system_prompt` is `None` when a user
    /// middleware sees it inside a `Conversation`. Whatever a user
    /// middleware writes there is not replaced: it is appended after
    /// the builder's system prompt, and per-block `cache_control` on a
    /// [`crate::SystemPrompt::Blocks`] value is preserved.
    ///
    /// The current iteration is not passed here, but it is available:
    /// the engine emits [`StreamChunk::StepStarted`] (carrying
    /// `iteration`) through [`Self::on_chunk`] right before calling
    /// `on_chat_request` for the same step, and
    /// [`RunConfig::max_iterations`] arrives in
    /// [`Self::on_run_started`]. A middleware that needs "how many
    /// iterations are left" records both keyed by `run_id` (e.g. in a
    /// `Mutex<HashMap<RunId, _>>`, removed in [`Self::on_run_finished`])
    /// and reads them here. Retries by [`crate::RetryingModel`] happen
    /// inside `chat_stream`, so this hook runs once per iteration; the
    /// only exception is the context-overflow recovery of a
    /// `Conversation`, which rebuilds the request and calls it again
    /// with the same `step_id`.
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
    /// Fired when a turn ends with nothing left for the engine to do —
    /// the model stopped for any reason other than
    /// [`FinishReason::ToolUse`] — right before the engine would finish
    /// the run. Return [`ContinueDecision::Continue`] to append a user
    /// message and keep the run going: the "completion gate" pattern,
    /// where a middleware verifies the result and sends the model back
    /// to work when the check fails.
    ///
    /// `reason` is the turn's finish reason ([`FinishReason::EndTurn`],
    /// [`FinishReason::MaxTokens`], [`FinishReason::StopSequence`] or
    /// [`FinishReason::Other`]); a gate usually acts only on `EndTurn`.
    /// Never fired for [`FinishReason::Aborted`]. `new_messages` is
    /// everything the run has added so far, including this turn's
    /// assistant message.
    ///
    /// The continuation stays inside the same run: the injected message
    /// lands in the history and in `new_messages`, the next step emits
    /// the usual `StepStarted` with the next `iteration`, usage keeps
    /// accumulating, and a single `RunFinished` closes the run. Every
    /// continuation counts against [`RunConfig::max_iterations`]; a
    /// gate that never passes ends the run with
    /// [`crate::AbortReason::MaxIterations`].
    ///
    /// If the step also completed tool calls, the injected blocks join
    /// the tool results in one user message. If the turn produced no
    /// assistant content at all, nothing separates the previous user
    /// message from the injected one; providers merge consecutive user
    /// turns.
    ///
    /// A gate on the child `Conversation` of a `SubAgentTool` with a
    /// wrap-up configured can defeat it: continuing after the wrap-up
    /// turn spends the last iteration (or runs into the hard timeout),
    /// and the parent gets an abort instead of the partial summary.
    /// Such a gate should budget for it, e.g. by tracking
    /// `StepStarted { iteration }` against `max_iterations`.
    ///
    /// Middlewares are asked in registration order and the first
    /// `Continue` wins; the rest are not asked for that turn. A
    /// `Continue` with no blocks counts as [`ContinueDecision::Stop`]:
    /// continuing without a new user message would send a request
    /// that ends on the assistant's own turn.
    async fn on_turn_end(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        reason: &FinishReason,
        new_messages: &[Message],
    ) -> ContinueDecision {
        ContinueDecision::Stop
    }
    /// Fired when a run terminates with a transport / setup-time
    /// error from the provider (i.e. an `Err` returned to the caller).
    /// Aborts via [`HookAction::Terminate`] /
    /// [`ToolDecision::Terminate`] / `RunConfig.cancellation` /
    /// `RunConfig.timeout` go through [`Self::on_run_finished`]
    /// instead — they are not errors.
    ///
    /// `usage` is what the run spent before failing, the same value the
    /// caller gets from `RunError::usage`: every provider turn that
    /// finished (each one a [`StreamChunk::TurnFinished`]) plus the
    /// usage tools reported through `ToolContext::report_usage`,
    /// sub-agents included. It follows the rules of
    /// [`StreamChunk::RunFinished::usage`] with one gap: a turn that
    /// fails mid-stream never reports its usage, so its tokens are not
    /// counted even though the provider may bill them. Summing
    /// `TurnFinished` in [`Self::on_chunk`] only recovers the run's own
    /// turns; this value also has the delegated spend.
    ///
    /// `partial_messages` holds the messages of the steps the run
    /// completed before failing, the same list the caller receives in
    /// the returned error. Every `tool_use` in it has its `tool_result`,
    /// and the step that failed is left out. It is empty when the first
    /// step failed. The conversation history is still rolled back; use
    /// this to record the work that already happened, for example tools
    /// with side effects. `usage` can include the failed step (its turn
    /// finished before, say, a tool registry error) while
    /// `partial_messages` does not: one is what was spent, the other
    /// what is safe to keep.
    ///
    /// Only fired for runs that started. An error raised before the
    /// run starts, such as `Conversation` compacting the history before
    /// the first step, reaches the caller without any hook.
    async fn on_run_error(
        &self,
        run_id: &RunId,
        err: &(dyn std::error::Error + Send + Sync),
        usage: &Usage,
        partial_messages: &[Message],
    ) {
    }

    // tools
    /// Fired before the engine invokes a tool. Return
    /// [`ToolDecision::Skip`] to feed a synthesized error result back
    /// to the model without running the tool, or
    /// [`ToolDecision::Terminate`] to abort the run. This is the
    /// gating hook; for input rewriting, use
    /// [`Self::on_before_tool_call_mut`].
    ///
    /// Calls of the same step may reach this hook in any order, or
    /// concurrently; see [Tool calls within a step](Self#tool-calls-within-a-step).
    async fn on_before_tool_call(&self, call: &ToolCallInfo, args: &Value) -> ToolDecision {
        ToolDecision::Continue
    }
    /// Mutating counterpart to [`Self::on_before_tool_call`]. Engines invoke
    /// every middleware's `on_before_tool_call_mut` (in registration
    /// order) **before** any `on_before_tool_call`, so input transforms
    /// (sanitization, redaction, defaulting) run as a phase ahead of
    /// gating decisions. Gating still belongs in `on_before_tool_call`;
    /// this hook only rewrites `args`.
    async fn on_before_tool_call_mut(&self, call: &ToolCallInfo, args: &mut Value) {}
    /// Fired after the engine has executed a tool but before the
    /// result is yielded to the stream consumer or recorded in
    /// history. Read-only; for output rewriting use
    /// [`Self::on_after_tool_call_mut`].
    async fn on_after_tool_call(
        &self,
        call: &ToolCallInfo,
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
        call: &ToolCallInfo,
        args: &Value,
        result: &mut ToolResultContent,
    ) {
    }
}

/// Identity of one tool call, passed to every tool hook of
/// [`ChatMiddleware`].
///
/// `call_id` is the provider-assigned id of the call: the same value
/// as `id` on [`StreamChunk::ToolCallFinished`] and `call_id` on
/// [`StreamChunk::ToolResult`], so a middleware can pair the hooks of
/// one call with each other and with the stream. Providers give every
/// call of a step its own id; key per-call state by `(run_id, call_id)`.
///
/// The arguments and result travel as separate hook parameters, since
/// the `_mut` hooks borrow them mutably.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ToolCallInfo {
    /// Run the call belongs to.
    pub run_id: RunId,
    /// Step (model turn) that produced the call.
    pub step_id: StepId,
    /// Provider-assigned tool call id.
    pub call_id: String,
    /// Tool name as the model called it.
    pub name: String,
}

impl ToolCallInfo {
    /// Build the identity of a tool call. The engine does this for
    /// every call it dispatches; use it directly to unit-test a
    /// middleware's tool hooks.
    pub fn new(
        run_id: RunId,
        step_id: StepId,
        call_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            run_id,
            step_id,
            call_id: call_id.into(),
            name: name.into(),
        }
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
    /// surfaces it as [`crate::FinishReason::Aborted`] carrying
    /// [`crate::AbortReason::Terminated`] and still fires
    /// [`ChatMiddleware::on_run_finished`].
    Terminate {
        /// Human-readable reason; threaded through
        /// [`crate::AbortReason::Terminated`].
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
    /// it as [`crate::FinishReason::Aborted`] carrying
    /// [`crate::AbortReason::ToolTerminated`] (with the tool's name)
    /// and fires [`ChatMiddleware::on_run_finished`]; results of tool
    /// calls in the same step that already completed are preserved in
    /// `new_messages`. Which calls those are follows the execution
    /// order, which is not guaranteed between calls of a step; see
    /// [Tool calls within a step](ChatMiddleware#tool-calls-within-a-step).
    Terminate {
        /// Human-readable reason; threaded through
        /// [`crate::AbortReason::ToolTerminated`].
        reason: String,
    },
}

/// Decision returned from [`ChatMiddleware::on_turn_end`] to either
/// let the run finish or keep it going with an injected user message.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub enum ContinueDecision {
    /// Default: finish the run with the turn's finish reason.
    #[default]
    Stop,
    /// Append a user message made of `blocks` to the history and run
    /// another iteration. The message is recorded in the run's
    /// `new_messages` like any other message the engine adds. Empty
    /// `blocks` count as [`ContinueDecision::Stop`].
    Continue {
        /// Content of the injected user message.
        blocks: Vec<UserBlock>,
    },
}

impl ContinueDecision {
    /// [`ContinueDecision::Continue`] with a single text block.
    pub fn continue_with(text: impl Into<String>) -> Self {
        ContinueDecision::Continue {
            blocks: vec![UserBlock::text(text)],
        }
    }
}
