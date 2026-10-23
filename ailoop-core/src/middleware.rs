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
/// The trait is declared with `#[async_trait]`, so an implementation
/// needs the same attribute on its `impl` block. The `ailoop` facade
/// re-exports the macro: write `#[ailoop::async_trait]` and skip adding
/// `async-trait` to your own `Cargo.toml`.
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
/// By default the engine runs the calls of a step one at a time, in the
/// model's order. With [`crate::RunConfig::tool_concurrency`] above 1,
/// consecutive calls to tools that report `concurrency_safe` form a
/// batch, and for each batch:
///
/// - the `on_before_tool_call_mut` / `on_before_tool_call` hooks of every
///   call run first, one call at a time in the model's order, so gates
///   such as approvals still see one call at a time and in sequence;
/// - the calls that passed then run at the same time, up to the limit;
/// - the `on_after_tool_call_mut` / `on_after_tool_call` hooks and the
///   `ToolResult` chunks follow, one call at a time in the model's
///   order, once every call of the batch has finished.
///
/// A call to any other tool is a batch of its own. So the hooks of
/// different calls can interleave (a batch's `before` hooks all run
/// before its first `after` hook), and tools of one batch overlap.
/// Beyond that, the order between calls of a step is not part of the
/// contract and may change. A middleware that must keep working then:
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
///   `&self`, and drops a run's entries in [`Self::on_run_finished`],
///   [`Self::on_run_error`] and [`Self::on_run_dropped`].
#[async_trait::async_trait]
#[allow(unused_variables)]
pub trait ChatMiddleware: Send + Sync {
    // chat
    /// Fired once per run before any provider call. Return
    /// [`HookAction::Terminate`] to abort early; the engine surfaces
    /// the reason as [`crate::FinishReason::Aborted`] and still fires
    /// [`Self::on_run_finished`] so observability is consistent.
    async fn on_run_started(&self, run: &RunStartInfo<'_>) -> HookAction {
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
    /// `step` carries the iteration and the run's cap: iterations left,
    /// counting this one, are `step.max_iterations - step.iteration`.
    /// Count with those rather than with calls to this hook. Retries by
    /// [`crate::RetryingModel`] happen inside `chat_stream` and do not
    /// call it again, but the context-overflow recovery of a
    /// `Conversation` rebuilds the request and calls it a second time
    /// for the same step, with the same `step_id` and `iteration`.
    async fn on_chat_request(&self, step: &StepInfo, req: &mut ChatRequest) {}
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
    ///
    /// The fields of a [`StreamChunk::RunFinished`] may be rewritten,
    /// but not its variant: if a middleware replaces it with another
    /// chunk, the engine restores the original, so every run's stream
    /// still ends in `RunFinished`.
    ///
    /// [`StreamChunk::RunFinished`]: crate::StreamChunk::RunFinished
    async fn on_chunk_mut(&self, chunk: &mut StreamChunk) {}
    /// Fired once per run after the engine emits its
    /// [`StreamChunk::RunFinished`]. Always fires — including aborted
    /// runs and runs terminated by middleware — so observers see a
    /// consistent close, unless the run fails ([`Self::on_run_error`])
    /// or its stream is dropped first ([`Self::on_run_dropped`]).
    /// `run.new_messages` covers everything the engine added to history
    /// this run. When the run was aborted mid-step,
    /// every `tool_use` in it still has its `tool_result`: completed
    /// calls keep theirs, the rest get a synthesized error.
    ///
    /// [`StreamChunk::RunFinished`]: crate::StreamChunk::RunFinished
    async fn on_run_finished(&self, run: &RunFinishedInfo<'_>) {}
    /// Fired when a turn ends with nothing left for the engine to do —
    /// the model stopped for any reason other than
    /// [`FinishReason::ToolUse`] — right before the engine would finish
    /// the run. Return [`ContinueDecision::Continue`] to append a user
    /// message and keep the run going: the "completion gate" pattern,
    /// where a middleware verifies the result and sends the model back
    /// to work when the check fails.
    ///
    /// `turn.reason` is the turn's finish reason ([`FinishReason::EndTurn`],
    /// [`FinishReason::MaxTokens`], [`FinishReason::StopSequence`] or
    /// [`FinishReason::Other`]); a gate usually acts only on `EndTurn`.
    /// Never fired for [`FinishReason::Aborted`], nor for a turn whose
    /// close delivered messages from the run's [`crate::RunControl`]:
    /// the run continues to answer them, and the hook fires when the
    /// model stops again with nothing queued. `turn.new_messages` is
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
    async fn on_turn_end(&self, turn: &TurnEndInfo<'_>) -> ContinueDecision {
        ContinueDecision::Stop
    }
    /// Fired when a run terminates with a transport / setup-time
    /// error from the provider (i.e. an `Err` returned to the caller).
    /// Aborts via [`HookAction::Terminate`] /
    /// [`ToolDecision::Terminate`] / `RunConfig.cancellation` /
    /// `RunConfig.timeout` go through [`Self::on_run_finished`]
    /// instead — they are not errors.
    ///
    /// `run.usage` is what the run spent before failing, the same value the
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
    /// `run.partial_messages` holds the messages of the steps the run
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
    async fn on_run_error(&self, run: &RunErrorInfo<'_>) {}
    /// Fired when the caller drops a run's stream before the run
    /// closed: a `select!` that picks another branch, an outer timeout,
    /// a client that disconnects. Neither [`Self::on_run_finished`] nor
    /// [`Self::on_run_error`] fires for that run, so this is where
    /// per-run state keyed by `run_id` has to be released; without it,
    /// a long-lived middleware keeps every dropped run's entry forever.
    ///
    /// Each middleware gets exactly one closing hook per run:
    /// `on_run_finished`, `on_run_error` or this one. A drop that lands
    /// while the engine is already firing a closing hook reaches only
    /// the middlewares that have not been called yet; a middleware
    /// whose closing hook was interrupted at an `.await` gets nothing
    /// more, so a closing hook should release its state before its
    /// first `.await`. The hook fires even when the drop interrupts
    /// [`Self::on_run_started`], possibly for a middleware that was not
    /// asked yet, so removing an entry that is not there must be fine.
    /// A stream dropped before it was ever polled never started a run
    /// and fires nothing.
    ///
    /// It runs synchronously inside the stream's `Drop`, so it cannot
    /// await and must not block: guard per-run state with a
    /// `std::sync::Mutex` (or `try_lock` an async one) and only clean
    /// up. There is no usage or message list; the run did not finish.
    fn on_run_dropped(&self, run_id: &RunId) {}

    // tools
    /// Fired before the engine invokes a tool. Return
    /// [`ToolDecision::Skip`] to feed a synthesized error result back
    /// to the model without running the tool,
    /// [`ToolDecision::Terminate`] to abort the run, or
    /// [`ToolDecision::Defer`] to suspend the run until someone decides.
    /// This is the gating hook; for input rewriting, use
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

/// Context passed to [`ChatMiddleware::on_run_started`].
///
/// Borrows from the engine for the duration of the hook. The struct is
/// `#[non_exhaustive]` so later releases can add fields without
/// breaking middlewares; build one with [`Self::new`] to unit-test a
/// hook.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RunStartInfo<'a> {
    /// Id of the run that is starting.
    pub run_id: &'a RunId,
    /// Messages the first request is built from: the conversation
    /// history, ending with the kickoff message.
    pub messages: &'a [Message],
    /// The run's configuration, including
    /// [`RunConfig::max_iterations`].
    pub config: &'a RunConfig,
}

impl<'a> RunStartInfo<'a> {
    /// Build the context of a starting run.
    pub fn new(run_id: &'a RunId, messages: &'a [Message], config: &'a RunConfig) -> Self {
        Self {
            run_id,
            messages,
            config,
        }
    }
}

/// Identity of one step (model turn), passed to
/// [`ChatMiddleware::on_chat_request`].
///
/// The request travels as a separate hook parameter, since the hook
/// borrows it mutably.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StepInfo {
    /// Run the step belongs to.
    pub run_id: RunId,
    /// The step itself; the same `step_id` the engine puts on
    /// [`StreamChunk::StepStarted`].
    pub step_id: StepId,
    /// Zero-based index of the step in its run; the same `iteration` as
    /// on [`StreamChunk::StepStarted`].
    pub iteration: usize,
    /// The run's iteration cap, [`RunConfig::max_iterations`]. The run
    /// aborts with [`crate::AbortReason::MaxIterations`] instead of
    /// starting step number `max_iterations`.
    pub max_iterations: usize,
}

impl StepInfo {
    /// Build the identity of a step, as the first step of a run with
    /// the default cap ([`crate::DEFAULT_MAX_ITERATIONS`]). Set both
    /// with [`Self::with_iteration`].
    pub fn new(run_id: RunId, step_id: StepId) -> Self {
        Self {
            run_id,
            step_id,
            iteration: 0,
            max_iterations: crate::DEFAULT_MAX_ITERATIONS,
        }
    }

    /// Set [`Self::iteration`] and [`Self::max_iterations`].
    pub fn with_iteration(mut self, iteration: usize, max_iterations: usize) -> Self {
        self.iteration = iteration;
        self.max_iterations = max_iterations;
        self
    }
}

/// Context passed to [`ChatMiddleware::on_run_finished`].
///
/// Carries the same values as the run's terminal
/// [`StreamChunk::RunFinished`]. `#[non_exhaustive]`; build one with
/// [`Self::new`] to unit-test a hook.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RunFinishedInfo<'a> {
    /// Id of the run that finished.
    pub run_id: &'a RunId,
    /// Why the run ended; aborts arrive as [`FinishReason::Aborted`].
    pub reason: &'a FinishReason,
    /// Total usage of the run, sub-agents included.
    pub usage: &'a Usage,
    /// Everything the run added to history.
    pub new_messages: &'a [Message],
}

impl<'a> RunFinishedInfo<'a> {
    /// Build the context of a finished run.
    pub fn new(
        run_id: &'a RunId,
        reason: &'a FinishReason,
        usage: &'a Usage,
        new_messages: &'a [Message],
    ) -> Self {
        Self {
            run_id,
            reason,
            usage,
            new_messages,
        }
    }
}

/// Context passed to [`ChatMiddleware::on_turn_end`].
///
/// `#[non_exhaustive]`; build one with [`Self::new`] to unit-test a
/// hook.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TurnEndInfo<'a> {
    /// Run the turn belongs to.
    pub run_id: &'a RunId,
    /// Step (model turn) that just ended.
    pub step_id: &'a StepId,
    /// The turn's finish reason; never [`FinishReason::Aborted`] or
    /// [`FinishReason::ToolUse`].
    pub reason: &'a FinishReason,
    /// Everything the run has added so far, including this turn's
    /// assistant message.
    pub new_messages: &'a [Message],
}

impl<'a> TurnEndInfo<'a> {
    /// Build the context of an ended turn.
    pub fn new(
        run_id: &'a RunId,
        step_id: &'a StepId,
        reason: &'a FinishReason,
        new_messages: &'a [Message],
    ) -> Self {
        Self {
            run_id,
            step_id,
            reason,
            new_messages,
        }
    }
}

/// Context passed to [`ChatMiddleware::on_run_error`].
///
/// `#[non_exhaustive]`; build one with [`Self::new`] to unit-test a
/// hook.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RunErrorInfo<'a> {
    /// Id of the run that failed.
    pub run_id: &'a RunId,
    /// The error returned to the caller.
    pub error: &'a (dyn std::error::Error + Send + Sync),
    /// What the run spent before failing.
    pub usage: &'a Usage,
    /// Messages of the steps the run completed before failing.
    pub partial_messages: &'a [Message],
}

impl<'a> RunErrorInfo<'a> {
    /// Build the context of a failed run.
    pub fn new(
        run_id: &'a RunId,
        error: &'a (dyn std::error::Error + Send + Sync),
        usage: &'a Usage,
        partial_messages: &'a [Message],
    ) -> Self {
        Self {
            run_id,
            error,
            usage,
            partial_messages,
        }
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
#[derive(Debug)]
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
#[derive(Debug)]
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
    /// and fires [`ChatMiddleware::on_run_finished`].
    ///
    /// Every call of the step still gets a `tool_result` in
    /// `new_messages`, in the model's order: calls that already
    /// completed keep their result, and this call and those that did
    /// not run get an `is_error` result reading
    /// `"Tool not run: the run was aborted (<reason>)"`, each also
    /// emitted as a [`crate::StreamChunk::ToolResult`] (no tool hook
    /// fires for them). Which calls completed follows the execution
    /// order, which is not guaranteed between calls of a step; see
    /// [Tool calls within a step](ChatMiddleware#tool-calls-within-a-step).
    Terminate {
        /// Human-readable reason; threaded through
        /// [`crate::AbortReason::ToolTerminated`].
        reason: String,
    },
    /// Neither run nor refuse the call now: suspend the run until
    /// someone decides. For an approval that cannot be given while the
    /// run waits (nobody present, a scheduled agent, a client that has
    /// disconnected).
    ///
    /// The run ends with [`crate::FinishReason::Suspended`] and a
    /// [`crate::Suspension`] holding the step's calls; it is resumed
    /// later, possibly in another process, with a decision per deferred
    /// call (`Conversation::resume`, or [`crate::RunConfig::resume`]).
    /// With calls running one at a time the step stops at this call, and
    /// the calls after it wait, gates not yet run, so none of them runs
    /// before the deferred one. Within a batch of concurrency-safe calls
    /// the rest of the batch still goes through its gates and the calls
    /// not deferred run, so one suspension can carry several deferred
    /// calls.
    Defer {
        /// Why the call waits; kept in the suspension for whoever
        /// decides.
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
