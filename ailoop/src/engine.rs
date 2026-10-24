use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use crate::errors::{EngineError, RunError};
use ailoop_core::{
    AbortReason, AssistantBlock, BackgroundConfig, BackgroundEnd, BackgroundToolResult, CallState,
    CancellationToken, ChatMiddleware, ChatRequest, CompletionModel, ContinueDecision, Drained,
    FinishReason, HookAction, Inbound, Message, Resume, ResumeDecision, RunConfig, RunControl,
    RunErrorInfo, RunFinishedInfo, RunId, RunStartInfo, StepId, StepInfo, StreamChunk,
    SuspendedCall, Suspension, ToolCallInfo, ToolDecision, ToolProgress, ToolResultContent,
    TurnEndInfo, Usage, UserBlock,
};
use ailoop_history::{CompactionError, CompactionStats, History};
use ailoop_tools::{
    ProgressSink, ToolActivation, ToolContext, ToolDyn, ToolRegistry, ToolRegistryError, UsageSink,
};
use async_stream::try_stream;
use futures::{FutureExt, StreamExt, stream::BoxStream};
use serde_json::Value;

/// Stream of engine chunks for one run.
type EngineStream<'a, E> = BoxStream<'a, Result<StreamChunk, RunError<E>>>;

type AbortFuture = Pin<Box<dyn Future<Output = AbortReason> + Send>>;

/// Builds the abort future that resolves with an [`AbortReason`] when the
/// configured timeout elapses or the [`CancellationToken`] is fired.
/// Resolves to a never-completing future when neither is configured.
fn build_abort_future(
    timeout: Option<Duration>,
    cancellation: Option<CancellationToken>,
) -> AbortFuture {
    Box::pin(async move {
        let cancel_fut = async move {
            match cancellation {
                Some(token) => {
                    token.cancelled().await;
                    AbortReason::Cancelled
                }
                None => std::future::pending::<AbortReason>().await,
            }
        };
        let timer_fut = async move {
            match timeout {
                Some(d) => {
                    tokio::time::sleep(d).await;
                    AbortReason::Timeout(d)
                }
                None => std::future::pending::<AbortReason>().await,
            }
        };
        // Cancel takes priority on simultaneous fire so callers can rely
        // on `AbortReason::Cancelled` in a configured race.
        tokio::select! {
            biased;
            reason = cancel_fut => reason,
            reason = timer_fut => reason,
        }
    })
}

/// Polls `fut` against the abort future. If the abort wins the race the
/// caller receives `Err(reason)` and `fut` is dropped — which cancels
/// any in-flight HTTP request, retry-backoff sleep, or tool execution
/// behind it.
async fn race_abort<F, T>(fut: F, abort: &mut AbortFuture) -> Result<T, AbortReason>
where
    F: Future<Output = T>,
{
    tokio::select! {
        biased;
        reason = &mut *abort => Err(reason),
        value = fut => Ok(value),
    }
}

/// Owns a run's closing hooks. Every middleware gets exactly one of
/// `on_run_finished` / `on_run_error` through [`Self::finished`] /
/// [`Self::failed`]; when the stream is dropped before (or while) that
/// happens, `Drop` fires `on_run_dropped` on the ones not reached yet.
/// It lives inside the stream body, so a stream that is never polled
/// never builds one and fires nothing.
struct RunGuard {
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
    run_id: RunId,
    /// How many middlewares, in order, have had their closing hook
    /// called. Counted before the hook is awaited: one interrupted at
    /// an `.await` has already been told the run is over.
    closed: AtomicUsize,
    /// The token tools see as [`ToolContext::cancellation`]. Cancelled
    /// when the run aborts or is dropped before it ends, so work a tool
    /// spawned stops even though only the tool future is dropped.
    tool_cancellation: CancellationToken,
    /// Set once the run reached its closing hooks (finished, aborted or
    /// failed); a drop before that cancels `tool_cancellation` and
    /// hands `taken` back to `control`.
    ended: AtomicBool,
    /// The run's [`RunControl`], if any.
    control: Option<RunControl>,
    /// Messages the run took from `control`. They live only in the
    /// run's messages, which a dropped run never commits, so a drop
    /// puts them back in the queue.
    taken: std::sync::Mutex<Vec<Inbound>>,
    /// Set once `RunFinished` reached the consumer: from then on the
    /// run is committed, whatever happens to the stream.
    handed_over: Arc<AtomicBool>,
    /// Set when the run fails with an error, whose delivered messages
    /// travel in its partial messages instead of going back in the
    /// queue.
    errored: AtomicBool,
    /// Where the real results of calls this run moves to the
    /// background go when the run does not take them.
    background: BackgroundConfig,
    /// The id under which `control` hands this run background results,
    /// while it takes them.
    background_run: Option<u64>,
    /// Background results the run took. They are reported once the run
    /// is over, when it is known whether its messages count: delivered
    /// if it handed `RunFinished` over or failed (they are in the
    /// error's partial messages), not delivered if it was rolled back.
    took_results: std::sync::Mutex<Vec<BackgroundToolResult>>,
}

impl RunGuard {
    fn new(
        middlewares: Vec<Arc<dyn ChatMiddleware>>,
        run_id: RunId,
        tool_cancellation: CancellationToken,
        control: Option<RunControl>,
        handed_over: Arc<AtomicBool>,
        background: BackgroundConfig,
    ) -> Self {
        let background_run = control.as_ref().map(RunControl::open_run);
        Self {
            middlewares,
            run_id,
            closed: AtomicUsize::new(0),
            tool_cancellation,
            ended: AtomicBool::new(false),
            control,
            taken: std::sync::Mutex::default(),
            handed_over,
            errored: AtomicBool::new(false),
            background,
            background_run,
            took_results: std::sync::Mutex::default(),
        }
    }

    /// Drains the run's [`RunControl`]: queued messages and the
    /// background results waiting for this run, remembering what it
    /// took so a drop can put the messages back and the results can be
    /// reported when the run is over. Empty without a control.
    fn drain(&self) -> Drained {
        let (Some(control), Some(run)) = (&self.control, self.background_run) else {
            return Drained::default();
        };
        let drained = control.drain(run);
        if !drained.messages.is_empty() {
            self.taken
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(drained.messages.iter().cloned());
        }
        if !drained.results.is_empty() {
            self.took_results
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(drained.results.iter().map(|(_, result)| result.clone()));
        }
        drained
    }

    /// Stops the run taking background results; those it took but did
    /// not drain go to the callback as not delivered. Idempotent.
    fn close_background(&self) {
        if let (Some(control), Some(run)) = (&self.control, self.background_run) {
            for result in control.close_run(run) {
                self.background.report(result);
            }
        }
    }

    /// Unregisters every call from the run's [`RunControl`]: none of
    /// them can be cancelled any more.
    fn clear_calls(&self) {
        if let Some(control) = &self.control {
            control.clear_calls();
        }
    }

    fn middlewares(&self) -> &[Arc<dyn ChatMiddleware>] {
        &self.middlewares
    }

    /// The middlewares whose closing hook is still due, marking each
    /// as closed when handed out.
    fn pending(&self) -> impl Iterator<Item = &Arc<dyn ChatMiddleware>> {
        self.middlewares
            .iter()
            .skip(self.closed.load(Ordering::Relaxed))
            .inspect(|_| {
                self.closed.fetch_add(1, Ordering::Relaxed);
            })
    }

    async fn finished(&self, reason: &FinishReason, usage: &Usage, new_messages: &[Message]) {
        self.ended.store(true, Ordering::Relaxed);
        self.close_background();
        for mw in self.pending() {
            mw.on_run_finished(&RunFinishedInfo::new(
                &self.run_id,
                reason,
                usage,
                new_messages,
            ))
            .await;
        }
    }

    async fn failed(
        &self,
        err: &(dyn std::error::Error + Send + Sync),
        usage: &Usage,
        partial_messages: &[Message],
    ) {
        self.ended.store(true, Ordering::Relaxed);
        self.errored.store(true, Ordering::Relaxed);
        self.close_background();
        for mw in self.pending() {
            mw.on_run_error(&RunErrorInfo::new(
                &self.run_id,
                err,
                usage,
                partial_messages,
            ))
            .await;
        }
    }
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        // However the run ended, none of its calls can be cancelled now,
        // and it takes no more background results.
        self.clear_calls();
        self.close_background();
        if !self.ended.load(Ordering::Relaxed) {
            self.tool_cancellation.cancel();
        }
        // A run dropped before it handed `RunFinished` over is rolled
        // back, even if its closing hooks had started.
        let committed =
            self.handed_over.load(Ordering::SeqCst) || self.errored.load(Ordering::Relaxed);
        if !committed && let Some(control) = &self.control {
            let taken = std::mem::take(self.taken.get_mut().unwrap_or_else(|e| e.into_inner()));
            control.restore(taken);
        }
        // The background results it took are not put back: a rolled
        // back run leaves them to the callback, as not delivered.
        let took = std::mem::take(
            self.took_results
                .get_mut()
                .unwrap_or_else(|e| e.into_inner()),
        );
        for mut result in took {
            result.delivered = committed;
            self.background.report(result);
        }
        for mw in self.pending() {
            mw.on_run_dropped(&self.run_id);
        }
    }
}

/// Fires the `on_run_finished` + `on_chunk` hook pair for an aborted
/// run and returns the `RunFinished` chunk for the caller to yield.
/// Centralised so every abort site (hook terminate, tool terminate,
/// timeout, cancellation) follows the same persistence discipline.
async fn fire_abort_hooks(
    guard: &RunGuard,
    reason: AbortReason,
    usage: Usage,
    new_messages: Vec<Message>,
) -> StreamChunk {
    let middlewares = guard.middlewares();
    let run_id = &guard.run_id;
    let finish_reason = FinishReason::Aborted(reason);
    // Tools see the abort before any closing hook runs.
    guard.tool_cancellation.cancel();
    guard.finished(&finish_reason, &usage, &new_messages).await;
    run_finished_chunk(middlewares, run_id, finish_reason, usage, new_messages).await
}

/// Builds the terminal [`StreamChunk::RunFinished`] and runs it through
/// the middleware chain. A middleware may rewrite its fields in
/// `on_chunk_mut`, but not its variant: if the chunk comes back as
/// anything else, the original is restored, so every stream still ends
/// in `RunFinished`.
async fn run_finished_chunk(
    middlewares: &[Arc<dyn ChatMiddleware>],
    run_id: &RunId,
    reason: FinishReason,
    usage: Usage,
    new_messages: Vec<Message>,
) -> StreamChunk {
    let original = (reason.clone(), usage, new_messages.clone());
    let mut chunk = StreamChunk::run_finished(run_id.clone(), reason, usage, new_messages);
    for mw in middlewares {
        mw.on_chunk_mut(&mut chunk).await;
    }
    if !matches!(chunk, StreamChunk::RunFinished { .. }) {
        let (reason, usage, new_messages) = original;
        chunk = StreamChunk::run_finished(run_id.clone(), reason, usage, new_messages);
    }
    for mw in middlewares {
        mw.on_chunk(&chunk).await;
    }
    chunk
}

/// Closes a step the run leaves part-way through. `tools_result` holds
/// the results the step already has; each call in `unanswered` gets
/// one too, in order, and its `ToolResult` chunk goes out through
/// `on_chunk_mut` / `on_chunk` without any tool hook (the tool never
/// ran). The step's results are stored, the run's messages committed,
/// and the abort hooks fired. Returns the chunks to yield, with
/// `RunFinished` last.
#[allow(clippy::too_many_arguments)]
async fn abort_step(
    guard: &RunGuard,
    step_id: &StepId,
    reason: AbortReason,
    usage: Usage,
    run_msgs: &mut RunMessages<'_>,
    mut tools_result: Vec<UserBlock>,
    unanswered: impl IntoIterator<Item = PendingCall>,
) -> Vec<StreamChunk> {
    let middlewares = guard.middlewares();
    let run_id = &guard.run_id;
    // No call of an aborted step can be cancelled any more.
    guard.clear_calls();
    let mut chunks = Vec::new();
    for call in unanswered {
        let (id, content) = match call {
            PendingCall::Run { id, .. } | PendingCall::Decided { id, .. } => {
                (id, not_run_result(&reason))
            }
            PendingCall::Rejected { id, content } => (id, content),
            // Moved to the background: it keeps running, so it keeps
            // its provisional result rather than "not run".
            PendingCall::Backgrounded { id, content } => {
                let mut chunk =
                    StreamChunk::tool_backgrounded(run_id.clone(), step_id.clone(), id.clone());
                for mw in middlewares {
                    mw.on_chunk_mut(&mut chunk).await;
                }
                for mw in middlewares {
                    mw.on_chunk(&chunk).await;
                }
                chunks.push(chunk);
                (id, content)
            }
            // Its chunk went out in the suspended run.
            PendingCall::Answered { id, content } => {
                tools_result.push(UserBlock::tool_result(id, content));
                continue;
            }
        };
        let mut chunk =
            StreamChunk::tool_result(run_id.clone(), step_id.clone(), id.clone(), content.clone());
        for mw in middlewares {
            mw.on_chunk_mut(&mut chunk).await;
        }
        for mw in middlewares {
            mw.on_chunk(&chunk).await;
        }
        chunks.push(chunk);
        tools_result.push(UserBlock::tool_result(id, content));
    }
    if !tools_result.is_empty() {
        // Results in the order of the step's calls, which the paths
        // above do not always produce (a deferred call answered after
        // the batch's other calls).
        let order: Vec<&str> = run_msgs
            .context()
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::Assistant { blocks } => Some(
                    blocks
                        .iter()
                        .filter_map(|block| match block {
                            AssistantBlock::ToolCall { call_id, .. } => Some(call_id.as_str()),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        let position = |block: &UserBlock| match block {
            UserBlock::ToolResult { call_id, .. } => order.iter().position(|id| id == call_id),
            _ => None,
        };
        tools_result.sort_by_key(|block| position(block).unwrap_or(usize::MAX));
        run_msgs.push(Message::User {
            blocks: tools_result,
        });
    }
    let new_messages = run_msgs.finish();
    chunks.push(fire_abort_hooks(guard, reason, usage, new_messages).await);
    chunks
}

/// Aborts a run before its first step. A resumed run that never got
/// to the suspended step still closes it, as [`abort_step`] would: the
/// calls answered in the suspended run keep their result and every
/// other one is answered as not run, so the history the run commits has
/// no `tool_use` without its `tool_result`.
async fn abort_before_step(
    guard: &RunGuard,
    resuming: Option<Resume>,
    reason: AbortReason,
    usage: Usage,
    run_msgs: &mut RunMessages<'_>,
) -> Vec<StreamChunk> {
    let Some(Resume { suspension, .. }) = resuming else {
        let new_messages = run_msgs.finish();
        return vec![fire_abort_hooks(guard, reason, usage, new_messages).await];
    };
    let unanswered: Vec<PendingCall> = suspension
        .calls
        .into_iter()
        .map(|call| match call.state {
            CallState::Answered { content } => PendingCall::Answered {
                id: call.call_id,
                content,
            },
            _ => PendingCall::Run {
                id: call.call_id,
                name: call.name,
                args: call.args,
            },
        })
        .collect();
    abort_step(
        guard,
        &suspension.step_id,
        reason,
        usage,
        run_msgs,
        Vec::new(),
        unanswered,
    )
    .await
}

/// The result recorded for a deferred call denied on resume.
fn denied_result(reason: &str) -> ToolResultContent {
    ToolResultContent::error(format!("Tool denied: {reason}"))
}

/// The result recorded for a call `RunControl::cancel_tool` stopped
/// before the tool was invoked.
fn cancelled_before_run() -> ToolResultContent {
    ToolResultContent::error("Tool not run: cancelled by the caller")
}

/// The result recorded for a call `RunControl::cancel_tool` stopped
/// while the tool ran. It may have had its effect already, so the text
/// does not claim otherwise.
fn cancelled_while_running() -> ToolResultContent {
    ToolResultContent::error(
        "Tool cancelled by the caller while running; it may have partly completed",
    )
}

/// The result recorded for a call the run was aborted before running.
fn not_run_result(reason: &AbortReason) -> ToolResultContent {
    ToolResultContent::error(format!("Tool not run: the run was aborted ({reason})"))
}

/// Turns an `Err` into a [`RunError`] carrying the run's usage so far
/// and its completed steps, and fires `on_run_error` with both.
macro_rules! bail_with_hooks {
    ($result: expr, $guard: expr, $usage: expr, $run_msgs: expr) => {
        match $result {
            Ok(v) => Ok(v),
            Err(e) => {
                let err: EngineError<_> = e.into();
                let usage: Usage = $usage;
                let partial = $run_msgs.completed().to_vec();
                $guard.failed(&err, &usage, &partial).await;
                Err(RunError::new(err, usage, partial))
            }
        }
    };
}

/// In-run context management switches for the [`History`]-backed
/// engine path. Set from `ConversationBuilder`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ContextOptions {
    /// Run [`History::compact_if_needed`] before every provider call
    /// after the first one.
    pub(crate) compact_between_iterations: bool,
    /// On a context-overflow setup error, force a compaction and
    /// reissue the request once.
    pub(crate) recover_from_overflow: bool,
}

/// A [`History`] the engine appends to in place for the duration of a
/// run, with rollback: unless the run handed its `RunFinished` over to
/// the consumer, dropping this value restores the messages and pin
/// mask captured at construction. That keeps the contract that a run
/// ending in `Err` (or a stream dropped mid-run, closing hooks
/// included) leaves the history exactly as it was when the run
/// started, even when the run compacted it along the way. The steps
/// completed before an `Err` travel in [`RunError`] instead.
struct ManagedHistory<'a> {
    history: &'a mut History,
    rollback: Option<(Vec<Message>, Vec<bool>)>,
    options: ContextOptions,
    /// The run's [`RunGuard::handed_over`], set by [`run_engine`].
    handed_over: Arc<AtomicBool>,
}

impl Drop for ManagedHistory<'_> {
    fn drop(&mut self) {
        if self.handed_over.load(Ordering::SeqCst) {
            return;
        }
        if let Some((messages, pinned)) = self.rollback.take() {
            self.history
                .replace_messages(messages, pinned)
                .expect("rollback state was captured from the same history");
        }
    }
}

enum RunContext<'a> {
    /// [`run_chat`]: a caller-owned message vector, no compaction.
    Plain(Vec<Message>),
    /// `Conversation`: the live history, compactable mid-run.
    Managed(ManagedHistory<'a>),
}

/// The context sent to the model plus an append-only log of what this
/// run added. The two diverge once the context is compacted mid-run:
/// `RunFinished.new_messages` and `StepFinished.new_messages_so_far`
/// keep reporting everything the run produced, while the request (and
/// the final history) carry the compacted context.
struct RunMessages<'a> {
    context: RunContext<'a>,
    new_messages: Vec<Message>,
    /// Length of the `new_messages` prefix made of finished steps,
    /// reported as [`RunError::partial_messages`] if the run fails.
    completed_len: usize,
}

impl<'a> RunMessages<'a> {
    fn context(&self) -> &[Message] {
        match &self.context {
            RunContext::Plain(messages) => messages,
            RunContext::Managed(managed) => managed.history.messages(),
        }
    }

    fn push(&mut self, message: Message) {
        match &mut self.context {
            RunContext::Plain(messages) => messages.push(message.clone()),
            RunContext::Managed(managed) => managed.history.add_message(message.clone()),
        }
        self.new_messages.push(message);
    }

    fn new_so_far(&self) -> &[Message] {
        &self.new_messages
    }

    /// Marks everything pushed so far as a finished step. Called once
    /// per step, after its tool results are pushed, so the prefix never
    /// ends on a `tool_use` without its `tool_result`.
    fn complete_step(&mut self) {
        self.completed_len = self.new_messages.len();
    }

    fn completed(&self) -> &[Message] {
        &self.new_messages[..self.completed_len]
    }

    /// Commit the run's changes to the managed history and hand back
    /// the run's messages. Called exactly once, right before the
    /// terminal `RunFinished` is built.
    fn finish(&mut self) -> Vec<Message> {
        std::mem::take(&mut self.new_messages)
    }

    fn managed(&mut self) -> Option<&mut ManagedHistory<'a>> {
        match &mut self.context {
            RunContext::Plain(_) => None,
            RunContext::Managed(managed) => Some(managed),
        }
    }
}

/// Runs the history's compaction strategy (forced, or only when over
/// budget) and reports whether it made progress, i.e. whether
/// [`History::estimated_tokens`] went down.
async fn compact(
    history: &mut History,
    force: bool,
) -> Result<Option<(CompactionStats, bool)>, CompactionError> {
    let before = history.estimated_tokens();
    let report = if force {
        history.force_compact().await?
    } else {
        match history.compact_if_needed().await? {
            Some(report) => report,
            None => return Ok(None),
        }
    };
    let progressed = history.estimated_tokens() < before;
    Ok(Some((report, progressed)))
}

/// Builds the [`StreamChunk::HistoryCompacted`] for `report` and runs
/// it through the middleware chain like every engine-emitted chunk.
async fn history_compacted_chunk(
    middlewares: &[Arc<dyn ChatMiddleware>],
    run_id: &RunId,
    report: CompactionStats,
) -> StreamChunk {
    let mut chunk = StreamChunk::history_compacted(
        run_id.clone(),
        report.before,
        report.after,
        report.strategy,
    );
    for mw in middlewares {
        mw.on_chunk_mut(&mut chunk).await;
    }
    for mw in middlewares {
        mw.on_chunk(&chunk).await;
    }
    chunk
}

/// Drives the agent loop end-to-end against a fixed `messages` slice and
/// the supplied [`ToolRegistry`], yielding a stream of [`StreamChunk`]s
/// that always terminates with a [`StreamChunk::RunFinished`].
///
/// Most callers should reach the engine through [`Conversation::run`] /
/// [`Conversation::stream`] instead — those wire history management,
/// system-prompt assembly, and per-request defaults that this entry
/// point leaves to the caller. Use this directly only when you need
/// engine-level access without a `History` in the loop.
///
/// Without a `History` there is nothing to compact: the context grows
/// with every tool result for the whole run, and a context-window
/// overflow from the provider surfaces as [`EngineError::Model`]. The
/// in-run compaction and overflow recovery described on
/// [`ConversationBuilder::compact_between_iterations`] and
/// [`ConversationBuilder::recover_from_context_overflow`] are only
/// available through [`Conversation`](crate::Conversation).
///
/// A run that fails yields a [`RunError`] whose
/// [`partial_messages`](RunError::partial_messages) hold the steps
/// completed before the failure.
///
/// When a turn requests several tools, the order in which those calls
/// run is not guaranteed; tool results are still recorded in the order
/// of the model's calls. See
/// [Tool calls within a step](ChatMiddleware#tool-calls-within-a-step).
///
/// Every `tool_use` in `RunFinished.new_messages` has its `tool_result`,
/// with one exception: a run that suspends on a deferred call
/// ([`FinishReason::Suspended`]) ends on its step's assistant message,
/// and the step's results travel in the [`StreamChunk::RunSuspended`]
/// chunk before `RunFinished`. Keep the two together; resume by passing
/// the history and a [`Resume`](ailoop_core::Resume) in
/// `RunConfig::resume`.
///
/// [`Conversation::run`]: crate::Conversation::run
/// [`Conversation::stream`]: crate::Conversation::stream
/// [`ConversationBuilder::compact_between_iterations`]: crate::ConversationBuilder::compact_between_iterations
/// [`ConversationBuilder::recover_from_context_overflow`]: crate::ConversationBuilder::recover_from_context_overflow
pub async fn run_chat<'a, M: CompletionModel>(
    model: &'a M,
    messages: Vec<Message>,
    tools: &'a ToolRegistry,
    config: RunConfig,
) -> Result<BoxStream<'a, Result<StreamChunk, RunError<M::Error>>>, RunError<M::Error>> {
    Ok(run_engine(
        model,
        RunContext::Plain(messages),
        tools,
        config,
        |_| false,
        None,
    ))
}

/// [`History`]-backed engine entry used by `Conversation`. The run
/// appends to `history` in place and commits on `RunFinished`; on `Err`
/// or when the stream is dropped mid-run, `history` is rolled back to
/// its state at the call. `is_overflow` classifies model setup errors
/// for [`ContextOptions::recover_from_overflow`]. `usage_parent`, when
/// set, receives every token this run spends (own turns and tool
/// reports) as it happens; `SubAgentTool` passes its own
/// `ToolContext::usage_sink` here.
pub(crate) fn run_with_history<'a, M: CompletionModel>(
    model: &'a M,
    history: &'a mut History,
    tools: &'a ToolRegistry,
    config: RunConfig,
    options: ContextOptions,
    is_overflow: fn(&M::Error) -> bool,
    usage_parent: Option<UsageSink>,
) -> EngineStream<'a, M::Error> {
    let rollback = Some((history.messages().to_vec(), history.pinned().to_vec()));
    let managed = ManagedHistory {
        history,
        rollback,
        options,
        handed_over: Arc::default(),
    };
    run_engine(
        model,
        RunContext::Managed(managed),
        tools,
        config,
        is_overflow,
        usage_parent,
    )
}

fn run_engine<'a, M: CompletionModel>(
    model: &'a M,
    mut context: RunContext<'a>,
    tools: &'a ToolRegistry,
    mut config: RunConfig,
    is_overflow: fn(&M::Error) -> bool,
    usage_parent: Option<UsageSink>,
) -> EngineStream<'a, M::Error> {
    // A resumed run starts by finishing the suspended step instead of
    // asking the model; `resumed_from` lets observers correlate it.
    // The run is committed once its `RunFinished` reaches the consumer,
    // atomically with that delivery: a stream dropped any earlier, its
    // closing hooks included, rolls the history back and puts the
    // messages it took from its `RunControl` back in the queue.
    let handed_over = Arc::new(AtomicBool::new(false));
    if let RunContext::Managed(managed) = &mut context {
        managed.handed_over = handed_over.clone();
    }
    let committed = handed_over.clone();
    let mut resuming = config.resume.take();
    if let Some(resume) = &resuming {
        config.resumed_from = Some(resume.suspension.run_id.clone());
    }
    let mut run_msgs = RunMessages {
        context,
        new_messages: Vec::new(),
        completed_len: 0,
    };
    let run_id = config.run_id.clone().unwrap_or_default();
    // Snapshot the catalog + active set once at run start. Per-turn
    // `req.tools` and per-dispatch `ToolContext`s are built from these
    // shared handles, so any mutation a tool makes via
    // `ctx.tools().activate(...)` is visible on the next turn without
    // touching the underlying `ToolRegistry`.
    let catalog = tools.catalog_arc();
    let active_snapshot = tools.snapshot_active();
    // The token every per-dispatch `ToolContext` gets: a child of
    // `RunConfig.cancellation` (or a fresh token when there is none),
    // so the caller cancelling reaches the tools, and the run can
    // cancel it on any abort (timeout, `Terminate`, a dropped stream)
    // without touching the caller's token. Built once outside the
    // loop and cloned per dispatch so every tool sees the same handle.
    let tool_cancellation = config
        .cancellation
        .as_ref()
        .map(CancellationToken::child_token)
        .unwrap_or_default();
    // Usage tools report through `ToolContext::report_usage` (e.g. a
    // sub-agent's child run). Reports land here the moment they are
    // made, so an abort that drops a tool mid-call still counts what
    // it already spent. When this run is itself a sub-agent, the sink
    // forwards into the parent's, and so on up to the outermost run.
    let delegated = match &usage_parent {
        Some(parent) => UsageSink::forwarding_to(parent.clone()),
        None => UsageSink::new(),
    };
    let stream = try_stream! {
        // The abort future resolves with a textual reason when either
        // the timeout elapses or the cancellation token fires; until
        // then it is `pending`, so wrapping any await with
        // `race_abort(_, &mut abort_fut)` is a no-op on the happy path.
        let mut abort_fut: AbortFuture = build_abort_future(
            config.timeout,
            config.cancellation.clone(),
        );
        // Fires `on_run_dropped` if the caller drops the stream before
        // every middleware got its closing hook.
        let guard = RunGuard::new(
            config.middlewares.clone(),
            run_id.clone(),
            tool_cancellation.clone(),
            config.control.clone(),
            handed_over.clone(),
            config.background.clone(),
        );
        // Progress tools report while they run, `(call_id, update)`.
        // Bounded and fed with `try_send`, so a chatty tool never blocks
        // and never grows memory: past `PROGRESS_BUFFER` updates are
        // dropped. Drained into `ToolProgress` chunks while a batch runs.
        let (progress_tx, mut progress_rx) =
            tokio::sync::mpsc::channel::<(String, ToolProgress)>(PROGRESS_BUFFER);

        for mw in &config.middlewares {
            let action = match race_abort(
                mw.on_run_started(&RunStartInfo::new(&run_id, run_msgs.context(), &config)),
                &mut abort_fut,
            ).await {
                Ok(a) => a,
                Err(reason) => {
                    let chunks = abort_before_step(
                        &guard, resuming.take(), reason, Usage::default(), &mut run_msgs,
                    ).await;
                    for chunk in chunks { yield chunk; }
                    return;
                }
            };
            match action {
                HookAction::Continue => {},
                HookAction::Terminate {reason} => {
                    let chunks = abort_before_step(
                        &guard, resuming.take(), AbortReason::terminated(reason),
                        Usage::default(), &mut run_msgs,
                    ).await;
                    for chunk in chunks { yield chunk; }
                    return;
                }
                _ => {}
            };
        }

        let mut chunk = StreamChunk::run_started(run_id.clone());
        for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
        yield chunk;

        let mut iteration = 0;
        let mut finish_reason = FinishReason::EndTurn;
        // Spend of this run's own provider turns. The run total adds
        // `delegated` (tool reports) wherever `RunFinished` is built.
        let mut usage_run = Usage::default();
        // Cleared for the rest of the run once a proactive compaction
        // fails to bring the history under budget: with the built-in
        // strategies only turns before this run's kickoff can be
        // dropped, so trying again every iteration would just repeat
        // the work (and, with `SummarizeStrategy`, a model call).
        let mut proactive_compaction = run_msgs
            .managed()
            .is_some_and(|m| m.options.compact_between_iterations);

        loop {
            if iteration >= config.max_iterations {
                // Every previous iteration pushed its tool results before
                // looping, so the run's messages have no dangling tool_use;
                // a resumed run stopped before its first step closes the
                // suspended one.
                let chunks = abort_before_step(
                    &guard, resuming.take(), AbortReason::MaxIterations(config.max_iterations),
                    usage_run + delegated.total(), &mut run_msgs,
                ).await;
                for chunk in chunks { yield chunk; }
                return;
            }

            // Iteration 0 is covered by the compaction `Conversation`
            // runs before the engine starts.
            if iteration > 0 && proactive_compaction {
                let managed = run_msgs.managed().expect("proactive compaction implies a managed history");
                let outcome = match race_abort(compact(managed.history, false), &mut abort_fut).await {
                    Ok(outcome) => outcome,
                    Err(reason) => {
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &guard, reason, usage_run + delegated.total(), new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                };
                match outcome {
                    Ok(None) => {}
                    Ok(Some((report, progressed))) => {
                        if !progressed || managed.history.needs_compaction() {
                            proactive_compaction = false;
                        }
                        yield history_compacted_chunk(&config.middlewares, &run_id, report).await;
                    }
                    // Best effort: nothing to drop yet. The overflow
                    // recovery below is the safety net.
                    Err(CompactionError::NotEnoughHistory) => proactive_compaction = false,
                    Err(e) => bail_with_hooks!(Err::<(), _>(EngineError::Context(e)), &guard, usage_run + delegated.total(), run_msgs)?,
                }
            }

            let step_id = StepId::new();
            let mut chunk = StreamChunk::step_started(run_id.clone(), step_id.clone(), iteration);
            for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
            for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
            yield chunk;

            let mut assistant_blocks = Vec::new();
            let mut text_buf = String::new();
            let mut reasoning_buf = String::new();

            let mut tool_calls = Vec::new();
            // Every call of the step in the model's order, with its
            // original arguments: what a suspension records.
            let mut step_calls: Vec<(String, String, Value)> = Vec::new();
            // Each call's own token, a child of the run's, so
            // `RunControl::cancel_tool` stops one call without touching
            // its siblings. A call is registered from its
            // `ToolCallStarted` on, while the turn still streams.
            let mut call_tokens: HashMap<String, CancellationToken> = HashMap::new();

            if let Some(resume) = resuming.take() {
                // Finishing a suspended step: its assistant message is
                // already in the history, so there is no request. Its
                // calls go through the dispatch below as recorded:
                // answered ones keep their result, deferred ones carry
                // their decision, waiting ones take their gates.
                finish_reason = FinishReason::ToolUse;
                let Resume { suspension, mut decisions, .. } = resume;
                for call in suspension.calls {
                    step_calls.push((call.call_id.clone(), call.name.clone(), call.args.clone()));
                    let SuspendedCall { call_id: id, name, args, state, .. } = call;
                    tool_calls.push(match state {
                        CallState::Answered { content } => PendingCall::Answered { id, content },
                        CallState::Deferred { .. } => {
                            // `Resume::new` checked every deferred call has one.
                            let decision = decisions
                                .remove(&id)
                                .unwrap_or_else(|| ResumeDecision::deny("no decision was given"));
                            PendingCall::Decided { id, name, args, decision }
                        }
                        _ => PendingCall::Run { id, name, args },
                    });
                }
            } else {

            // At most one overflow recovery per iteration: the request
            // is rebuilt from the compacted context and every
            // `on_chat_request` runs again for the same `step_id`.
            let mut overflow_recovered = false;
            let mut adapter_stream = loop {
                let mut req = ChatRequest::new(run_msgs.context().to_vec(), config.max_tokens);
                // Re-read the active set fresh each turn — a tool from the
                // previous step may have called `ctx.tools().activate(...)`
                // and that change must reach the model on this turn.
                req.tools = Some({
                    // Poisoning only means another thread panicked
                    // mid-update; the set itself is still consistent.
                    let active = active_snapshot.lock().unwrap_or_else(|e| e.into_inner());
                    catalog
                        .iter()
                        .filter(|(name, _)| active.contains(*name))
                        .map(|(_, tool)| tool.tool_definition())
                        .collect()
                });
                req.system_prompt = config.system_prompt.clone();

                let mut aborted = None;
                let step = StepInfo::new(run_id.clone(), step_id.clone())
                    .with_iteration(iteration, config.max_iterations);
                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_chat_request(&step, &mut req),
                        &mut abort_fut,
                    ).await {
                        aborted = Some(reason);
                        break;
                    }
                }
                if let Some(reason) = aborted {
                    let new_messages = run_msgs.finish();
                    let chunk = fire_abort_hooks(
                        &guard, reason, usage_run + delegated.total(), new_messages,
                    ).await;
                    yield chunk;
                    return;
                }

                let error = match race_abort(model.chat_stream(req), &mut abort_fut).await {
                    Ok(Ok(stream)) => break stream,
                    Ok(Err(error)) => error,
                    Err(reason) => {
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &guard, reason, usage_run + delegated.total(), new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                };

                let recoverable = is_overflow(&error)
                    && run_msgs.managed().is_some_and(|m| m.options.recover_from_overflow);
                if !recoverable {
                    bail_with_hooks!(Err::<(), _>(EngineError::Model(error)), &guard, usage_run + delegated.total(), run_msgs)?;
                    unreachable!();
                }
                if overflow_recovered {
                    bail_with_hooks!(Err::<(), _>(EngineError::ContextOverflow(error)), &guard, usage_run + delegated.total(), run_msgs)?;
                    unreachable!();
                }
                overflow_recovered = true;

                let managed = run_msgs.managed().expect("recoverable implies a managed history");
                let outcome = match race_abort(compact(managed.history, true), &mut abort_fut).await {
                    Ok(outcome) => outcome,
                    Err(reason) => {
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &guard, reason, usage_run + delegated.total(), new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                };
                match outcome {
                    Ok(Some((report, true))) => {
                        yield history_compacted_chunk(&config.middlewares, &run_id, report).await;
                    }
                    // Nothing the strategy can drop, or dropping it did
                    // not shrink the prompt: resending would fail again.
                    Ok(_) | Err(CompactionError::NotEnoughHistory) => {
                        bail_with_hooks!(Err::<(), _>(EngineError::ContextOverflow(error)), &guard, usage_run + delegated.total(), run_msgs)?;
                    }
                    Err(e) => bail_with_hooks!(Err::<(), _>(EngineError::Context(e)), &guard, usage_run + delegated.total(), run_msgs)?,
                }
            };

            loop {
                let next = match race_abort(adapter_stream.next(), &mut abort_fut).await {
                    Ok(n) => n,
                    Err(reason) => {
                        // Preserve any complete blocks the assistant has
                        // produced before the abort so history stays
                        // consistent — only blocks closed by their `*End`
                        // chunk are in `assistant_blocks`, partial tool
                        // calls (start without end) are not. The finished
                        // calls never ran, so `abort_step` answers them.
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::text(text_buf));
                        }
                        if !assistant_blocks.is_empty() {
                            run_msgs.push(Message::Assistant { blocks: assistant_blocks });
                        }
                        let chunks = abort_step(
                            &guard, &step_id, reason, usage_run + delegated.total(),
                            &mut run_msgs, Vec::new(), tool_calls,
                        ).await;
                        for chunk in chunks { yield chunk; }
                        return;
                    }
                };
                let chunk = match next {
                    Some(c) => c,
                    None => break,
                };
                let chunk = bail_with_hooks!(chunk.map_err(EngineError::Model), &guard, usage_run + delegated.total(), run_msgs)?;

                // Mutating phase first: every `_mut` runs before any
                // observer, so the engine itself, the assistant-history
                // builder below, and the stream consumer all see the
                // same fully-mutated chunk. See the trait doc on
                // `on_chunk_mut` for the contract.
                let mut chunk = chunk;
                for mw in &config.middlewares {
                    mw.on_chunk_mut(&mut chunk).await;
                }
                if let StreamChunk::ToolCallStarted { call_id, .. } = &chunk {
                    let token = tool_cancellation.child_token();
                    if let Some(control) = &config.control {
                        control.register_call(call_id.clone(), token.clone());
                    }
                    call_tokens.insert(call_id.clone(), token);
                }
                for mw in &config.middlewares {
                    mw.on_chunk(&chunk).await;
                }

                match &chunk {
                    StreamChunk::TextDelta { delta } => {
                        text_buf.push_str(delta);
                    },
                    StreamChunk::ReasoningDelta { delta } => {
                        reasoning_buf.push_str(delta);
                    },
                    StreamChunk::ToolCallStarted { .. } => {
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::text(std::mem::take(&mut text_buf)));
                        }
                    },
                    StreamChunk::ToolCallFinished { call_id: id, name, args } => {
                        assistant_blocks.push(AssistantBlock::tool_call(id.clone(), name.clone(), args.clone()));
                        step_calls.push((id.clone(), name.clone(), args.clone()));
                        tool_calls.push(PendingCall::Run { id: id.clone(), name: name.clone(), args: args.clone() })
                    },
                    StreamChunk::ToolCallMalformed { call_id: id, name, raw, error } => {
                        // Providers require an object input on replay; the
                        // raw text travels back in the error result instead.
                        assistant_blocks.push(AssistantBlock::tool_call(
                            id.clone(), name.clone(), Value::Object(Default::default()),
                        ));
                        step_calls.push((id.clone(), name.clone(), Value::Object(Default::default())));
                        tool_calls.push(PendingCall::Rejected {
                            id: id.clone(),
                            content: malformed_args_result(name, raw, error),
                        })
                    },
                    StreamChunk::ReasoningFinished { signature } => {
                        // Reasoning blocks must keep their original position
                        // relative to text and tool_use; flush any pending
                        // text first so order on replay matches the wire.
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::text(std::mem::take(&mut text_buf)));
                        }
                        assistant_blocks.push(AssistantBlock::Reasoning {
                            text: std::mem::take(&mut reasoning_buf),
                            signature: signature.clone(),
                        });
                    },
                    StreamChunk::RedactedReasoningBlock { data } => {
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::text(std::mem::take(&mut text_buf)));
                        }
                        assistant_blocks.push(AssistantBlock::RedactedReasoning {
                            data: data.clone(),
                        });
                    },
                    StreamChunk::TurnFinished { reason, usage, .. } => {
                        finish_reason = reason.clone();
                        usage_run += *usage;
                        // Forward this run's own spend to an enclosing
                        // run (sub-agent) as it happens, so it counts
                        // there even if this run is later dropped.
                        if let Some(parent) = &usage_parent {
                            parent.report(*usage);
                        }
                        continue;
                    },
                    _=> ()
                }


                yield chunk;
            }

            if !text_buf.is_empty() {
                assistant_blocks.push(AssistantBlock::text(text_buf));
            }

            if !assistant_blocks.is_empty() {
                run_msgs.push(Message::Assistant { blocks: assistant_blocks });
            }
            }

            // Calls go through in batches, in the model's order. With
            // `tool_concurrency` above 1, a run of consecutive calls to
            // `concurrency_safe` tools forms one batch; any other call is
            // a batch of its own. For each batch:
            //
            // 1. gates, one call at a time in the model's order (active
            //    set, `cancel_tool`, `before` hooks, the tool decision),
            //    so approvals, `AntiLoop` and `MaxToolCalls` see the same
            //    sequence they would without concurrency;
            // 2. the calls that passed run together, up to
            //    `tool_concurrency` at once;
            // 3. `after` hooks and `ToolResult` chunks, in the model's
            //    order, once the whole batch is done.
            //
            // With the default of 1 every batch holds one call, which is
            // plain sequential execution. On abort every call is
            // answered: completed batches keep their results, and the
            // calls of the batch in flight and those after it get an
            // error from `abort_step`. A call that ran but has not been
            // through its `after` hooks counts as not run, so no result
            // skips a transform like `Sanitize`.
            //
            // Every call of the step that can still run stays registered
            // (and one the turn did not announce, such as a call of a
            // resumed step, is registered now), so a call still waiting
            // its turn can be cancelled too. A call that cannot run
            // (malformed, or started but never finished) is dropped.
            // Each is unregistered once it has its result, and the
            // `RunGuard` clears what is left when the run ends.
            let runnable: HashSet<String> = tool_calls
                .iter()
                .filter_map(|call| match call {
                    PendingCall::Run { id, .. } | PendingCall::Decided { id, .. } => Some(id.clone()),
                    PendingCall::Rejected { .. }
                    | PendingCall::Answered { .. }
                    | PendingCall::Backgrounded { .. } => None,
                })
                .collect();
            call_tokens.retain(|id, _| {
                let keep = runnable.contains(id);
                if !keep && let Some(control) = &config.control {
                    control.unregister_call(id);
                }
                keep
            });
            for id in runnable {
                call_tokens.entry(id.clone()).or_insert_with(|| {
                    let token = tool_cancellation.child_token();
                    if let Some(control) = &config.control {
                        control.register_call(id, token.clone());
                    }
                    token
                });
            }
            let concurrency = config.tool_concurrency.max(1);
            let batchable = |call: &PendingCall| match call {
                PendingCall::Run { name, .. } => {
                    concurrency > 1 && catalog.get(name).is_some_and(|tool| tool.concurrency_safe())
                }
                _ => false,
            };
            // Deferred calls of the step, by id: the arguments their gate
            // saw and the reason it gave. Once a batch defers a call, the
            // step stops after that batch and the run suspends.
            let mut deferred: HashMap<String, (Value, String)> = HashMap::new();
            let mut tools_result = Vec::new();
            let mut pending = tool_calls.into_iter().peekable();
            while deferred.is_empty() {
                let Some(first) = pending.next() else {
                    break;
                };
                let mut batch = Vec::new();
                if batchable(&first) {
                    while let Some(next) = pending.next_if(|c| batchable(c)) {
                        batch.push(next);
                    }
                }
                batch.insert(0, first);

                // Phase 1: gates, in the model's order.
                let mut slots: Vec<Slot> = Vec::with_capacity(batch.len());
                let mut gating = batch.into_iter();
                while let Some(call) = gating.next() {
                    // Only tools in the run's active set can run. A name the
                    // model was never shown (deferred, or not registered at
                    // all) gets the same "not found" reply either way. The
                    // set is read at dispatch time, not when the model was
                    // prompted.
                    let call = match call {
                        PendingCall::Run { id, name, .. }
                            if !active_snapshot
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .contains(&name) =>
                        {
                            let content = unavailable_tool(&name, &ToolActivation::new(catalog.clone(), active_snapshot.clone()));
                            PendingCall::Rejected { id, content }
                        }
                        // Cancelled through `RunControl::cancel_tool` before
                        // its turn came: like a rejection, no hook fires.
                        PendingCall::Run { id, .. }
                            if call_tokens.get(&id).is_some_and(CancellationToken::is_cancelled) =>
                        {
                            PendingCall::Rejected { id, content: cancelled_before_run() }
                        }
                        call => call,
                    };
                    let (id, name, mut args) = match call {
                        PendingCall::Run { id, name, args } => (id, name, args),
                        // Nothing runs, so no tool hook fires: the synthesized
                        // error only goes out as a ToolResult chunk.
                        PendingCall::Rejected { id, content }
                        | PendingCall::Backgrounded { id, content } => {
                            slots.push(Slot::Answered { id, content, emit: true });
                            continue;
                        }
                        // Answered in the suspended run, which already
                        // emitted its result.
                        PendingCall::Answered { id, content } => {
                            slots.push(Slot::Answered { id, content, emit: false });
                            continue;
                        }
                        // Decided on resume: its gates ran in the suspended
                        // run, so they do not run again.
                        PendingCall::Decided { id, name, args, decision } => {
                            let token = call_tokens
                                .get(&id)
                                .cloned()
                                .unwrap_or_else(|| tool_cancellation.child_token());
                            let call = ToolCallInfo::new(run_id.clone(), step_id.clone(), id.clone(), name.clone());
                            let content = match decision {
                                ResumeDecision::Approve if token.is_cancelled() => Some(cancelled_before_run()),
                                ResumeDecision::Approve => None,
                                ResumeDecision::Deny { reason } => Some(denied_result(&reason)),
                                _ => Some(ToolResultContent::error("unsupported ResumeDecision variant")),
                            };
                            slots.push(Slot::Gated { id, name, args, call, token, content, moved: false });
                            continue;
                        }
                    };
                    let token = call_tokens
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| tool_cancellation.child_token());
                    let call = ToolCallInfo::new(run_id.clone(), step_id.clone(), id.clone(), name.clone());

                    // Input-transform phase: every `_mut` runs before any
                    // gating decision so a sanitizer can rewrite args before
                    // an `ApprovalMiddleware` sees them. Mutated `args` flow
                    // through to the tool invocation below.
                    let mut abort_reason = None;
                    for mw in &config.middlewares {
                        if let Err(reason) = race_abort(
                            mw.on_before_tool_call_mut(&call, &mut args),
                            &mut abort_fut,
                        ).await {
                            abort_reason = Some(reason);
                            break;
                        }
                    }
                    let decision = match abort_reason {
                        Some(reason) => Err(reason),
                        None => race_abort(
                            run_tool_chain(&config.middlewares, &call, &args),
                            &mut abort_fut,
                        ).await,
                    };
                    let content = match decision {
                        // Cancelled while its `before` hooks ran: the tool is
                        // not invoked, but the `after` hooks still pair up.
                        Ok(ToolDecision::Continue) if token.is_cancelled() => Some(cancelled_before_run()),
                        Ok(ToolDecision::Continue) => None,
                        Ok(ToolDecision::Skip {reason}) => {
                            Some(ToolResultContent::error(format!("Tool skipped: {reason}")))
                        },
                        Ok(ToolDecision::Terminate {reason}) => {
                            // No call of the batch runs.
                            let reason = AbortReason::tool_terminated(name.clone(), id.clone(), reason);
                            let current = PendingCall::Run { id, name, args };
                            let unanswered: Vec<PendingCall> = slots.into_iter().map(Slot::into_unanswered)
                                .chain(std::iter::once(current)).chain(gating).chain(pending).collect();
                            let chunks = abort_step(
                                &guard, &step_id, reason, usage_run + delegated.total(),
                                &mut run_msgs, tools_result, unanswered,
                            ).await;
                            for chunk in chunks { yield chunk; }
                            return;
                        }
                        Ok(ToolDecision::Defer {reason}) => {
                            deferred.insert(id.clone(), (args.clone(), reason));
                            slots.push(Slot::Deferred { id, name, args });
                            continue;
                        }
                        Ok(_) => Some(ToolResultContent::error("unsupported ToolDecision variant")),
                        Err(reason) => {
                            let current = PendingCall::Run { id, name, args };
                            let unanswered: Vec<PendingCall> = slots.into_iter().map(Slot::into_unanswered)
                                .chain(std::iter::once(current)).chain(gating).chain(pending).collect();
                            let chunks = abort_step(
                                &guard, &step_id, reason, usage_run + delegated.total(),
                                &mut run_msgs, tools_result, unanswered,
                            ).await;
                            for chunk in chunks { yield chunk; }
                            return;
                        }
                    };
                    slots.push(Slot::Gated { id, name, args, call, token, content, moved: false });
                }

                // Phase 2: run the calls that passed, together. Each
                // call's own token stops just that call: its future is
                // dropped even if the tool ignores the token. A run abort
                // fires those tokens too, but `race_abort` checks the
                // run's abort first.
                //
                // With a `RunControl`, a call to a backgroundable tool
                // runs in a task of its own instead, with a token not
                // tied to the run, so `RunControl::background_tool` can
                // stop waiting for it without stopping it. Until then the
                // engine stands in for the run: dropping or cancelling
                // the call here fires the task's token. Every call the
                // engine lets go of while it was moved lands in `moved_calls`.
                let moved_calls: std::sync::Mutex<HashSet<String>> = std::sync::Mutex::default();
                let moved_calls = &moved_calls;
                let runs = slots.iter().enumerate().filter_map(|(index, slot)| match slot {
                    Slot::Gated { id, name, args, call, token, content: None, .. } => {
                        let progress = ProgressSink::new(id.clone(), progress_tx.clone());
                        let background = match (&config.control, guard.background_run, catalog.get(name)) {
                            (Some(control), Some(background_run), Some(tool)) if tool.backgroundable() => {
                                Some(BackgroundTask {
                                    tool: tool.clone(),
                                    args: args.clone(),
                                    call: call.clone(),
                                    control: control.clone(),
                                    background_run,
                                    middlewares: config.middlewares.clone(),
                                    background: config.background.clone(),
                                })
                            }
                            _ => None,
                        };
                        let ctx = ToolContext::new(
                            run_id.clone(),
                            step_id.clone(),
                            id.clone(),
                            ToolActivation::new(catalog.clone(), active_snapshot.clone()),
                            if background.is_some() { CancellationToken::new() } else { token.clone() },
                        )
                        .with_usage_sink(delegated.clone())
                        .with_progress_sink(progress.clone());
                        if let Some(task) = background {
                            return Some(async move {
                                let _close = CloseOnDrop(progress);
                                if token.is_cancelled() {
                                    return (index, CallOutput::NotStarted);
                                }
                                let control = task.control.clone();
                                let (key, on_moved) = control
                                    .register_background(id.clone(), ctx.cancellation().clone());
                                // Whatever ends this wait (a result, a move,
                                // `cancel_tool`, the run's abort dropping it),
                                // the engine lets go of the call here.
                                let _release = ReleaseOnDrop { control: &control, key, call_id: id, moved: moved_calls };
                                let mut joined = tokio::spawn(task.run(ctx, key));
                                let moved = async {
                                    // An error means the call finished
                                    // (or was let go of) unmoved.
                                    if on_moved.await.is_err() {
                                        std::future::pending::<()>().await;
                                    }
                                };
                                let output = tokio::select! {
                                    biased;
                                    _ = moved => CallOutput::Moved,
                                    _ = token.cancelled() => CallOutput::Cancelled,
                                    joined = &mut joined => match joined {
                                        Ok(Some(content)) => CallOutput::Finished(Ok(content)),
                                        Ok(None) => CallOutput::Moved,
                                        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
                                        Err(_) => CallOutput::Cancelled,
                                    },
                                };
                                (index, output)
                            }.boxed());
                        }
                        Some(async move {
                            // Updates sent once the call is over (it
                            // returned, or its future was dropped) are
                            // stale, even while the batch still runs.
                            let _close = CloseOnDrop(progress);
                            // Cancelled after its gates but before its turn
                            // to run came (a sibling's gate, a full batch):
                            // the tool was never invoked.
                            if token.is_cancelled() {
                                return (index, CallOutput::NotStarted);
                            }
                            let output = tokio::select! {
                                biased;
                                _ = token.cancelled() => CallOutput::Cancelled,
                                result = tools.tool_call_with_ctx(name, args.clone(), &ctx) => CallOutput::Finished(result),
                            };
                            (index, output)
                        }.boxed())
                    }
                    _ => None,
                }).collect::<Vec<_>>();
                // Only this batch's calls may report: progress from a call
                // that already has its result (a task it left running) is
                // stale and dropped, here and below.
                let running: HashSet<String> = slots
                    .iter()
                    .filter_map(|slot| match slot {
                        Slot::Gated { id, content: None, .. } => Some(id.clone()),
                        _ => None,
                    })
                    .collect();
                while progress_rx.try_recv().is_ok() {}
                // Unordered, so a call that finishes frees its slot for
                // the next one even while an earlier call still runs;
                // each output carries its slot index.
                let outputs = {
                    let mut batch = std::pin::pin!(race_abort(
                        futures::stream::iter(runs).buffer_unordered(concurrency).collect::<Vec<_>>(),
                        &mut abort_fut,
                    ));
                    loop {
                        // `yield` stays outside `select!`: the stream macro
                        // does not rewrite it inside another macro.
                        let event = tokio::select! {
                            biased;
                            outputs = &mut batch => BatchEvent::Done(outputs),
                            Some(update) = progress_rx.recv() => BatchEvent::Progress(update),
                        };
                        match event {
                            BatchEvent::Done(outputs) => break outputs,
                            BatchEvent::Progress((call_id, _)) if !running.contains(&call_id) => {}
                            BatchEvent::Progress((call_id, progress)) => {
                                let mut chunk = StreamChunk::tool_progress(
                                    run_id.clone(), step_id.clone(), call_id, progress,
                                );
                                for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                                for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                                yield chunk;
                            }
                        }
                    }
                };
                // What arrived as the batch finished still goes out before
                // the results.
                while let Ok((call_id, progress)) = progress_rx.try_recv() {
                    if !running.contains(&call_id) {
                        continue;
                    }
                    let mut chunk = StreamChunk::tool_progress(
                        run_id.clone(), step_id.clone(), call_id, progress,
                    );
                    for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                    for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                    yield chunk;
                }
                // Every call the engine let go of while it was moved
                // answers with its provisional result, here or in an
                // abort: the tool goes on running either way.
                let moved_calls = std::mem::take(&mut *moved_calls.lock().unwrap_or_else(|e| e.into_inner()));
                for slot in &mut slots {
                    if let Slot::Gated { id, args, call, content, moved, .. } = slot
                        && moved_calls.contains(id)
                    {
                        *moved = true;
                        *content = Some(config.background.notice_for(call, args));
                    }
                }
                let outputs = match outputs {
                    Ok(outputs) => outputs,
                    Err(reason) => {
                        let unanswered: Vec<PendingCall> = slots.into_iter().map(Slot::into_unanswered)
                            .chain(pending).collect();
                        let chunks = abort_step(
                            &guard, &step_id, reason, usage_run + delegated.total(),
                            &mut run_msgs, tools_result, unanswered,
                        ).await;
                        for chunk in chunks { yield chunk; }
                        return;
                    }
                };
                for (index, output) in outputs {
                    let Slot::Gated { name, content, moved: false, .. } = &mut slots[index] else {
                        continue;
                    };
                    *content = Some(match output {
                        // Recorded with the moved calls above.
                        CallOutput::Moved => continue,
                        CallOutput::NotStarted => cancelled_before_run(),
                        CallOutput::Cancelled => cancelled_while_running(),
                        CallOutput::Finished(Ok(content)) => content,
                        // Unreachable while the active set stays a subset
                        // of the catalog; kept in-band all the same.
                        CallOutput::Finished(Err(ToolRegistryError::NotFound(_))) => {
                            unavailable_tool(name, &ToolActivation::new(catalog.clone(), active_snapshot.clone()))
                        },
                        CallOutput::Finished(Err(other)) => bail_with_hooks!(Err(EngineError::Tool(other)), &guard, usage_run + delegated.total(), run_msgs)?,
                    });
                }

                // Phase 3: `after` hooks and results, in the model's order.
                // A deferred call has no result yet: it waits in the
                // suspension, unless the run aborts first.
                let mut held = Vec::new();
                let mut answering = slots.into_iter();
                while let Some(slot) = answering.next() {
                    let (id, content, emit, moved) = match slot {
                        Slot::Answered { id, content, emit } => (id, content, emit, false),
                        slot @ Slot::Deferred { .. } => {
                            held.push(slot.into_unanswered());
                            continue;
                        }
                        Slot::Gated { id, args, call, content, moved, .. } => {
                            let mut content = content.unwrap_or_else(|| {
                                ToolResultContent::error("Tool not run: no result was recorded")
                            });
                            // Output-transform phase: every `_mut` runs before any
                            // observer, so observers and the engine's emitted
                            // `ToolResult` chunk all see the same mutated result.
                            let mut abort_reason = None;
                            for mw in &config.middlewares {
                                if let Err(reason) = race_abort(
                                    mw.on_after_tool_call_mut(&call, &args, &mut content),
                                    &mut abort_fut,
                                ).await {
                                    abort_reason = Some(reason);
                                    break;
                                }
                            }
                            if abort_reason.is_none() {
                                for mw in &config.middlewares {
                                    if let Err(reason) = race_abort(
                                        mw.on_after_tool_call(&call, &args, &content),
                                        &mut abort_fut,
                                    ).await {
                                        abort_reason = Some(reason);
                                        break;
                                    }
                                }
                            }
                            if let Some(reason) = abort_reason {
                                // The call that was in its `after` hooks keeps
                                // its result (whatever the partially-applied
                                // transforms left it as) so history isn't left
                                // with a tool_call missing its tool_result.
                                tools_result.push(UserBlock::tool_result(id, content));
                                let unanswered: Vec<PendingCall> = held.into_iter()
                                    .chain(answering.map(Slot::into_unanswered))
                                    .chain(pending).collect();
                                let chunks = abort_step(
                                    &guard, &step_id, reason, usage_run + delegated.total(),
                                    &mut run_msgs, tools_result, unanswered,
                                ).await;
                                for chunk in chunks { yield chunk; }
                                return;
                            }
                            (id, content, true, moved)
                        }
                    };

                    // Too late to cancel once the result is decided, before
                    // anyone sees it.
                    if let Some(control) = &config.control { control.unregister_call(&id); }
                    if moved {
                        let mut chunk = StreamChunk::tool_backgrounded(run_id.clone(), step_id.clone(), id.clone());
                        for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                        yield chunk;
                    }
                    if emit {
                        let mut chunk = StreamChunk::tool_result(
                            run_id.clone(),
                            step_id.clone(),
                            id.clone(),
                            content.clone(),
                        );
                        for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                        yield chunk;
                    }

                    tools_result.push(UserBlock::tool_result(id, content));
                }
            }

            // A gate deferred a call: the run suspends mid-step. The
            // step's assistant message stays in the run's messages, its
            // results go into the suspension instead (providers need them
            // all in the one user message after it), and the inbox is not
            // drained: the resumed run closes the step.
            if !deferred.is_empty() {
                // The calls after the deferring batch never reached their
                // gates and wait for the resume, except a call the run
                // already rejected (malformed arguments), which a resume
                // would run with the empty arguments that stand in for
                // the ones the model failed to produce, and a call
                // `cancel_tool` stopped, which the resumed run (with
                // tokens of its own) would not know about. Both are
                // answered now.
                for call in pending {
                    let answer = match call {
                        PendingCall::Rejected { id, content } => Some((id, content)),
                        PendingCall::Run { id, .. }
                            if call_tokens.get(&id).is_some_and(CancellationToken::is_cancelled) =>
                        {
                            Some((id, cancelled_before_run()))
                        }
                        _ => None,
                    };
                    if let Some((id, content)) = answer {
                        let mut chunk = StreamChunk::tool_result(
                            run_id.clone(), step_id.clone(), id.clone(), content.clone(),
                        );
                        for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                        yield chunk;
                        tools_result.push(UserBlock::tool_result(id, content));
                    }
                }
                let mut answered: HashMap<String, ToolResultContent> = tools_result
                    .into_iter()
                    .filter_map(|block| match block {
                        UserBlock::ToolResult { call_id, content, .. } => Some((call_id, content)),
                        _ => None,
                    })
                    .collect();
                let calls = step_calls
                    .into_iter()
                    .map(|(id, name, args)| {
                        if let Some((args, reason)) = deferred.remove(&id) {
                            SuspendedCall::new(id, name, args, CallState::Deferred { reason })
                        } else if let Some(content) = answered.remove(&id) {
                            SuspendedCall::new(id, name, args, CallState::Answered { content })
                        } else {
                            SuspendedCall::new(id, name, args, CallState::Waiting)
                        }
                    })
                    .collect();
                // The suspended calls run, if at all, in the resumed run,
                // with tokens of their own.
                guard.clear_calls();
                let suspension = Suspension::new(run_id.clone(), step_id.clone(), calls);
                let new_messages = run_msgs.finish();
                let usage_total = usage_run + delegated.total();

                let mut chunk = StreamChunk::run_suspended(run_id.clone(), suspension);
                for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                yield chunk;

                guard.finished(&FinishReason::Suspended, &usage_total, &new_messages).await;
                let chunk = run_finished_chunk(
                    &config.middlewares,
                    &run_id,
                    FinishReason::Suspended,
                    usage_total,
                    new_messages,
                )
                .await;
                yield chunk;
                return;
            }

            // Messages queued on the run's `RunControl` enter here, once
            // every call of the turn is done: after the tool results, in
            // arrival order, behind the real results of calls this run
            // moved to the background. Aborts never get here, so they
            // stay queued. The run takes background results until it
            // ends (`RunGuard::close_background`), which hands those
            // that came too late to the callback; the guard reports the
            // ones it took once the run is over.
            let drained = guard.drain();
            let mut background_finished = Vec::new();
            let mut inbound = Vec::with_capacity(drained.results.len() + drained.messages.len());
            for (message, result) in drained.results {
                background_finished.push(result);
                inbound.push(message);
            }
            inbound.extend(drained.messages);
            let inbound_ids: Vec<String> = inbound.iter().map(|m| m.id.clone()).collect();
            let inbound_blocks: Vec<UserBlock> =
                inbound.into_iter().flat_map(|m| m.blocks).collect();

            // Completion gate: a turn that ends without tool calls to run
            // would finish the run, unless a middleware asks to continue
            // with an injected user message. Aborts never get here, and
            // neither does a turn that just took inbound blocks: the run
            // goes on to answer them, and the gate runs once the model
            // stops again with nothing queued.
            let mut continue_blocks = None;
            if inbound_blocks.is_empty()
                && !matches!(finish_reason, FinishReason::ToolUse | FinishReason::Aborted(_))
            {
                let decision = match race_abort(
                    run_turn_end_chain(
                        &config.middlewares, &run_id, &step_id, &finish_reason,
                        run_msgs.new_so_far(), &tools_result,
                    ),
                    &mut abort_fut,
                ).await {
                    Ok(d) => d,
                    Err(reason) => {
                        let chunks = abort_step(
                            &guard, &step_id, reason, usage_run + delegated.total(),
                            &mut run_msgs, tools_result, Vec::new(),
                        ).await;
                        for chunk in chunks { yield chunk; }
                        return;
                    }
                };
                if let ContinueDecision::Continue { blocks } = decision {
                    continue_blocks = Some(blocks);
                }
            }
            let continue_run = continue_blocks.is_some() || !inbound_blocks.is_empty();

            // Tool results and injected blocks share one user message
            // rather than two consecutive ones. After an empty turn (no
            // assistant blocks) the injected message follows the previous
            // user message directly; providers merge the two.
            let mut user_blocks = tools_result;
            user_blocks.extend(inbound_blocks);
            user_blocks.extend(continue_blocks.unwrap_or_default());
            if !user_blocks.is_empty() {
                run_msgs.push(Message::User { blocks: user_blocks });
            }
            run_msgs.complete_step();

            for result in background_finished {
                let mut chunk = StreamChunk::background_tool_finished(
                    run_id.clone(),
                    step_id.clone(),
                    result.call_id,
                    result.name,
                    result.content,
                    result.cancelled,
                );
                for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                yield chunk;
            }

            if !inbound_ids.is_empty() {
                let mut chunk = StreamChunk::inbound_delivered(
                    run_id.clone(),
                    step_id.clone(),
                    inbound_ids,
                );
                for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                yield chunk;
            }

            let mut chunk = StreamChunk::step_finished(
                run_id.clone(),
                step_id.clone(),
                iteration,
                Arc::new(run_msgs.new_so_far().to_vec()),
            );
            for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
            for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
            yield chunk;

            if !matches!(finish_reason, FinishReason::ToolUse) && !continue_run {
                break;
            }

            iteration += 1;
        }

        let new_messages = run_msgs.finish();
        let usage_total = usage_run + delegated.total();

        guard.finished(&finish_reason, &usage_total, &new_messages).await;

        let chunk = run_finished_chunk(
            &config.middlewares,
            &run_id,
            finish_reason,
            usage_total,
            new_messages,
        )
        .await;
        yield chunk;
    };

    Box::pin(stream.inspect(move |item| {
        if let Ok(StreamChunk::RunFinished { .. }) = item {
            committed.store(true, Ordering::SeqCst);
        }
    }))
}

/// Closes a call's [`ProgressSink`] when the call's future ends.
struct CloseOnDrop(ProgressSink);

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// Lets go of a backgroundable call when the engine stops waiting for
/// it, recording it in `moved` if it was moved by then.
struct ReleaseOnDrop<'a> {
    control: &'a RunControl,
    key: u64,
    call_id: &'a String,
    moved: &'a std::sync::Mutex<HashSet<String>>,
}

impl Drop for ReleaseOnDrop<'_> {
    fn drop(&mut self) {
        if self.control.release_background(self.key) {
            self.moved
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(self.call_id.clone());
        }
    }
}

/// A call to a backgroundable tool, with what its task needs to deliver
/// the result on its own once it is moved.
struct BackgroundTask {
    tool: Arc<dyn ToolDyn>,
    args: Value,
    call: ToolCallInfo,
    control: RunControl,
    background_run: u64,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
    background: BackgroundConfig,
}

impl BackgroundTask {
    /// Runs the tool, tracked under `key`. Returns its result when the
    /// engine is waiting for it; once the call is moved, delivers the
    /// result itself and returns `None`.
    async fn run(self, ctx: ToolContext, key: u64) -> Option<ToolResultContent> {
        let token = ctx.cancellation().clone();
        let (content, cancelled, panic) = tokio::select! {
            biased;
            _ = token.cancelled() => (cancelled_in_background(), true, None),
            output = std::panic::AssertUnwindSafe(self.tool.call(self.args.clone(), &ctx)).catch_unwind() => {
                match output {
                    Ok(content) => (content, false, None),
                    Err(panic) => (
                        ToolResultContent::error("Tool panicked while running in the background"),
                        false,
                        Some(panic),
                    ),
                }
            }
        };
        match self.control.finish_background(key) {
            BackgroundEnd::Engine => {
                // Not moved: the panic belongs to the run, as with any
                // other tool.
                if let Some(panic) = panic {
                    std::panic::resume_unwind(panic);
                }
                Some(content)
            }
            BackgroundEnd::Moved => {
                self.deliver(content, cancelled).await;
                None
            }
            _ => None,
        }
    }

    /// Hands a moved call's result to its run if the run still takes
    /// it, and to the callback otherwise, after the run's output
    /// transforms.
    async fn deliver(self, mut content: ToolResultContent, cancelled: bool) {
        for mw in &self.middlewares {
            mw.on_after_tool_call_mut(&self.call, &self.args, &mut content)
                .await;
        }
        let ToolCallInfo {
            run_id,
            step_id,
            call_id,
            name,
            ..
        } = self.call;
        let mut result =
            BackgroundToolResult::new(run_id, step_id, call_id, name, self.args, content);
        result.cancelled = cancelled;
        let message = result.to_inbound();
        if let Some(result) =
            self.control
                .offer_background_result(self.background_run, message, result)
        {
            self.background.report(result);
        }
    }
}

/// The real result of a call moved to the background, as stopped by
/// `RunControl::cancel_background`.
fn cancelled_in_background() -> ToolResultContent {
    ToolResultContent::error(
        "Background tool call cancelled by the caller; it may have partly completed",
    )
}

/// How many progress updates a run buffers before dropping new ones.
const PROGRESS_BUFFER: usize = 64;

/// How a call of a batch ended.
enum CallOutput {
    /// `RunControl::cancel_tool` stopped it before the tool was invoked.
    NotStarted,
    /// `RunControl::cancel_tool` stopped it while the tool ran.
    Cancelled,
    /// `RunControl::background_tool` moved it; it runs on without the
    /// run.
    Moved,
    /// The tool returned.
    Finished(Result<ToolResultContent, ToolRegistryError>),
}

/// What a batch in flight produced next: its results, or a progress
/// update from one of its calls.
enum BatchEvent<T> {
    Done(T),
    Progress((String, ToolProgress)),
}

/// One call of a batch on its way through the gate, execution and
/// `after` phases of a step.
enum Slot {
    /// Answered before its gates ran (rejected, or cancelled before its
    /// turn); no tool hook fires for it.
    Answered {
        id: String,
        content: ToolResultContent,
        /// Whether its `ToolResult` chunk still has to go out (not when
        /// the suspended run already emitted it).
        emit: bool,
    },
    /// A gate returned `Defer`; it waits in the run's suspension.
    Deferred {
        id: String,
        name: String,
        args: Value,
    },
    /// Through its `before` hooks. `content` is set up front when the
    /// tool will not be invoked (skipped, or cancelled during its
    /// gates), and after execution otherwise; for a call moved to the
    /// background (`moved`) it is the provisional result.
    Gated {
        id: String,
        name: String,
        args: Value,
        call: ToolCallInfo,
        token: CancellationToken,
        content: Option<ToolResultContent>,
        moved: bool,
    },
}

impl Slot {
    /// The call as `abort_step` takes it: an answered call keeps its
    /// result, a moved one its provisional result, and any other gated
    /// one is answered as not run.
    fn into_unanswered(self) -> PendingCall {
        match self {
            Slot::Answered {
                id,
                content,
                emit: true,
            } => PendingCall::Rejected { id, content },
            Slot::Answered { id, content, .. } => PendingCall::Answered { id, content },
            Slot::Gated {
                id,
                content: Some(content),
                moved: true,
                ..
            } => PendingCall::Backgrounded { id, content },
            Slot::Gated { id, name, args, .. } | Slot::Deferred { id, name, args } => {
                PendingCall::Run { id, name, args }
            }
        }
    }
}

/// A tool call collected from the provider turn, executed after the
/// stream ends.
enum PendingCall {
    Run {
        id: String,
        name: String,
        args: Value,
    },
    /// The tool is never invoked (arguments were not a JSON object, or
    /// the tool is not in the active set) and `content` is the error
    /// reply sent back to the model.
    Rejected {
        id: String,
        content: ToolResultContent,
    },
    /// Resuming: answered in the suspended run. Keeps its result; its
    /// `ToolResult` chunk went out there.
    Answered {
        id: String,
        content: ToolResultContent,
    },
    /// Resuming: deferred in the suspended run, now decided. Skips its
    /// gates, which ran there.
    Decided {
        id: String,
        name: String,
        args: Value,
        decision: ResumeDecision,
    },
    /// Moved to the background before its `after` hooks ran, in a step
    /// the run leaves: answered with its provisional result.
    Backgrounded {
        id: String,
        content: ToolResultContent,
    },
}

/// Longest slice of the raw arguments echoed back to the model.
const MALFORMED_RAW_ECHO_BYTES: usize = 1024;

/// Error reply for a [`StreamChunk::ToolCallMalformed`]: a short
/// explanation plus the `{"INVALID_JSON": raw}` wrapper Anthropic
/// recommends, with `raw` capped so a long truncated payload is not
/// replayed in full.
fn malformed_args_result(name: &str, raw: &str, error: &str) -> ToolResultContent {
    let echoed = if raw.len() > MALFORMED_RAW_ECHO_BYTES {
        let mut end = MALFORMED_RAW_ECHO_BYTES;
        while !raw.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &raw[..end])
    } else {
        raw.to_string()
    };
    let wrapper = serde_json::json!({ "INVALID_JSON": echoed });
    ToolResultContent::error(format!(
        "Invalid JSON arguments for tool '{name}': {error}. The tool was not run; \
         call it again with complete, valid JSON.\n{wrapper}"
    ))
}

/// Error reply for a call to a tool outside the run's active set,
/// listing the tools the model can call instead. Deferred and
/// unregistered names get the same reply, so it does not reveal which
/// hidden tools exist.
fn unavailable_tool(name: &str, tools: &ToolActivation) -> ToolResultContent {
    let available: Vec<String> = tools
        .list_active()
        .into_iter()
        .map(|def| def.name)
        .collect();
    ToolResultContent::error(format!(
        "Tool '{name}' not found. Available tools: [{}]",
        available.join(", ")
    ))
}

async fn run_tool_chain(
    chain: &[Arc<dyn ChatMiddleware>],
    call: &ToolCallInfo,
    args: &Value,
) -> ToolDecision {
    for mw in chain {
        match mw.on_before_tool_call(call, args).await {
            ToolDecision::Continue => continue,
            terminate_or_skip => return terminate_or_skip,
        }
    }
    ToolDecision::Continue
}

/// Asks every middleware's [`ChatMiddleware::on_turn_end`] in
/// registration order; the first non-empty `Continue` wins (an empty
/// one counts as `Stop`, so the next middleware is asked). `new_messages` is the
/// run's messages so far; `pending` are this step's tool results that
/// have not been pushed yet (only non-empty when the model stopped
/// without `ToolUse` after completing tool calls), appended so the hook
/// sees the step as it will be recorded.
async fn run_turn_end_chain(
    chain: &[Arc<dyn ChatMiddleware>],
    run_id: &RunId,
    step_id: &StepId,
    reason: &FinishReason,
    new_messages: &[Message],
    pending: &[UserBlock],
) -> ContinueDecision {
    let owned;
    let new_messages = if pending.is_empty() {
        new_messages
    } else {
        owned = [
            new_messages,
            &[Message::User {
                blocks: pending.to_vec(),
            }],
        ]
        .concat();
        &owned
    };
    for mw in chain {
        match mw
            .on_turn_end(&TurnEndInfo::new(run_id, step_id, reason, new_messages))
            .await
        {
            ContinueDecision::Stop => continue,
            ContinueDecision::Continue { blocks } if !blocks.is_empty() => {
                return ContinueDecision::Continue { blocks };
            }
            _ => continue,
        }
    }
    ContinueDecision::Stop
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::ToolDefinition;
    use ailoop_core::testing::ScriptedModel;
    use ailoop_tools::ToolDyn;
    use serde_json::json;

    struct GetWeather;

    #[async_trait::async_trait]
    impl ToolDyn for GetWeather {
        fn name(&self) -> &str {
            "get_weather"
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "get_weather",
                "stub",
                json!({"type":"object","properties":{},"required":[]}),
                vec![],
            )
        }
        async fn call(&self, _: serde_json::Value, _ctx: &ToolContext) -> ToolResultContent {
            ToolResultContent::text("sunny")
        }
    }

    /// The built-in middlewares keep per-run state keyed by `RunId`.
    /// A caller that drops the stream mid-run fires neither
    /// `on_run_finished` nor `on_run_error`, so without
    /// `on_run_dropped` that state stays in the map forever.
    #[tokio::test]
    async fn builtin_middlewares_release_run_state_when_dropped() {
        use crate::{AntiLoop, ApprovalMiddleware, MaxToolCalls};

        let model = ScriptedModel::new([
            vec![
                StreamChunk::ToolCallFinished {
                    call_id: "toolu_1".into(),
                    name: "get_weather".into(),
                    args: json!({}),
                },
                StreamChunk::TurnFinished {
                    reason: FinishReason::ToolUse,
                    usage: Usage::default(),
                    service_tier: None,
                },
            ],
            vec![StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                usage: Usage::default(),
                service_tier: None,
            }],
        ]);
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(GetWeather)).unwrap();

        let anti_loop = Arc::new(AntiLoop::new());
        let max_calls = Arc::new(MaxToolCalls::new(10));
        let approval = Arc::new(ApprovalMiddleware::approve_all(|_req| async {
            ToolDecision::Continue
        }));
        let mut config = RunConfig::default();
        config.middlewares = vec![anti_loop.clone(), max_calls.clone(), approval.clone()];

        let mut stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
            .await
            .unwrap();
        // Stop right after the tool ran: every middleware holds state.
        while let Some(chunk) = stream.next().await {
            if matches!(chunk.unwrap(), StreamChunk::ToolResult { .. }) {
                break;
            }
        }
        assert_eq!(anti_loop.tracked_runs(), 1);
        assert_eq!(max_calls.tracked_runs(), 1);
        assert_eq!(approval.tracked_runs(), 1);

        drop(stream);
        assert_eq!(anti_loop.tracked_runs(), 0, "AntiLoop kept the dropped run");
        assert_eq!(
            max_calls.tracked_runs(),
            0,
            "MaxToolCalls kept the dropped run"
        );
        assert_eq!(
            approval.tracked_runs(),
            0,
            "ApprovalMiddleware kept the dropped run"
        );
    }

    #[test]
    fn malformed_args_result_wraps_and_caps_raw() {
        let short = malformed_args_result("write_file", r#"{"a":"#, "EOF while parsing");
        assert!(short.is_error);
        let ailoop_core::ToolResultBlock::Text { text } = &short.blocks[0] else {
            panic!("expected a text block");
        };
        assert!(
            text.starts_with("Invalid JSON arguments for tool 'write_file': EOF while parsing.")
        );
        assert!(text.ends_with(r#"{"INVALID_JSON":"{\"a\":"}"#), "{text}");

        // Multi-byte chars straddling the cap must not split.
        let long = "é".repeat(MALFORMED_RAW_ECHO_BYTES);
        let capped = malformed_args_result("t", &long, "EOF");
        let ailoop_core::ToolResultBlock::Text { text } = &capped.blocks[0] else {
            panic!("expected a text block");
        };
        let wrapper: Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
        let echoed = wrapper["INVALID_JSON"].as_str().unwrap();
        assert!(echoed.ends_with('…'));
        assert!(echoed.len() <= MALFORMED_RAW_ECHO_BYTES + '…'.len_utf8());
    }

    /// Engine-level counterpart to the state-machine test in
    /// `ailoop-anthropic`: a thinking turn that ends in a tool call must
    /// land in history as a single assistant message whose blocks are
    /// `[Reasoning{text, signature}, ToolCall{...}]` in that order.
    /// Anthropic rejects requests where the order does not match what was
    /// streamed, so this is load-bearing for tool-use chains.
    #[tokio::test]
    async fn assistant_message_preserves_reasoning_then_tool_call_order() {
        let turn1 = vec![
            StreamChunk::ReasoningDelta {
                delta: "thinking ".into(),
            },
            StreamChunk::ReasoningDelta {
                delta: "step.".into(),
            },
            StreamChunk::ReasoningFinished {
                signature: Some("sig-xyz".into()),
            },
            StreamChunk::ToolCallStarted {
                call_id: "toolu_1".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_1".into(),
                name: "get_weather".into(),
                args: json!({"location": "SF"}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
                service_tier: None,
            },
        ];
        // Turn 2 just ends the run; we only care about the assistant
        // turn that issued the tool call.
        let turn2 = vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        }];

        let model = ScriptedModel::new([turn1, turn2]);

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(GetWeather)).unwrap();

        let stream = run_chat(
            &model,
            vec![Message::user("what's the weather?")],
            &registry,
            RunConfig::default(),
        )
        .await
        .expect("run_chat should start");

        let chunks: Vec<_> = stream.collect().await;

        let new_messages = chunks
            .into_iter()
            .find_map(|c| match c {
                Ok(StreamChunk::RunFinished { new_messages, .. }) => Some(new_messages),
                _ => None,
            })
            .expect("run should emit RunFinished");

        let assistant_blocks = new_messages
            .iter()
            .find_map(|m| match m {
                Message::Assistant { blocks } => Some(blocks),
                _ => None,
            })
            .expect("new_messages should contain the assistant turn");

        assert_eq!(
            assistant_blocks.len(),
            2,
            "expected exactly Reasoning + ToolCall, got {assistant_blocks:?}"
        );
        match &assistant_blocks[0] {
            AssistantBlock::Reasoning { text, signature } => {
                assert_eq!(text, "thinking step.");
                assert_eq!(signature.as_deref(), Some("sig-xyz"));
            }
            other => panic!("expected Reasoning first, got {other:?}"),
        }
        match &assistant_blocks[1] {
            AssistantBlock::ToolCall {
                call_id: id,
                name,
                args,
                ..
            } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(args, &json!({"location": "SF"}));
            }
            other => panic!("expected ToolCall second, got {other:?}"),
        }
    }

    /// All engine-emitted chunks of a run share the same `RunId`. Within
    /// a step, `StepId` matches across `StepStarted`, `ToolResult`, and
    /// `StepFinished`. Distinct iterations get distinct `StepId`s. This
    /// is the contract observability middlewares rely on to correlate
    /// concurrent runs and step-level spans.
    #[tokio::test]
    async fn engine_chunks_share_run_id_and_step_ids_match_per_iteration() {
        let turn1 = vec![
            StreamChunk::ToolCallStarted {
                call_id: "toolu_1".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_1".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
                service_tier: None,
            },
        ];
        let turn2 = vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        }];

        let model = ScriptedModel::new([turn1, turn2]);

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(GetWeather)).unwrap();

        let stream = run_chat(
            &model,
            vec![Message::user("hi")],
            &registry,
            RunConfig::default(),
        )
        .await
        .expect("run_chat should start");

        let chunks: Vec<StreamChunk> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|c| c.unwrap())
            .collect();

        let run_ids: Vec<&RunId> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::RunStarted { run_id, .. }
                | StreamChunk::StepStarted { run_id, .. }
                | StreamChunk::StepFinished { run_id, .. }
                | StreamChunk::ToolResult { run_id, .. }
                | StreamChunk::RunFinished { run_id, .. } => Some(run_id),
                _ => None,
            })
            .collect();
        assert!(!run_ids.is_empty(), "expected engine chunks with RunId");
        let first = run_ids[0];
        for id in &run_ids[1..] {
            assert_eq!(*id, first, "all engine-emitted chunks must share RunId");
        }

        let mut step_ids_by_iter: std::collections::HashMap<usize, Vec<&StepId>> =
            std::collections::HashMap::new();
        for c in &chunks {
            match c {
                StreamChunk::StepStarted {
                    step_id, iteration, ..
                }
                | StreamChunk::StepFinished {
                    step_id, iteration, ..
                } => {
                    step_ids_by_iter
                        .entry(*iteration)
                        .or_default()
                        .push(step_id);
                }
                StreamChunk::ToolResult { step_id, .. } => {
                    step_ids_by_iter.entry(0).or_default().push(step_id);
                }
                _ => {}
            }
        }
        for (iter, ids) in &step_ids_by_iter {
            let s = ids[0];
            for id in &ids[1..] {
                assert_eq!(*id, s, "iteration {iter} step_ids must match");
            }
        }

        let iter_step_ids: Vec<&StepId> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::StepStarted { step_id, .. } => Some(step_id),
                _ => None,
            })
            .collect();
        if iter_step_ids.len() >= 2 {
            assert_ne!(
                iter_step_ids[0], iter_step_ids[1],
                "distinct iterations must mint distinct StepIds"
            );
        }
    }

    /// `HookAction::Terminate` from `on_run_started` must still drive the planned
    /// termination contract: every middleware sees `on_run_finished` once with
    /// `FinishReason::Aborted`. Observers like `TokenBudget` accumulate the
    /// final turn there; if the engine emits `RunFinished` without firing the
    /// hook, those middlewares miss aborted runs entirely.
    #[tokio::test]
    async fn on_run_finished_fires_on_hook_terminate() {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct AbortingMw {
            finished_count: AtomicUsize,
            last_reason: Mutex<Option<FinishReason>>,
        }

        #[async_trait::async_trait]
        impl ChatMiddleware for AbortingMw {
            async fn on_run_started(&self, _run: &RunStartInfo<'_>) -> HookAction {
                HookAction::Terminate {
                    reason: "budget exceeded".into(),
                }
            }
            async fn on_run_finished(&self, run: &RunFinishedInfo<'_>) {
                self.finished_count.fetch_add(1, Ordering::SeqCst);
                *self.last_reason.lock().unwrap() = Some(run.reason.clone());
            }
        }

        let mw = Arc::new(AbortingMw {
            finished_count: AtomicUsize::new(0),
            last_reason: Mutex::new(None),
        });
        let model = ScriptedModel::new(Vec::<Vec<StreamChunk>>::new());
        let registry = ToolRegistry::new();
        let mut config = RunConfig::default();
        config.middlewares = vec![mw.clone()];

        let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
            .await
            .expect("run_chat should start");
        let chunks: Vec<_> = stream.collect().await;

        assert_eq!(
            mw.finished_count.load(Ordering::SeqCst),
            1,
            "on_run_finished must fire exactly once on HookAction::Terminate"
        );
        match mw.last_reason.lock().unwrap().as_ref() {
            Some(FinishReason::Aborted(AbortReason::Terminated { reason, .. })) => {
                assert_eq!(reason, "budget exceeded")
            }
            other => panic!("expected Aborted reason, got {other:?}"),
        }

        let finished = chunks
            .into_iter()
            .find_map(|c| match c {
                Ok(StreamChunk::RunFinished { reason, .. }) => Some(reason),
                _ => None,
            })
            .expect("run should emit RunFinished");
        assert!(
            matches!(
                finished,
                FinishReason::Aborted(AbortReason::Terminated { ref reason, .. })
                    if reason == "budget exceeded"
            ),
            "RunFinished.reason mismatch: {finished:?}"
        );
    }

    /// When middleware aborts mid-step on the second of two tool calls, the
    /// `User { ToolResult }` message for the already-executed first tool must
    /// land in `RunFinished.new_messages`. Otherwise `Conversation::stream`
    /// extends history with an assistant message carrying tool_uses but no
    /// matching tool_results — the next provider call rejects with HTTP 400.
    #[tokio::test]
    async fn tool_terminate_preserves_prior_tool_results_in_history() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct TerminateOnSecondToolMw {
            calls: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ChatMiddleware for TerminateOnSecondToolMw {
            async fn on_before_tool_call(
                &self,
                _call: &ToolCallInfo,
                _args: &Value,
            ) -> ToolDecision {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    ToolDecision::Continue
                } else {
                    ToolDecision::Terminate {
                        reason: "policy".into(),
                    }
                }
            }
        }

        let turn = vec![
            StreamChunk::ToolCallStarted {
                call_id: "toolu_a".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_a".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::ToolCallStarted {
                call_id: "toolu_b".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_b".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
                service_tier: None,
            },
        ];
        let model = ScriptedModel::new([turn]);

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(GetWeather)).unwrap();

        let mw = Arc::new(TerminateOnSecondToolMw {
            calls: AtomicUsize::new(0),
        });
        let mut config = RunConfig::default();
        config.middlewares = vec![mw.clone()];

        let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
            .await
            .expect("run_chat should start");
        let chunks: Vec<_> = stream.collect().await;

        let new_messages = chunks
            .into_iter()
            .find_map(|c| match c {
                Ok(StreamChunk::RunFinished {
                    reason: FinishReason::Aborted(_),
                    new_messages,
                    ..
                }) => Some(new_messages),
                _ => None,
            })
            .expect("run should emit RunFinished{Aborted}");

        let assistant_tool_call_ids: Vec<&str> = new_messages
            .iter()
            .filter_map(|m| match m {
                Message::Assistant { blocks } => Some(blocks),
                _ => None,
            })
            .flat_map(|blocks| blocks.iter())
            .filter_map(|b| match b {
                AssistantBlock::ToolCall { call_id: id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            assistant_tool_call_ids,
            vec!["toolu_a", "toolu_b"],
            "assistant turn must carry both tool_calls"
        );

        let user_tool_result_ids: Vec<&str> = new_messages
            .iter()
            .filter_map(|m| match m {
                Message::User { blocks } => Some(blocks),
                _ => None,
            })
            .flat_map(|blocks| blocks.iter())
            .filter_map(|b| match b {
                UserBlock::ToolResult { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            user_tool_result_ids,
            vec!["toolu_a", "toolu_b"],
            "every tool_call needs its ToolResult in history"
        );
    }

    /// `Terminate` on the first of two calls leaves neither call without
    /// a `tool_result`: both get a synthesized error, in the model's
    /// order, and each goes out as a `ToolResult` chunk.
    #[tokio::test]
    async fn tool_terminate_on_first_call_answers_every_call() {
        struct TerminateMw;

        #[async_trait::async_trait]
        impl ChatMiddleware for TerminateMw {
            async fn on_before_tool_call(&self, _: &ToolCallInfo, _: &Value) -> ToolDecision {
                ToolDecision::Terminate {
                    reason: "policy".into(),
                }
            }
        }

        let turn = vec![
            StreamChunk::ToolCallFinished {
                call_id: "toolu_a".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_b".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
                service_tier: None,
            },
        ];
        let model = ScriptedModel::new([turn]);
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(GetWeather)).unwrap();
        let mut config = RunConfig::default();
        config.middlewares = vec![Arc::new(TerminateMw)];

        let chunks: Vec<_> = run_chat(&model, vec![Message::user("hi")], &registry, config)
            .await
            .expect("run_chat should start")
            .map(|c| c.expect("aborts are not errors"))
            .collect()
            .await;

        let chunk_ids: Vec<&str> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::ToolResult { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(chunk_ids, vec!["toolu_a", "toolu_b"]);

        let Some(StreamChunk::RunFinished {
            reason: FinishReason::Aborted(AbortReason::ToolTerminated { .. }),
            new_messages,
            ..
        }) = chunks.last()
        else {
            panic!(
                "expected RunFinished{{ToolTerminated}} last, got {:?}",
                chunks.last()
            );
        };
        let Some(Message::User { blocks }) = new_messages.last() else {
            panic!("expected a user message with the results, got {new_messages:?}");
        };
        let results: Vec<(&str, &ToolResultContent)> = blocks
            .iter()
            .filter_map(|b| match b {
                UserBlock::ToolResult {
                    call_id, content, ..
                } => Some((call_id.as_str(), content)),
                _ => None,
            })
            .collect();
        assert_eq!(
            results.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec!["toolu_a", "toolu_b"]
        );
        for (_, content) in results {
            assert!(content.is_error);
            assert_eq!(
                content.blocks,
                vec![ailoop_core::ToolResultBlock::Text {
                    text: "Tool not run: the run was aborted (policy)".into()
                }]
            );
        }
    }

    fn tokens(input: u64, output: u64) -> Usage {
        let mut u = Usage::default();
        u.input_tokens = input;
        u.output_tokens = output;
        u
    }

    fn weather_call_turn(usage: Usage) -> Vec<StreamChunk> {
        vec![
            StreamChunk::ToolCallStarted {
                call_id: "toolu_1".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_1".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage,
                service_tier: None,
            },
        ]
    }

    fn end_turn(usage: Usage) -> Vec<StreamChunk> {
        vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage,
            service_tier: None,
        }]
    }

    /// Stand-in for a tool that calls an LLM itself: reports `usage`,
    /// then takes `delay` to answer.
    struct ReportingWeather {
        usage: Usage,
        delay: std::time::Duration,
    }

    #[async_trait::async_trait]
    impl ToolDyn for ReportingWeather {
        fn name(&self) -> &str {
            "get_weather"
        }
        fn tool_definition(&self) -> ToolDefinition {
            GetWeather.tool_definition()
        }
        async fn call(&self, _: serde_json::Value, ctx: &ToolContext) -> ToolResultContent {
            ctx.report_usage(self.usage);
            tokio::time::sleep(self.delay).await;
            ToolResultContent::text("sunny")
        }
    }

    async fn run_finished(
        model: &ScriptedModel,
        tool: Arc<dyn ToolDyn>,
        config: RunConfig,
    ) -> (FinishReason, Usage) {
        let mut registry = ToolRegistry::new();
        registry.register(tool).unwrap();
        let stream = run_chat(model, vec![Message::user("hi")], &registry, config)
            .await
            .expect("run_chat should start");
        stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .find_map(|c| match c {
                Ok(StreamChunk::RunFinished { reason, usage, .. }) => Some((reason, usage)),
                _ => None,
            })
            .expect("run should emit RunFinished")
    }

    /// A tool that reports nothing leaves `RunFinished.usage` as the
    /// plain sum of the run's own turns.
    #[tokio::test]
    async fn run_usage_is_sum_of_turns_when_no_tool_reports() {
        let model = ScriptedModel::new([weather_call_turn(tokens(10, 1)), end_turn(tokens(20, 2))]);
        let (reason, usage) =
            run_finished(&model, Arc::new(GetWeather), RunConfig::default()).await;
        assert!(matches!(reason, FinishReason::EndTurn));
        assert_eq!((usage.input_tokens, usage.output_tokens), (30, 3));
    }

    /// Usage a tool reports through `ToolContext::report_usage` is
    /// added to the run total.
    #[tokio::test]
    async fn tool_reported_usage_is_added_to_run_usage() {
        let model = ScriptedModel::new([weather_call_turn(tokens(10, 1)), end_turn(tokens(20, 2))]);
        let tool = ReportingWeather {
            usage: tokens(100, 40),
            delay: std::time::Duration::ZERO,
        };
        let (_, usage) = run_finished(&model, Arc::new(tool), RunConfig::default()).await;
        assert_eq!((usage.input_tokens, usage.output_tokens), (130, 43));
    }

    /// A report made before the run times out and drops the tool
    /// future still counts in the aborted run's total.
    #[tokio::test]
    async fn tool_reported_usage_survives_timeout_abort() {
        let model = ScriptedModel::new([weather_call_turn(tokens(10, 1)), end_turn(tokens(20, 2))]);
        let tool = ReportingWeather {
            usage: tokens(100, 40),
            delay: std::time::Duration::from_secs(60),
        };
        let mut config = RunConfig::default();
        config.timeout = Some(std::time::Duration::from_millis(50));
        let (reason, usage) = run_finished(&model, Arc::new(tool), config).await;
        assert!(
            matches!(reason, FinishReason::Aborted(AbortReason::Timeout(_))),
            "expected timeout abort, got {reason:?}"
        );
        assert_eq!((usage.input_tokens, usage.output_tokens), (110, 41));
    }

    /// Keeps the `ToolContext::cancellation` token of its last call,
    /// signals that it started, then takes `delay` to answer.
    struct TokenKeeper {
        token: Arc<std::sync::Mutex<Option<CancellationToken>>>,
        started: Arc<tokio::sync::Notify>,
        delay: std::time::Duration,
    }

    impl TokenKeeper {
        fn new(delay: std::time::Duration) -> Self {
            Self {
                token: Arc::default(),
                started: Arc::default(),
                delay,
            }
        }

        fn token(&self) -> CancellationToken {
            self.token
                .lock()
                .unwrap()
                .clone()
                .expect("the tool should have run")
        }
    }

    #[async_trait::async_trait]
    impl ToolDyn for TokenKeeper {
        fn name(&self) -> &str {
            "get_weather"
        }
        fn tool_definition(&self) -> ToolDefinition {
            GetWeather.tool_definition()
        }
        async fn call(&self, _: serde_json::Value, ctx: &ToolContext) -> ToolResultContent {
            *self.token.lock().unwrap() = Some(ctx.cancellation().clone());
            self.started.notify_one();
            tokio::time::sleep(self.delay).await;
            ToolResultContent::text("sunny")
        }
    }

    /// A run timeout cancels the token tools see, so work a tool
    /// spawned (which outlives the dropped tool future) stops too. The
    /// caller's own token is left alone. Paused time: the timeout
    /// elapses only once the tool is waiting, however slow the machine.
    #[tokio::test(start_paused = true)]
    async fn run_timeout_cancels_tool_token() {
        let model = ScriptedModel::new([weather_call_turn(tokens(1, 1)), end_turn(tokens(1, 1))]);
        let tool = Arc::new(TokenKeeper::new(std::time::Duration::from_secs(60)));
        let caller = CancellationToken::new();
        let mut config = RunConfig::default();
        config.timeout = Some(std::time::Duration::from_millis(50));
        config.cancellation = Some(caller.clone());
        let (reason, _) = run_finished(&model, tool.clone(), config).await;
        assert!(
            matches!(reason, FinishReason::Aborted(AbortReason::Timeout(_))),
            "expected timeout abort, got {reason:?}"
        );
        assert!(tool.token().is_cancelled(), "tools must see the timeout");
        assert!(!caller.is_cancelled(), "the caller's token is not touched");
    }

    /// Cancelling the caller's token still reaches the tools.
    #[tokio::test]
    async fn caller_cancellation_reaches_tool_token() {
        let model = ScriptedModel::new([weather_call_turn(tokens(1, 1)), end_turn(tokens(1, 1))]);
        let tool = Arc::new(TokenKeeper::new(std::time::Duration::from_secs(60)));
        let caller = CancellationToken::new();
        let mut config = RunConfig::default();
        config.cancellation = Some(caller.clone());
        let (canceller, started) = (caller.clone(), tool.started.clone());
        tokio::spawn(async move {
            started.notified().await;
            canceller.cancel();
        });
        let (reason, _) = run_finished(&model, tool.clone(), config).await;
        assert!(
            matches!(reason, FinishReason::Aborted(AbortReason::Cancelled)),
            "expected cancel abort, got {reason:?}"
        );
        assert!(tool.token().is_cancelled());
    }

    /// A tool's `Terminate` ends the run, and work an earlier call
    /// spawned is told so.
    #[tokio::test]
    async fn tool_terminate_cancels_tool_token() {
        struct TerminateSecondCall(AtomicUsize);

        #[async_trait::async_trait]
        impl ChatMiddleware for TerminateSecondCall {
            async fn on_before_tool_call(&self, _: &ToolCallInfo, _: &Value) -> ToolDecision {
                if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    ToolDecision::Continue
                } else {
                    ToolDecision::Terminate {
                        reason: "policy".into(),
                    }
                }
            }
        }

        let model = ScriptedModel::new([
            weather_call_turn(tokens(1, 1)),
            weather_call_turn(tokens(1, 1)),
        ]);
        let tool = Arc::new(TokenKeeper::new(std::time::Duration::ZERO));
        let mut config = RunConfig::default();
        config.middlewares = vec![Arc::new(TerminateSecondCall(AtomicUsize::new(0)))];
        let (reason, _) = run_finished(&model, tool.clone(), config).await;
        assert!(
            matches!(
                reason,
                FinishReason::Aborted(AbortReason::ToolTerminated { .. })
            ),
            "expected tool terminate, got {reason:?}"
        );
        assert!(tool.token().is_cancelled());
    }

    /// A run that finishes normally leaves the token alone.
    #[tokio::test]
    async fn finished_run_leaves_tool_token_alone() {
        let model = ScriptedModel::new([weather_call_turn(tokens(1, 1)), end_turn(tokens(1, 1))]);
        let tool = Arc::new(TokenKeeper::new(std::time::Duration::ZERO));
        let (reason, _) = run_finished(&model, tool.clone(), RunConfig::default()).await;
        assert!(matches!(reason, FinishReason::EndTurn));
        assert!(!tool.token().is_cancelled());
    }

    /// Dropping the stream while a tool runs drops the tool future;
    /// the token tells whatever it spawned.
    #[tokio::test]
    async fn dropping_the_stream_cancels_tool_token() {
        let model = ScriptedModel::new([weather_call_turn(tokens(1, 1)), end_turn(tokens(1, 1))]);
        let tool = Arc::new(TokenKeeper::new(std::time::Duration::from_secs(60)));
        let mut registry = ToolRegistry::new();
        registry.register(tool.clone()).unwrap();
        let mut stream = run_chat(
            &model,
            vec![Message::user("hi")],
            &registry,
            RunConfig::default(),
        )
        .await
        .expect("run_chat should start");
        // Poll until the tool has started, then drop the stream.
        loop {
            tokio::select! {
                biased;
                _ = tool.started.notified() => break,
                chunk = stream.next() => {
                    chunk.expect("the tool starts before the run ends").expect("no error");
                }
            }
        }
        drop(stream);
        assert!(tool.token().is_cancelled());
    }
}
