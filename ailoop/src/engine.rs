use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::errors::{EngineError, RunError};
use ailoop_core::{
    AbortReason, AssistantBlock, CancellationToken, ChatMiddleware, ChatRequest, CompletionModel,
    ContinueDecision, FinishReason, HookAction, Message, RunConfig, RunId, StepId, StreamChunk,
    ToolCallInfo, ToolDecision, ToolResultContent, Usage, UserBlock,
};
use ailoop_history::{CompactionError, CompactionReport, History};
use ailoop_tools::{
    ToolActivation, ToolContext, ToolRegistry, UsageSink, errors::ToolRegistryError,
};
use async_stream::try_stream;
use futures::{StreamExt, stream::BoxStream};
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
}

impl RunGuard {
    fn new(middlewares: Vec<Arc<dyn ChatMiddleware>>, run_id: RunId) -> Self {
        Self {
            middlewares,
            run_id,
            closed: AtomicUsize::new(0),
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
        for mw in self.pending() {
            mw.on_run_finished(&self.run_id, reason, usage, new_messages)
                .await;
        }
    }

    async fn failed(
        &self,
        err: &(dyn std::error::Error + Send + Sync),
        usage: &Usage,
        partial_messages: &[Message],
    ) {
        for mw in self.pending() {
            mw.on_run_error(&self.run_id, err, usage, partial_messages)
                .await;
        }
    }
}

impl Drop for RunGuard {
    fn drop(&mut self) {
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
    let mut chunk = StreamChunk::RunFinished {
        run_id: run_id.clone(),
        reason,
        usage,
        new_messages,
    };
    for mw in middlewares {
        mw.on_chunk_mut(&mut chunk).await;
    }
    if !matches!(chunk, StreamChunk::RunFinished { .. }) {
        let (reason, usage, new_messages) = original;
        chunk = StreamChunk::RunFinished {
            run_id: run_id.clone(),
            reason,
            usage,
            new_messages,
        };
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
    let mut chunks = Vec::new();
    for call in unanswered {
        let (id, content) = match call {
            PendingCall::Run { id, .. } => (id, not_run_result(&reason)),
            PendingCall::Rejected { id, content } => (id, content),
        };
        let mut chunk = StreamChunk::ToolResult {
            run_id: run_id.clone(),
            step_id: step_id.clone(),
            call_id: id.clone(),
            content: content.clone(),
        };
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
        run_msgs.push(Message::User {
            blocks: tools_result,
        });
    }
    let new_messages = run_msgs.finish();
    chunks.push(fire_abort_hooks(guard, reason, usage, new_messages).await);
    chunks
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
/// run, with rollback: until [`Self::commit`] runs, dropping this value
/// restores the messages and pin mask captured at construction. That
/// keeps the pre-existing contract that a run ending in `Err` (or a
/// stream dropped mid-run) leaves the history exactly as it was when
/// the run started, even when the run compacted it along the way. The
/// steps completed before an `Err` travel in [`RunError`] instead.
struct ManagedHistory<'a> {
    history: &'a mut History,
    rollback: Option<(Vec<Message>, Vec<bool>)>,
    options: ContextOptions,
}

impl ManagedHistory<'_> {
    fn commit(&mut self) {
        self.rollback = None;
    }
}

impl Drop for ManagedHistory<'_> {
    fn drop(&mut self) {
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
        if let RunContext::Managed(managed) = &mut self.context {
            managed.commit();
        }
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
) -> Result<Option<(CompactionReport, bool)>, CompactionError> {
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
    report: CompactionReport,
) -> StreamChunk {
    let mut chunk = StreamChunk::HistoryCompacted {
        run_id: run_id.clone(),
        before_count: report.before,
        after_count: report.after,
        strategy: report.strategy,
    };
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
/// [`Conversation::run`]: crate::Conversation::run
/// [`Conversation::stream`]: crate::Conversation::stream
/// [`ConversationBuilder::compact_between_iterations`]: crate::ConversationBuilder::compact_between_iterations
/// [`ConversationBuilder::recover_from_context_overflow`]: crate::ConversationBuilder::recover_from_context_overflow
pub async fn run_chat<'a, M: CompletionModel + Sync + Send>(
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
pub(crate) fn run_with_history<'a, M: CompletionModel + Sync + Send>(
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

fn run_engine<'a, M: CompletionModel + Sync + Send>(
    model: &'a M,
    context: RunContext<'a>,
    tools: &'a ToolRegistry,
    config: RunConfig,
    is_overflow: fn(&M::Error) -> bool,
    usage_parent: Option<UsageSink>,
) -> EngineStream<'a, M::Error> {
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
    // Mirror `RunConfig.cancellation` into a token we hand to every
    // per-dispatch `ToolContext`. When the caller did not supply one,
    // a fresh never-cancelled token is the right neutral value — tools
    // that `select!` on `ctx.cancellation().cancelled()` simply pend
    // forever. Built once outside the loop and cloned per dispatch so
    // every tool sees the same handle.
    let tool_cancellation = config.cancellation.clone().unwrap_or_default();
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
        let guard = RunGuard::new(config.middlewares.clone(), run_id.clone());

        for mw in &config.middlewares {
            let action = match race_abort(
                mw.on_run_started(&run_id, run_msgs.context(), &config),
                &mut abort_fut,
            ).await {
                Ok(a) => a,
                Err(reason) => {
                    let chunk = fire_abort_hooks(
                        &guard, reason, Usage::default(), vec![],
                    ).await;
                    yield chunk;
                    return;
                }
            };
            match action {
                HookAction::Continue => {},
                HookAction::Terminate {reason} => {
                    let chunk = fire_abort_hooks(
                        &guard, AbortReason::Terminated { reason }, Usage::default(), vec![],
                    ).await;
                    yield chunk;
                    return;
                }
                _ => {}
            };
        }

        let mut chunk = StreamChunk::RunStarted { run_id: run_id.clone() };
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
                // looping, so the run's messages have no dangling tool_use.
                let new_messages = run_msgs.finish();
                let chunk = fire_abort_hooks(
                    &guard, AbortReason::MaxIterations(config.max_iterations), usage_run + delegated.total(), new_messages,
                ).await;
                yield chunk;
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
            let mut chunk = StreamChunk::StepStarted { run_id: run_id.clone(), step_id: step_id.clone(), iteration };
            for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
            for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
            yield chunk;

            let mut assistant_blocks = Vec::new();
            let mut text_buf = String::new();
            let mut reasoning_buf = String::new();

            let mut tool_calls = Vec::new();

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
                    let active = active_snapshot.lock().expect("active_snapshot lock");
                    catalog
                        .iter()
                        .filter(|(name, _)| active.contains(*name))
                        .map(|(_, tool)| tool.tool_definition())
                        .collect()
                });
                req.system_prompt = config.system_prompt.clone();

                let mut aborted = None;
                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_chat_request(&run_id, &step_id, &mut req),
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
                    StreamChunk::ToolCallFinished { id, name, args } => {
                        assistant_blocks.push(AssistantBlock::tool_call(id.clone(), name.clone(), args.clone()));
                        tool_calls.push(PendingCall::Run { id: id.clone(), name: name.clone(), args: args.clone() })
                    },
                    StreamChunk::ToolCallMalformed { id, name, raw, error } => {
                        // Providers require an object input on replay; the
                        // raw text travels back in the error result instead.
                        assistant_blocks.push(AssistantBlock::tool_call(
                            id.clone(), name.clone(), Value::Object(Default::default()),
                        ));
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

            // Calls run one at a time in the model's order. That is an
            // implementation detail, not part of the contract: the
            // `ChatMiddleware` docs leave the order between calls of a
            // step open so they can run concurrently later. What must
            // hold either way: per-call hook order, `tools_result` in
            // the model's order, and on abort every call answered:
            // completed calls keep their result, the rest get an error
            // from `abort_step`.
            let mut tools_result = Vec::new();
            let mut pending = tool_calls.into_iter();
            while let Some(call) = pending.next() {
                // Only tools in the run's active set can run. A name the
                // model was never shown (deferred, or not registered at
                // all) gets the same "not found" reply either way. The
                // set is read at dispatch time, not when the model was
                // prompted.
                let call = match call {
                    PendingCall::Run { id, name, .. }
                        if !active_snapshot.lock().expect("active_snapshot lock").contains(&name) =>
                    {
                        let content = unavailable_tool(&name, &ToolActivation::new(catalog.clone(), active_snapshot.clone()));
                        PendingCall::Rejected { id, content }
                    }
                    call => call,
                };
                let (id, name, mut args) = match call {
                    PendingCall::Run { id, name, args } => (id, name, args),
                    // Nothing runs, so no tool hook fires: the synthesized
                    // error only goes out as a ToolResult chunk.
                    PendingCall::Rejected { id, content } => {
                        let mut chunk = StreamChunk::ToolResult {
                            run_id: run_id.clone(),
                            step_id: step_id.clone(),
                            call_id: id.clone(),
                            content: content.clone(),
                        };
                        for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                        yield chunk;
                        tools_result.push(UserBlock::tool_result(id, content));
                        continue;
                    }
                };
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
                if let Some(abort_reason) = abort_reason {
                    let current = PendingCall::Run { id, name, args };
                    let chunks = abort_step(
                        &guard, &step_id, abort_reason, usage_run + delegated.total(),
                        &mut run_msgs, tools_result, std::iter::once(current).chain(pending),
                    ).await;
                    for chunk in chunks { yield chunk; }
                    return;
                }

                let decision = match race_abort(
                    run_tool_chain(&config.middlewares, &call, &args),
                    &mut abort_fut,
                ).await {
                    Ok(d) => d,
                    Err(reason) => {
                        let current = PendingCall::Run { id, name, args };
                        let chunks = abort_step(
                            &guard, &step_id, reason, usage_run + delegated.total(),
                            &mut run_msgs, tools_result, std::iter::once(current).chain(pending),
                        ).await;
                        for chunk in chunks { yield chunk; }
                        return;
                    }
                };

                let mut content = match decision {
                    ToolDecision::Continue => {
                        let ctx = ToolContext::new(
                            run_id.clone(),
                            step_id.clone(),
                            id.clone(),
                            ToolActivation::new(catalog.clone(), active_snapshot.clone()),
                            tool_cancellation.clone(),
                        )
                        .with_usage_sink(delegated.clone());
                        let call_result = race_abort(
                            tools.tool_call_with_ctx(&name, args.clone(), &ctx),
                            &mut abort_fut,
                        ).await;
                        match call_result {
                            Ok(Ok(content)) => content,
                            // Unreachable while the active set stays a subset
                            // of the catalog; kept in-band all the same.
                            Ok(Err(ToolRegistryError::NotFound(_))) => {
                                unavailable_tool(&name, &ToolActivation::new(catalog.clone(), active_snapshot.clone()))
                            },
                            Ok(Err(other)) => bail_with_hooks!(Err(EngineError::Tool(other)), &guard, usage_run + delegated.total(), run_msgs)?,
                            Err(reason) => {
                                let current = PendingCall::Run { id, name, args };
                                let chunks = abort_step(
                                    &guard, &step_id, reason, usage_run + delegated.total(),
                                    &mut run_msgs, tools_result, std::iter::once(current).chain(pending),
                                ).await;
                                for chunk in chunks { yield chunk; }
                                return;
                            }
                        }
                    },
                    ToolDecision::Skip {reason} => {
                        ToolResultContent::error(format!("Tool skipped: {reason}"))
                    },
                    ToolDecision::Terminate {reason} => {
                        let reason = AbortReason::ToolTerminated { tool_name: name.clone(), reason };
                        let current = PendingCall::Run { id, name, args };
                        let chunks = abort_step(
                            &guard, &step_id, reason, usage_run + delegated.total(),
                            &mut run_msgs, tools_result, std::iter::once(current).chain(pending),
                        ).await;
                        for chunk in chunks { yield chunk; }
                        return;
                    }
                    _ => ToolResultContent::error("unsupported ToolDecision variant"),
                };

                // Output-transform phase: every `_mut` runs before any
                // observer, so observers and the engine's emitted
                // `ToolResult` chunk all see the same mutated result.
                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_after_tool_call_mut(&call, &args, &mut content),
                        &mut abort_fut,
                    ).await {
                        // Same persistence discipline as the observer
                        // path below: the just-completed tool's result
                        // (whatever the partially-applied transforms
                        // left it as) must land in history so the next
                        // assistant turn isn't missing a tool_result.
                        tools_result.push(UserBlock::tool_result(id.clone(), content.clone()));
                        let chunks = abort_step(
                            &guard, &step_id, reason, usage_run + delegated.total(),
                            &mut run_msgs, tools_result, pending,
                        ).await;
                        for chunk in chunks { yield chunk; }
                        return;
                    }
                }

                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_after_tool_call(&call, &args, &content),
                        &mut abort_fut,
                    ).await {
                        // Preserve the just-completed tool's result so
                        // history isn't left with a tool_call missing
                        // its tool_result on the next assistant turn.
                        tools_result.push(UserBlock::tool_result(id.clone(), content.clone()));
                        let chunks = abort_step(
                            &guard, &step_id, reason, usage_run + delegated.total(),
                            &mut run_msgs, tools_result, pending,
                        ).await;
                        for chunk in chunks { yield chunk; }
                        return;
                    }
                }

                let mut chunk = StreamChunk::ToolResult {
                    run_id: run_id.clone(),
                    step_id: step_id.clone(),
                    call_id: id.clone(),
                    content: content.clone(),
                };
                for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                yield chunk;

                tools_result.push(UserBlock::tool_result(id, content));
            }

            // Completion gate: a turn that ends without tool calls to run
            // would finish the run, unless a middleware asks to continue
            // with an injected user message. Aborts never get here.
            let mut continue_blocks = None;
            if !matches!(finish_reason, FinishReason::ToolUse | FinishReason::Aborted(_)) {
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
            let continue_run = continue_blocks.is_some();

            // Tool results and injected blocks share one user message
            // rather than two consecutive ones. After an empty turn (no
            // assistant blocks) the injected message follows the previous
            // user message directly; providers merge the two.
            let mut user_blocks = tools_result;
            user_blocks.extend(continue_blocks.unwrap_or_default());
            if !user_blocks.is_empty() {
                run_msgs.push(Message::User { blocks: user_blocks });
            }
            run_msgs.complete_step();

            let mut chunk = StreamChunk::StepFinished {
                run_id: run_id.clone(),
                step_id: step_id.clone(),
                iteration,
                new_messages_so_far: Arc::new(run_msgs.new_so_far().to_vec()),
            };
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

    Box::pin(stream)
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
        match mw.on_turn_end(run_id, step_id, reason, new_messages).await {
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
        fn name(&self) -> String {
            "get_weather".into()
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
                    id: "toolu_1".into(),
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
                id: "toolu_1".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                id: "toolu_1".into(),
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
            AssistantBlock::ToolCall { id, name, args, .. } => {
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
                id: "toolu_1".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                id: "toolu_1".into(),
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
                StreamChunk::RunStarted { run_id }
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
            async fn on_run_started(
                &self,
                _run_id: &RunId,
                _messages: &[Message],
                _config: &RunConfig,
            ) -> HookAction {
                HookAction::Terminate {
                    reason: "budget exceeded".into(),
                }
            }
            async fn on_run_finished(
                &self,
                _run_id: &RunId,
                reason: &FinishReason,
                _usage: &Usage,
                _new_messages: &[Message],
            ) {
                self.finished_count.fetch_add(1, Ordering::SeqCst);
                *self.last_reason.lock().unwrap() = Some(reason.clone());
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
            Some(FinishReason::Aborted(AbortReason::Terminated { reason })) => {
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
                FinishReason::Aborted(AbortReason::Terminated { ref reason })
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
                id: "toolu_a".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                id: "toolu_a".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::ToolCallStarted {
                id: "toolu_b".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                id: "toolu_b".into(),
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
                AssistantBlock::ToolCall { id, .. } => Some(id.as_str()),
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
                id: "toolu_a".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::ToolCallFinished {
                id: "toolu_b".into(),
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

    fn tokens(input: u32, output: u32) -> Usage {
        let mut u = Usage::default();
        u.input_tokens = input;
        u.output_tokens = output;
        u
    }

    fn weather_call_turn(usage: Usage) -> Vec<StreamChunk> {
        vec![
            StreamChunk::ToolCallStarted {
                id: "toolu_1".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallFinished {
                id: "toolu_1".into(),
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
        fn name(&self) -> String {
            "get_weather".into()
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
}
