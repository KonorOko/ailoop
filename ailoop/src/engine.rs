use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::errors::EngineError;
use ailoop_core::{
    AbortReason, AssistantBlock, CancellationToken, ChatMiddleware, ChatRequest, CompletionModel,
    ContinueDecision, FinishReason, HookAction, Message, RunConfig, RunId, StepId, StreamChunk,
    ToolDecision, ToolResultContent, Usage, UserBlock,
};
use ailoop_history::{CompactionError, CompactionReport, History};
use ailoop_tools::{ToolActivation, ToolContext, ToolRegistry, errors::ToolRegistryError};
use async_stream::try_stream;
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;

/// Stream of engine chunks for one run.
type EngineStream<'a, E> = BoxStream<'a, Result<StreamChunk, EngineError<E>>>;

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

/// Fires the `on_run_finished` + `on_chunk` hook pair for an aborted
/// run and returns the `RunFinished` chunk for the caller to yield.
/// Centralised so every abort site (hook terminate, tool terminate,
/// timeout, cancellation) follows the same persistence discipline.
async fn fire_abort_hooks(
    middlewares: &[Arc<dyn ChatMiddleware>],
    run_id: &RunId,
    reason: AbortReason,
    usage: Usage,
    new_messages: Vec<Message>,
) -> StreamChunk {
    let finish_reason = FinishReason::Aborted(reason);
    for mw in middlewares {
        mw.on_run_finished(run_id, &finish_reason, &usage, &new_messages)
            .await;
    }
    let mut chunk = StreamChunk::RunFinished {
        run_id: run_id.clone(),
        reason: finish_reason,
        usage,
        new_messages,
    };
    for mw in middlewares {
        mw.on_chunk_mut(&mut chunk).await;
    }
    for mw in middlewares {
        mw.on_chunk(&chunk).await;
    }
    chunk
}

macro_rules! bail_with_hooks {
    ($result: expr, $chain: expr, $run_id: expr) => {
        match $result {
            Ok(v) => Ok(v),
            Err(e) => {
                let err: EngineError<_> = e.into();
                for mw in $chain {
                    mw.on_run_error($run_id, &err).await;
                }
                Err(err)
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
/// the run started, even when the run compacted it along the way.
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
/// [`Conversation::run`]: crate::Conversation::run
/// [`Conversation::stream`]: crate::Conversation::stream
/// [`ConversationBuilder::compact_between_iterations`]: crate::ConversationBuilder::compact_between_iterations
/// [`ConversationBuilder::recover_from_context_overflow`]: crate::ConversationBuilder::recover_from_context_overflow
pub async fn run_chat<'a, M: CompletionModel + Sync + Send>(
    model: &'a M,
    messages: Vec<Message>,
    tools: &'a ToolRegistry,
    config: RunConfig,
) -> Result<BoxStream<'a, Result<StreamChunk, EngineError<M::Error>>>, EngineError<M::Error>> {
    Ok(run_engine(
        model,
        RunContext::Plain(messages),
        tools,
        config,
        |_| false,
    ))
}

/// [`History`]-backed engine entry used by `Conversation`. The run
/// appends to `history` in place and commits on `RunFinished`; on `Err`
/// or when the stream is dropped mid-run, `history` is rolled back to
/// its state at the call. `is_overflow` classifies model setup errors
/// for [`ContextOptions::recover_from_overflow`].
pub(crate) fn run_with_history<'a, M: CompletionModel + Sync + Send>(
    model: &'a M,
    history: &'a mut History,
    tools: &'a ToolRegistry,
    config: RunConfig,
    options: ContextOptions,
    is_overflow: fn(&M::Error) -> bool,
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
    )
}

fn run_engine<'a, M: CompletionModel + Sync + Send>(
    model: &'a M,
    context: RunContext<'a>,
    tools: &'a ToolRegistry,
    config: RunConfig,
    is_overflow: fn(&M::Error) -> bool,
) -> EngineStream<'a, M::Error> {
    let mut run_msgs = RunMessages {
        context,
        new_messages: Vec::new(),
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
    let stream = try_stream! {
        // The abort future resolves with a textual reason when either
        // the timeout elapses or the cancellation token fires; until
        // then it is `pending`, so wrapping any await with
        // `race_abort(_, &mut abort_fut)` is a no-op on the happy path.
        let mut abort_fut: AbortFuture = build_abort_future(
            config.timeout,
            config.cancellation.clone(),
        );

        for mw in &config.middlewares {
            let action = match race_abort(
                mw.on_run_started(&run_id, run_msgs.context(), &config),
                &mut abort_fut,
            ).await {
                Ok(a) => a,
                Err(reason) => {
                    let chunk = fire_abort_hooks(
                        &config.middlewares, &run_id, reason, Usage::default(), vec![],
                    ).await;
                    yield chunk;
                    return;
                }
            };
            match action {
                HookAction::Continue => {},
                HookAction::Terminate {reason} => {
                    let chunk = fire_abort_hooks(
                        &config.middlewares, &run_id, AbortReason::Terminated { reason }, Usage::default(), vec![],
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
                    &config.middlewares, &run_id, AbortReason::MaxIterations(config.max_iterations), usage_run, new_messages,
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
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
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
                    Err(e) => bail_with_hooks!(Err::<(), _>(EngineError::Context(e)), &config.middlewares, &run_id)?,
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
                        &config.middlewares, &run_id, reason, usage_run, new_messages,
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
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                };

                let recoverable = is_overflow(&error)
                    && run_msgs.managed().is_some_and(|m| m.options.recover_from_overflow);
                if !recoverable {
                    bail_with_hooks!(Err::<(), _>(EngineError::Model(error)), &config.middlewares, &run_id)?;
                    unreachable!();
                }
                if overflow_recovered {
                    bail_with_hooks!(Err::<(), _>(EngineError::ContextOverflow(error)), &config.middlewares, &run_id)?;
                    unreachable!();
                }
                overflow_recovered = true;

                let managed = run_msgs.managed().expect("recoverable implies a managed history");
                let outcome = match race_abort(compact(managed.history, true), &mut abort_fut).await {
                    Ok(outcome) => outcome,
                    Err(reason) => {
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
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
                        bail_with_hooks!(Err::<(), _>(EngineError::ContextOverflow(error)), &config.middlewares, &run_id)?;
                    }
                    Err(e) => bail_with_hooks!(Err::<(), _>(EngineError::Context(e)), &config.middlewares, &run_id)?,
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
                        // calls (start without end) are not.
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::text(text_buf));
                        }
                        if !assistant_blocks.is_empty() {
                            run_msgs.push(Message::Assistant { blocks: assistant_blocks });
                        }
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                };
                let chunk = match next {
                    Some(c) => c,
                    None => break,
                };
                let chunk = bail_with_hooks!(chunk.map_err(EngineError::Model), &config.middlewares, &run_id)?;

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
                        tool_calls.push(PendingCall::Malformed {
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

            let mut tools_result = Vec::new();
            for call in tool_calls {
                let (id, name, mut args) = match call {
                    PendingCall::Run { id, name, args } => (id, name, args),
                    // Nothing runs, so no tool hook fires: the synthesized
                    // error only goes out as a ToolResult chunk.
                    PendingCall::Malformed { id, content } => {
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

                // Input-transform phase: every `_mut` runs before any
                // gating decision so a sanitizer can rewrite args before
                // an `ApprovalMiddleware` sees them. Mutated `args` flow
                // through to the tool invocation below.
                let mut abort_reason = None;
                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_before_tool_call_mut(&run_id, &step_id, &name, &mut args),
                        &mut abort_fut,
                    ).await {
                        abort_reason = Some(reason);
                        break;
                    }
                }
                if let Some(abort_reason) = abort_reason {
                    if !tools_result.is_empty() {
                        run_msgs.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                    }
                    let new_messages = run_msgs.finish();
                    let chunk = fire_abort_hooks(
                        &config.middlewares, &run_id, abort_reason, usage_run, new_messages,
                    ).await;
                    yield chunk;
                    return;
                }

                let decision = match race_abort(
                    run_tool_chain(&config.middlewares, &run_id, &step_id, &name, &args),
                    &mut abort_fut,
                ).await {
                    Ok(d) => d,
                    Err(reason) => {
                        if !tools_result.is_empty() {
                            run_msgs.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                        }
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                };

                let mut content = match decision {
                    ToolDecision::Continue => {
                        let ctx = ToolContext::new(
                            run_id.clone(),
                            step_id.clone(),
                            ToolActivation::new(catalog.clone(), active_snapshot.clone()),
                            tool_cancellation.clone(),
                        );
                        let call_result = race_abort(
                            tools.tool_call_with_ctx(&name, args.clone(), &ctx),
                            &mut abort_fut,
                        ).await;
                        match call_result {
                            Ok(Ok(content)) => content,
                            Ok(Err(ToolRegistryError::NotFound(_))) => {
                                let available_tools: Vec<String> = {
                                    let active = active_snapshot.lock().expect("active_snapshot lock");
                                    catalog
                                        .iter()
                                        .filter(|(n, _)| active.contains(*n))
                                        .map(|(n, _)| n.clone())
                                        .collect()
                                };
                                ToolResultContent::error(format!("Tool '{name}' not found. Available tools: [{}]", available_tools.join(", ")))
                            },
                            Ok(Err(other)) => bail_with_hooks!(Err(EngineError::Tool(other)), &config.middlewares, &run_id)?,
                            Err(reason) => {
                                if !tools_result.is_empty() {
                                    run_msgs.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                                }
                                let new_messages = run_msgs.finish();
                                let chunk = fire_abort_hooks(
                                    &config.middlewares, &run_id, reason, usage_run, new_messages,
                                ).await;
                                yield chunk;
                                return;
                            }
                        }
                    },
                    ToolDecision::Skip {reason} => {
                        ToolResultContent::error(format!("Tool skipped: {reason}"))
                    },
                    ToolDecision::Terminate {reason} => {
                        if !tools_result.is_empty() {
                            run_msgs.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                        }
                        let new_messages = run_msgs.finish();
                        let reason = AbortReason::ToolTerminated { tool_name: name.clone(), reason };
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                    _ => ToolResultContent::error("unsupported ToolDecision variant"),
                };

                // Output-transform phase: every `_mut` runs before any
                // observer, so observers and the engine's emitted
                // `ToolResult` chunk all see the same mutated result.
                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_after_tool_call_mut(&run_id, &step_id, &name, &args, &mut content),
                        &mut abort_fut,
                    ).await {
                        // Same persistence discipline as the observer
                        // path below: the just-completed tool's result
                        // (whatever the partially-applied transforms
                        // left it as) must land in history so the next
                        // assistant turn isn't missing a tool_result.
                        tools_result.push(UserBlock::tool_result(id.clone(), content.clone()));
                        run_msgs.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                }

                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_after_tool_call(&run_id, &step_id, &name, &args, &content),
                        &mut abort_fut,
                    ).await {
                        // Preserve the just-completed tool's result so
                        // history isn't left with a tool_call missing
                        // its tool_result on the next assistant turn.
                        tools_result.push(UserBlock::tool_result(id.clone(), content.clone()));
                        run_msgs.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
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
                        if !tools_result.is_empty() {
                            run_msgs.push(Message::User { blocks: tools_result });
                        }
                        let new_messages = run_msgs.finish();
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
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

        for mw in &config.middlewares {
            mw.on_run_finished(&run_id, &finish_reason, &usage_run, &new_messages).await;
        }

        let mut chunk = StreamChunk::RunFinished {
            run_id: run_id.clone(),
            reason: finish_reason,
            usage: usage_run,
            new_messages,
        };
        for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
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
    /// Arguments were not a JSON object; the tool is never invoked and
    /// `content` is the error reply sent back to the model.
    Malformed {
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

async fn run_tool_chain(
    chain: &[Arc<dyn ChatMiddleware>],
    run_id: &RunId,
    step_id: &StepId,
    name: &str,
    args: &Value,
) -> ToolDecision {
    for mw in chain {
        match mw.on_before_tool_call(run_id, step_id, name, args).await {
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
                _run_id: &RunId,
                _step_id: &StepId,
                _name: &str,
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
        assert!(
            user_tool_result_ids.contains(&"toolu_a"),
            "first tool's ToolResult must be preserved in history, got {user_tool_result_ids:?}"
        );
    }
}
