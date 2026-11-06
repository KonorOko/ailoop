//! In-run context management through the public `Conversation`
//! surface: the pre-run compaction, compaction between iterations
//! (`ConversationBuilder::compact_between_iterations`) and recovery from
//! a provider context-window overflow
//! (`ConversationBuilder::recover_from_context_overflow`).
//!
//! Budgets use the default `CharTokenizer` (`len() / 4`). Overflow is
//! simulated with `ScriptedError("context_overflow")`, which reports
//! `ProviderError::is_context_overflow() == true`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ailoop::{
    AbortReason, AssistantBlock, CancellationToken, ChatMiddleware, ChatRequest, CompactionError,
    CompletionModel, Conversation, EngineError, FinishReason, History, HookAction, Message,
    ResumeDecision, RunError, RunErrorInfo, RunFinishedInfo, RunId, RunOptions, RunStartInfo,
    StepInfo, StreamChunk, SummarizeStrategy, ToolCallInfo, ToolDecision, ToolDefinition,
    ToolResultContent, Usage, UserBlock,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel, ScriptedTurn};
use ailoop_tools::{ToolContext, ToolDyn};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Value, json};
use tokio::sync::Notify;

/// `ScriptedModel` that also records how many messages each request
/// carried, so tests can see what compaction did to the context.
struct RecordingModel {
    inner: ScriptedModel,
    request_sizes: Arc<Mutex<Vec<usize>>>,
}

impl RecordingModel {
    fn new(turns: Vec<ScriptedTurn>) -> (Self, Arc<Mutex<Vec<usize>>>) {
        let request_sizes = Arc::new(Mutex::new(Vec::new()));
        let model = Self {
            inner: ScriptedModel::with_turns(turns),
            request_sizes: request_sizes.clone(),
        };
        (model, request_sizes)
    }
}

#[async_trait]
impl CompletionModel for RecordingModel {
    type Error = ScriptedError;

    fn name(&self) -> &str {
        "recording"
    }

    fn model(&self) -> &str {
        "recording"
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        self.request_sizes.lock().unwrap().push(req.messages.len());
        self.inner.chat_stream(req).await
    }
}

/// Returns a 2 400-character result (600 tokens under `CharTokenizer`).
struct BigTool;

#[async_trait]
impl ToolDyn for BigTool {
    fn name(&self) -> &str {
        "big"
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "big",
            "stub",
            json!({"type":"object","properties":{},"required":[]}),
            vec![],
        )
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        ToolResultContent::text("x".repeat(2_400))
    }
}

#[derive(Default)]
struct Counters {
    chat_requests: AtomicUsize,
    run_errors: AtomicUsize,
}

#[async_trait]
impl ChatMiddleware for Counters {
    async fn on_chat_request(&self, _step: &StepInfo, _: &mut ChatRequest) {
        self.chat_requests.fetch_add(1, Ordering::SeqCst);
    }
    async fn on_run_error(&self, _run: &RunErrorInfo<'_>) {
        self.run_errors.fetch_add(1, Ordering::SeqCst);
    }
}

fn tool_call_turn(id: &str) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::ToolCallStarted {
            call_id: id.into(),
            name: "big".into(),
        }),
        Ok(StreamChunk::ToolCallFinished {
            call_id: id.into(),
            name: "big".into(),
            args: json!({}),
        }),
        Ok(StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
            model: None,
        }),
    ])
}

fn text_turn(text: &str) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::TextDelta { delta: text.into() }),
        Ok(StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
            model: None,
        }),
    ])
}

fn overflow_turn() -> ScriptedTurn {
    Err(ScriptedError("context_overflow: prompt is too long".into()))
}

/// Seeds two earlier user/assistant exchanges of 400 characters each
/// (100 tokens per message, 400 in total).
fn seed_prior_turns<M>(chat: &mut Conversation<M>)
where
    M: CompletionModel,
    M::Error: ailoop::ProviderError,
{
    for i in 0..2 {
        chat.push_message(Message::user(format!("{i}{}", "q".repeat(399))));
        chat.push_message(Message::assistant_text(format!("{i}{}", "a".repeat(399))));
    }
}

/// Every assistant `ToolCall` is answered by a `ToolResult` in the
/// very next message, and every `ToolResult` answers a call in the
/// message right before it.
fn assert_no_orphans(messages: &[Message]) {
    for (i, msg) in messages.iter().enumerate() {
        match msg {
            Message::Assistant { blocks } => {
                for b in blocks {
                    if let AssistantBlock::ToolCall { call_id: id, .. } = b {
                        let answered = matches!(messages.get(i + 1), Some(Message::User { blocks })
                            if blocks.iter().any(|b| matches!(b,
                                UserBlock::ToolResult { call_id, .. } if call_id == id)));
                        assert!(answered, "tool_use {id} at {i} has no tool_result");
                    }
                }
            }
            Message::User { blocks } => {
                for b in blocks {
                    if let UserBlock::ToolResult { call_id, .. } = b {
                        let called = i > 0
                            && matches!(&messages[i - 1], Message::Assistant { blocks }
                                if blocks.iter().any(|b| matches!(b,
                                    AssistantBlock::ToolCall { call_id: id, .. } if id == call_id)));
                        assert!(called, "tool_result {call_id} at {i} has no tool_use");
                    }
                }
            }
            _ => {}
        }
    }
}

async fn collect(
    chat: &mut Conversation<RecordingModel>,
    input: &str,
) -> Vec<Result<StreamChunk, RunError<ScriptedError>>> {
    chat.stream(input)
        .await
        .expect("run starts")
        .collect()
        .await
}

fn compacted_counts(
    chunks: &[Result<StreamChunk, RunError<ScriptedError>>],
) -> Vec<(usize, usize)> {
    chunks
        .iter()
        .filter_map(|c| match c {
            Ok(StreamChunk::HistoryCompacted {
                before_count,
                after_count,
                ..
            }) => Some((*before_count, *after_count)),
            _ => None,
        })
        .collect()
}

/// A tool result pushes the context over budget mid-run. With
/// `compact_between_iterations(true)` the engine compacts before the
/// second model call, reports it with the run's `RunId` between the
/// first `StepFinished` and the second `StepStarted`, and the run
/// finishes with a consistent history.
#[tokio::test]
async fn compacts_between_iterations_when_enabled() {
    let (model, sizes) = RecordingModel::new(vec![tool_call_turn("t1"), text_turn("done")]);
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .history(History::builder(1_000).preserve_n_last(1))
        .compact_between_iterations(true)
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let chunks = collect(&mut chat, "go").await;
    let chunks: Vec<StreamChunk> = chunks.into_iter().map(|c| c.expect("no error")).collect();

    let run_id = match &chunks[0] {
        StreamChunk::RunStarted { run_id, .. } => run_id.clone(),
        other => panic!("expected RunStarted first, got {other:?}"),
    };
    assert!(
        matches!(chunks[1], StreamChunk::StepStarted { iteration: 0, .. }),
        "no pre-run compaction expected, got {:?}",
        chunks[1],
    );
    let pos = |pred: &dyn Fn(&StreamChunk) -> bool| chunks.iter().position(pred).unwrap();
    let compacted = pos(&|c| matches!(c, StreamChunk::HistoryCompacted { .. }));
    let first_step_finished = pos(&|c| matches!(c, StreamChunk::StepFinished { iteration: 0, .. }));
    let second_step_started = pos(&|c| matches!(c, StreamChunk::StepStarted { iteration: 1, .. }));
    assert!(first_step_finished < compacted && compacted < second_step_started);
    match &chunks[compacted] {
        StreamChunk::HistoryCompacted {
            run_id: compacted_run_id,
            before_count,
            after_count,
            strategy,
            ..
        } => {
            assert_eq!(compacted_run_id, &run_id);
            // 4 prior + kickoff + tool_use + tool_result → kickoff + pair.
            assert_eq!((*before_count, *after_count), (7, 3));
            assert_eq!(*strategy, "truncate");
        }
        _ => unreachable!(),
    }

    assert_eq!(*sizes.lock().unwrap(), vec![5, 3]);

    let new_messages = match chunks.last() {
        Some(StreamChunk::RunFinished {
            reason: FinishReason::EndTurn,
            new_messages,
            ..
        }) => new_messages,
        other => panic!("expected RunFinished(EndTurn), got {other:?}"),
    };
    assert_eq!(new_messages.len(), 3, "tool_use, tool_result, final text");

    let history = chat.messages();
    assert_eq!(history.len(), 4, "kickoff + pair + final text");
    assert!(matches!(&history[0], Message::User { .. }));
    assert_no_orphans(history);
}

#[tokio::test]
async fn does_not_compact_between_iterations_by_default() {
    let (model, sizes) = RecordingModel::new(vec![tool_call_turn("t1"), text_turn("done")]);
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .history(History::builder(1_000).preserve_n_last(1))
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let chunks = collect(&mut chat, "go").await;
    assert!(compacted_counts(&chunks).is_empty());
    assert_eq!(*sizes.lock().unwrap(), vec![5, 7]);
    assert_eq!(chat.messages().len(), 8);
}

/// The provider rejects the first request as too long; the engine
/// forces a compaction, emits `HistoryCompacted`, reissues the request
/// (running `on_chat_request` again) and the run succeeds.
#[tokio::test]
async fn overflow_compacts_and_retries_once() {
    let (model, sizes) = RecordingModel::new(vec![overflow_turn(), text_turn("fits now")]);
    let counters = Arc::new(Counters::default());
    let mut chat = Conversation::builder(model)
        .middleware(counters.clone())
        .history(History::builder(100_000).preserve_n_last(1))
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let chunks = collect(&mut chat, "go").await;
    assert_eq!(compacted_counts(&chunks), vec![(5, 1)]);
    assert!(matches!(
        chunks.last(),
        Some(Ok(StreamChunk::RunFinished {
            reason: FinishReason::EndTurn,
            ..
        }))
    ));
    assert_eq!(*sizes.lock().unwrap(), vec![5, 1]);
    assert_eq!(counters.chat_requests.load(Ordering::SeqCst), 2);
    assert_eq!(counters.run_errors.load(Ordering::SeqCst), 0);

    let history = chat.messages();
    assert_eq!(history.len(), 2, "kickoff + reply");
    assert!(matches!(&history[1], Message::Assistant { .. }));
}

/// Runs into an overflow whose forced compaction calls a summarizer
/// model scripted to play `summarizer`, and returns the run's error.
async fn failed_summary(summarizer: ScriptedTurn) -> RunError<ScriptedError> {
    let (model, _) = RecordingModel::new(vec![overflow_turn(), text_turn("unused")]);
    let strategy = SummarizeStrategy::new(Arc::new(ScriptedModel::with_turns([summarizer])))
        .with_max_requests(1);
    let mut chat = Conversation::builder(model)
        .history(
            History::builder(100_000)
                .preserve_n_last(1)
                .strategy(Box::new(strategy)),
        )
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);
    chat.run("go").await.expect_err("summary fails")
}

/// A summarizer that, like the main model, rejects what does not fit:
/// any request longer than `limit` characters gets an overflow error.
struct LimitedSummarizer {
    limit: usize,
    requests: AtomicUsize,
}

#[async_trait]
impl CompletionModel for LimitedSummarizer {
    type Error = ScriptedError;

    fn name(&self) -> &str {
        "limited"
    }

    fn model(&self) -> &str {
        "limited"
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let size = serde_json::to_string(&req.messages).unwrap().len();
        if size > self.limit {
            return Err(ScriptedError("context_overflow: prompt is too long".into()));
        }
        ScriptedModel::with_turns([text_turn("compact summary")])
            .chat_stream(req)
            .await
    }
}

/// The main model overflows and so does the summarizer's first request.
/// The strategy splits the prefix, so the recovery still succeeds.
#[tokio::test]
async fn overflow_recovery_survives_a_summarizer_overflow() {
    let (model, sizes) = RecordingModel::new(vec![overflow_turn(), text_turn("fits now")]);
    let summarizer = Arc::new(LimitedSummarizer {
        limit: 1_000,
        requests: AtomicUsize::new(0),
    });
    let mut chat = Conversation::builder(model)
        .history(
            History::builder(100_000)
                .preserve_n_last(1)
                .strategy(Box::new(SummarizeStrategy::new(summarizer.clone()))),
        )
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let chunks = collect(&mut chat, "go").await;

    assert!(matches!(
        chunks.last(),
        Some(Ok(StreamChunk::RunFinished {
            reason: FinishReason::EndTurn,
            ..
        }))
    ));
    assert_eq!(compacted_counts(&chunks), vec![(5, 2)]);
    assert!(summarizer.requests.load(Ordering::SeqCst) > 1);
    assert_eq!(*sizes.lock().unwrap(), vec![5, 2]);

    let history = chat.messages();
    assert_eq!(history.len(), 3, "summary + kickoff + reply");
    assert!(matches!(&history[0], Message::User { blocks }
        if matches!(&blocks[0], UserBlock::Text { text, .. } if text.contains("compact summary"))));
}

/// The summarizer failing when the stream opens, or after some text.
fn failing_summaries(error: &str) -> [ScriptedTurn; 2] {
    [
        Err(ScriptedError(error.into())),
        Ok(vec![
            Ok(StreamChunk::TextDelta {
                delta: "partial".into(),
            }),
            Err(ScriptedError(error.into())),
        ]),
    ]
}

/// A summary too large for the model is told apart through
/// `EngineError::Context`, without comparing text, and the provider's
/// error can be read back from the source chain.
#[tokio::test]
async fn summarizer_overflow_is_typed_through_the_engine_error() {
    for summarizer in failing_summaries("context_overflow: summary too long") {
        let err = failed_summary(summarizer).await;
        let EngineError::Context(compaction) = err.kind() else {
            panic!("expected EngineError::Context, got {err:?}");
        };
        assert!(compaction.is_context_overflow(), "{compaction:?}");

        let CompactionError::SummarizationFailed {
            context_overflow, ..
        } = compaction
        else {
            panic!("expected SummarizationFailed, got {compaction:?}");
        };
        assert!(*context_overflow);

        let source = std::error::Error::source(compaction).expect("source");
        assert_eq!(
            source.downcast_ref::<ScriptedError>(),
            Some(&ScriptedError("context_overflow: summary too long".into())),
        );
    }
}

/// Any other summarizer failure is not reported as an overflow.
#[tokio::test]
async fn summarizer_outage_is_not_an_overflow_through_the_engine_error() {
    for summarizer in failing_summaries("network outage") {
        let err = failed_summary(summarizer).await;
        let EngineError::Context(compaction) = err.kind() else {
            panic!("expected EngineError::Context, got {err:?}");
        };
        assert!(!compaction.is_context_overflow(), "{compaction:?}");
        assert!(
            std::error::Error::source(compaction)
                .and_then(|source| source.downcast_ref::<ScriptedError>())
                .is_some(),
            "{compaction:?}"
        );
    }
}

/// A third-party strategy builds the error from the facade alone.
#[test]
fn compaction_error_is_constructible_from_the_facade() {
    let err = CompactionError::summarization_failed("too big", true, None);
    assert!(err.is_context_overflow());
    assert_eq!(err.to_string(), "Summarization failed: too big");

    let err = CompactionError::summarization_failed_from(ScriptedError("outage".into()));
    assert!(!err.is_context_overflow());
}

/// Overflow on the second model call, after a tool ran: the forced
/// compaction drops earlier turns but keeps this run's tool_use /
/// tool_result pair intact.
#[tokio::test]
async fn overflow_after_tool_call_keeps_pairs_intact() {
    let (model, sizes) = RecordingModel::new(vec![
        tool_call_turn("t1"),
        overflow_turn(),
        text_turn("done"),
    ]);
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .history(History::builder(100_000).preserve_n_last(1))
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let outcome = chat.run("go").await.expect("recovered");
    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(outcome.new_messages.len(), 3);
    assert_eq!(*sizes.lock().unwrap(), vec![5, 7, 3]);

    let history = chat.messages();
    assert_eq!(history.len(), 4);
    assert_no_orphans(history);
}

/// Overflow persists after the one allowed retry: typed error, and the
/// history is rolled back to its state at run start, without the
/// run's half-finished tool turn and without the forced compaction.
#[tokio::test]
async fn persistent_overflow_is_a_typed_error_and_rolls_back() {
    let (model, sizes) =
        RecordingModel::new(vec![tool_call_turn("t1"), overflow_turn(), overflow_turn()]);
    let counters = Arc::new(Counters::default());
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .middleware(counters.clone())
        .history(History::builder(100_000).preserve_n_last(1))
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);
    let before: Vec<String> = chat.messages().iter().map(|m| format!("{m:?}")).collect();

    let err = chat.run("go").await.expect_err("overflow persists");
    match err.into_kind() {
        EngineError::ContextOverflow(ScriptedError(msg)) => {
            assert!(msg.contains("context_overflow"))
        }
        other => panic!("expected ContextOverflow, got {other:?}"),
    }
    assert_eq!(*sizes.lock().unwrap(), vec![5, 7, 3]);
    assert_eq!(counters.run_errors.load(Ordering::SeqCst), 1);

    let history = chat.messages();
    assert_eq!(history.len(), 5, "prior turns + kickoff");
    let after: Vec<String> = history.iter().take(4).map(|m| format!("{m:?}")).collect();
    assert_eq!(after, before, "prior turns restored verbatim");
    assert_no_orphans(history);
}

/// Nothing to drop (the run is the whole history): no pointless retry.
#[tokio::test]
async fn overflow_with_nothing_to_compact_fails_without_retry() {
    let (model, sizes) = RecordingModel::new(vec![overflow_turn(), text_turn("unused")]);
    let mut chat = Conversation::builder(model)
        .history(History::builder(100_000).preserve_n_last(1))
        .build()
        .unwrap();

    let err = chat.run("go").await.expect_err("cannot shrink");
    assert!(
        matches!(err.kind(), EngineError::ContextOverflow(_)),
        "{err:?}"
    );
    assert_eq!(
        err.provider_kind(),
        Some(ailoop::ProviderErrorKind::ContextOverflow)
    );
    assert_eq!(*sizes.lock().unwrap(), vec![1]);
    assert_eq!(chat.messages().len(), 1);
}

#[tokio::test]
async fn overflow_is_a_model_error_when_recovery_is_off() {
    let (model, sizes) = RecordingModel::new(vec![overflow_turn(), text_turn("unused")]);
    let mut chat = Conversation::builder(model)
        .history(History::builder(100_000).preserve_n_last(1))
        .recover_from_context_overflow(false)
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let err = chat.run("go").await.expect_err("no recovery");
    assert!(matches!(err.kind(), EngineError::Model(_)), "{err:?}");
    assert_eq!(
        err.kind().provider_kind(),
        Some(ailoop::ProviderErrorKind::ContextOverflow)
    );
    assert_eq!(*sizes.lock().unwrap(), vec![5]);
    assert_eq!(chat.messages().len(), 5);
}

/// Dropping the stream mid-run discards the run's changes, including a
/// compaction it already performed, as before in-run compaction existed.
#[tokio::test]
async fn dropping_the_stream_mid_run_rolls_back_history() {
    let (model, _) = RecordingModel::new(vec![tool_call_turn("t1"), text_turn("done")]);
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .history(History::builder(1_000).preserve_n_last(1))
        .compact_between_iterations(true)
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    {
        let mut stream = chat.stream("go").await.unwrap();
        while let Some(chunk) = stream.next().await {
            if matches!(chunk.unwrap(), StreamChunk::HistoryCompacted { .. }) {
                break;
            }
        }
    }

    assert_eq!(chat.messages().len(), 5, "prior turns + kickoff");
}

/// Long enough that only a run stuck on the blocked summarizer reaches
/// it; every other wait in these tests is immediate.
const STUCK: Duration = Duration::from_secs(5);

/// Controls a [`GatedSummarizer`] from the test.
#[derive(Clone, Default)]
struct SummarizerGate {
    /// Notified when a summary request starts.
    entered: Arc<Notify>,
    /// Lets a blocked summary request go on.
    release: Arc<Notify>,
    requests: Arc<AtomicUsize>,
    /// Summary requests still in flight, so a test can see an aborted
    /// one was dropped rather than left running.
    in_flight: Arc<AtomicUsize>,
}

impl SummarizerGate {
    fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }
    fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

struct InFlight(Arc<AtomicUsize>);

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Summarizer whose request blocks until [`SummarizerGate::release`] is
/// notified, then answers with `inner`.
struct GatedSummarizer {
    gate: SummarizerGate,
    inner: ScriptedModel,
}

#[async_trait]
impl CompletionModel for GatedSummarizer {
    type Error = ScriptedError;

    fn name(&self) -> &str {
        "gated-summarizer"
    }

    fn model(&self) -> &str {
        "gated-summarizer"
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        self.gate.requests.fetch_add(1, Ordering::SeqCst);
        self.gate.in_flight.fetch_add(1, Ordering::SeqCst);
        let _in_flight = InFlight(self.gate.in_flight.clone());
        self.gate.entered.notify_one();
        self.gate.release.notified().await;
        self.inner.chat_stream(req).await
    }
}

/// Logs the run's hooks and chunks in order.
#[derive(Default)]
struct Lifecycle {
    log: Mutex<Vec<String>>,
    started_with: Mutex<Vec<usize>>,
}

impl Lifecycle {
    fn push(&self, entry: impl Into<String>) {
        self.log.lock().unwrap().push(entry.into());
    }
    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
    fn count(&self, entry: &str) -> usize {
        self.log().iter().filter(|e| *e == entry).count()
    }
    /// Exactly one closing hook fired, and it was `expected`.
    fn assert_closed_once_by(&self, expected: &str) {
        let closing = ["on_run_finished", "on_run_error", "on_run_dropped"];
        let log = self.log();
        let seen: Vec<&String> = log
            .iter()
            .filter(|e| closing.contains(&e.as_str()))
            .collect();
        assert_eq!(seen, [expected], "closing hooks in {log:?}");
    }
}

#[async_trait]
impl ChatMiddleware for Lifecycle {
    async fn on_run_started(&self, run: &RunStartInfo<'_>) -> HookAction {
        self.started_with.lock().unwrap().push(run.messages.len());
        self.push("on_run_started");
        HookAction::Continue
    }
    async fn on_chat_request(&self, _step: &StepInfo, _: &mut ChatRequest) {
        self.push("on_chat_request");
    }
    async fn on_chunk(&self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::RunStarted { .. } => self.push("RunStarted"),
            StreamChunk::HistoryCompacted { .. } => self.push("HistoryCompacted"),
            StreamChunk::StepStarted { .. } => self.push("StepStarted"),
            StreamChunk::RunFinished { .. } => self.push("RunFinished"),
            _ => {}
        }
    }
    async fn on_run_finished(&self, _run: &RunFinishedInfo<'_>) {
        self.push("on_run_finished");
    }
    async fn on_run_error(&self, _run: &RunErrorInfo<'_>) {
        self.push("on_run_error");
    }
    fn on_run_dropped(&self, _run_id: &RunId) {
        self.push("on_run_dropped");
    }
}

fn summary_turn() -> ScriptedTurn {
    text_turn("the summary")
}

/// What [`over_budget_chat`] hands back: the conversation, its
/// summarizer's gate, its lifecycle log and the main model's request
/// sizes.
type OverBudgetChat = (
    Conversation<RecordingModel>,
    SummarizerGate,
    Arc<Lifecycle>,
    Arc<Mutex<Vec<usize>>>,
);

/// A conversation whose next kickoff goes over budget, so the run
/// starts by summarizing the four seeded messages through a
/// [`GatedSummarizer`] answering with `summary`. The turns in `turns`
/// are the main model's.
fn over_budget_chat(turns: Vec<ScriptedTurn>, summary: ScriptedTurn) -> OverBudgetChat {
    let gate = SummarizerGate::default();
    let summarizer = GatedSummarizer {
        gate: gate.clone(),
        inner: ScriptedModel::with_turns([summary]),
    };
    let lifecycle = Arc::new(Lifecycle::default());
    let (model, sizes) = RecordingModel::new(turns);
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .history(
            History::builder(300)
                .preserve_n_last(1)
                .strategy(Box::new(SummarizeStrategy::new(Arc::new(summarizer)))),
        )
        .middleware(lifecycle.clone())
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);
    (chat, gate, lifecycle, sizes)
}

/// The four seeded messages and the kickoff, untouched.
fn assert_nothing_lost(messages: &[Message]) {
    assert_eq!(messages.len(), 5, "prior turns + kickoff: {messages:?}");
    let text = |m: &Message| serde_json::to_string(m).unwrap();
    for (i, repeated) in ["qqqq", "aaaa", "qqqq", "aaaa"].iter().enumerate() {
        assert!(text(&messages[i]).contains(repeated), "message {i} changed");
    }
    assert!(text(&messages[4]).contains("go"), "kickoff stays last");
}

fn assert_compacted(messages: &[Message]) {
    assert_eq!(messages.len(), 2, "summary + kickoff: {messages:?}");
    let summary = serde_json::to_string(&messages[0]).unwrap();
    assert!(summary.contains("the summary"), "{summary}");
    assert!(serde_json::to_string(&messages[1]).unwrap().contains("go"));
}

/// Runs to the end, failing the test if the run gets stuck.
async fn drain(
    chat: &mut Conversation<RecordingModel>,
    options: RunOptions,
) -> Vec<Result<StreamChunk, RunError<ScriptedError>>> {
    tokio::time::timeout(STUCK, async {
        chat.stream_with_options("go", options)
            .await
            .expect("run starts")
            .collect()
            .await
    })
    .await
    .expect("the run is stuck behind the blocked summary")
}

/// The chunks of a run aborted while summarizing: `RunStarted`, then an
/// empty `RunFinished` with the abort reason.
fn aborted_reason(chunks: Vec<Result<StreamChunk, RunError<ScriptedError>>>) -> AbortReason {
    let mut chunks = chunks
        .into_iter()
        .map(|c| c.expect("aborts are not errors"));
    assert!(matches!(
        chunks.next(),
        Some(StreamChunk::RunStarted { .. })
    ));
    let finished = chunks.next();
    assert!(chunks.next().is_none(), "RunFinished is the last chunk");
    match finished {
        Some(StreamChunk::RunFinished {
            reason: FinishReason::Aborted(reason),
            new_messages,
            ..
        }) => {
            assert!(new_messages.is_empty(), "{new_messages:?}");
            reason
        }
        other => panic!("expected an aborted RunFinished, got {other:?}"),
    }
}

/// Cancelling while the summarizer is blocked ends the run at once with
/// `Cancelled` and drops the request. The history keeps every message,
/// kickoff included, as an abort an instant later would.
#[tokio::test]
async fn cancelling_interrupts_the_pre_run_compaction() {
    let (mut chat, gate, lifecycle, sizes) = over_budget_chat(vec![], summary_turn());
    let token = CancellationToken::new();
    tokio::spawn({
        let (token, entered) = (token.clone(), gate.entered.clone());
        async move {
            entered.notified().await;
            token.cancel();
        }
    });

    let chunks = drain(&mut chat, RunOptions::new().cancellation(token)).await;

    assert!(matches!(aborted_reason(chunks), AbortReason::Cancelled));
    assert_eq!(gate.requests(), 1, "the summary had started");
    assert_eq!(gate.in_flight(), 0, "the summary request was dropped");
    lifecycle.assert_closed_once_by("on_run_finished");
    assert!(
        sizes.lock().unwrap().is_empty(),
        "the model was never asked"
    );
    assert_nothing_lost(chat.messages());
}

/// The run's timeout covers the pre-run compaction.
#[tokio::test]
async fn timeout_interrupts_the_pre_run_compaction() {
    let (mut chat, gate, lifecycle, sizes) = over_budget_chat(vec![], summary_turn());
    let limit = Duration::from_millis(20);

    let chunks = drain(&mut chat, RunOptions::new().timeout(limit)).await;

    assert!(matches!(aborted_reason(chunks), AbortReason::Timeout(d) if d == limit));
    assert_eq!(gate.requests(), 1, "the summary had started");
    assert_eq!(gate.in_flight(), 0, "the summary request was dropped");
    lifecycle.assert_closed_once_by("on_run_finished");
    assert!(
        sizes.lock().unwrap().is_empty(),
        "the model was never asked"
    );
    assert_nothing_lost(chat.messages());
}

/// A stream dropped during the summary rolls back like any other
/// dropped stream: the kickoff stays, nothing is lost, and only
/// `on_run_dropped` closes the run.
#[tokio::test]
async fn dropping_the_stream_during_the_pre_run_compaction_rolls_back() {
    let (mut chat, gate, lifecycle, _) = over_budget_chat(vec![], summary_turn());

    tokio::time::timeout(STUCK, async {
        let mut stream = chat.stream("go").await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert!(matches!(first, StreamChunk::RunStarted { .. }));
        tokio::select! {
            _ = gate.entered.notified() => {}
            chunk = stream.next() => panic!("the summary is blocked, got {chunk:?}"),
        }
    })
    .await
    .expect("the run is stuck behind the blocked summary");

    assert_eq!(gate.in_flight(), 0, "the summary request was dropped");
    lifecycle.assert_closed_once_by("on_run_dropped");
    assert_nothing_lost(chat.messages());
}

/// The compaction runs after `on_run_started` and `RunStarted`, before
/// the first step, whatever `compact_between_iterations` says; hooks and
/// chunks agree on the order, and the first request is the compacted one.
#[tokio::test]
async fn pre_run_compaction_comes_after_run_started_and_before_the_first_step() {
    for between_iterations in [false, true] {
        let gate = SummarizerGate::default();
        gate.release.notify_one();
        let summarizer = GatedSummarizer {
            gate: gate.clone(),
            inner: ScriptedModel::with_turns([summary_turn()]),
        };
        let lifecycle = Arc::new(Lifecycle::default());
        let (model, sizes) = RecordingModel::new(vec![text_turn("done")]);
        let mut chat = Conversation::builder(model)
            .history(
                History::builder(300)
                    .preserve_n_last(1)
                    .strategy(Box::new(SummarizeStrategy::new(Arc::new(summarizer)))),
            )
            .compact_between_iterations(between_iterations)
            .middleware(lifecycle.clone())
            .build()
            .unwrap();
        seed_prior_turns(&mut chat);

        let chunks = drain(&mut chat, RunOptions::new()).await;

        let chunks: Vec<StreamChunk> = chunks.into_iter().map(|c| c.unwrap()).collect();
        assert!(matches!(chunks[0], StreamChunk::RunStarted { .. }));
        match &chunks[1] {
            StreamChunk::HistoryCompacted {
                before_count,
                after_count,
                strategy,
                ..
            } => {
                assert_eq!((*before_count, *after_count), (5, 2));
                assert_eq!(*strategy, "summarize");
            }
            other => panic!("expected HistoryCompacted after RunStarted, got {other:?}"),
        }
        assert!(matches!(
            chunks[2],
            StreamChunk::StepStarted { iteration: 0, .. }
        ));
        assert_eq!(
            lifecycle.log(),
            [
                "on_run_started",
                "RunStarted",
                "HistoryCompacted",
                "StepStarted",
                "on_chat_request",
                "on_run_finished",
                "RunFinished",
            ],
        );
        assert_eq!(*lifecycle.started_with.lock().unwrap(), [5]);
        assert_eq!(
            *sizes.lock().unwrap(),
            [2],
            "the first request is compacted"
        );
        assert_eq!(gate.requests(), 1);
    }
}

/// An `Err` in the first step keeps the summary already paid for: the
/// rollback goes back to the compacted history with the kickoff.
#[tokio::test]
async fn model_error_keeps_the_pre_run_compaction() {
    let (mut chat, gate, lifecycle, _) =
        over_budget_chat(vec![Err(ScriptedError("boom".into()))], summary_turn());
    gate.release.notify_one();

    let err = tokio::time::timeout(STUCK, chat.run("go"))
        .await
        .unwrap()
        .expect_err("the model fails");

    assert!(matches!(err.kind(), EngineError::Model(_)), "{err:?}");
    lifecycle.assert_closed_once_by("on_run_error");
    assert_compacted(chat.messages());
}

/// A later step failing rolls back the run's own messages, down to the
/// compacted baseline rather than past it.
#[tokio::test]
async fn model_error_after_a_step_rolls_back_to_the_pre_run_compaction() {
    let (mut chat, gate, _, _) = over_budget_chat(
        vec![tool_call_turn("t1"), Err(ScriptedError("boom".into()))],
        summary_turn(),
    );
    gate.release.notify_one();

    let err = tokio::time::timeout(STUCK, chat.run("go"))
        .await
        .unwrap()
        .expect_err("the second turn fails");

    assert_eq!(err.partial_messages().len(), 2, "tool_use + tool_result");
    assert_compacted(chat.messages());
}

/// A stream dropped after the pre-run compaction keeps it too, while
/// `dropping_the_stream_mid_run_rolls_back_history` covers the mid-run
/// compactions, which are undone.
#[tokio::test]
async fn dropping_the_stream_after_the_pre_run_compaction_keeps_it() {
    let (mut chat, gate, lifecycle, _) = over_budget_chat(vec![text_turn("never")], summary_turn());
    gate.release.notify_one();

    tokio::time::timeout(STUCK, async {
        let mut stream = chat.stream("go").await.unwrap();
        while let Some(chunk) = stream.next().await {
            if matches!(chunk.unwrap(), StreamChunk::HistoryCompacted { .. }) {
                break;
            }
        }
    })
    .await
    .unwrap();

    lifecycle.assert_closed_once_by("on_run_dropped");
    assert_compacted(chat.messages());
}

/// A failed summary arrives through the stream, after `RunStarted`, and
/// fires `on_run_error` once. The history keeps the kickoff, and `run`
/// still returns `Err`.
#[tokio::test]
async fn failed_pre_run_compaction_arrives_through_the_stream() {
    let (mut chat, gate, lifecycle, sizes) = over_budget_chat(
        vec![text_turn("never")],
        Err(ScriptedError("summary refused".into())),
    );
    gate.release.notify_one();

    let chunks = drain(&mut chat, RunOptions::new()).await;

    assert_eq!(chunks.len(), 2, "{chunks:?}");
    assert!(matches!(chunks[0], Ok(StreamChunk::RunStarted { .. })));
    let err = chunks[1].as_ref().expect_err("the summary fails");
    assert!(matches!(err.kind(), EngineError::Context(_)), "{err:?}");
    assert_eq!(err.usage(), Usage::default());
    assert!(err.partial_messages().is_empty());
    lifecycle.assert_closed_once_by("on_run_error");
    assert_eq!(lifecycle.count("RunFinished"), 0);
    assert!(
        sizes.lock().unwrap().is_empty(),
        "the model was never asked"
    );
    assert_nothing_lost(chat.messages());
}

#[tokio::test]
async fn run_returns_err_when_the_pre_run_compaction_fails() {
    let (mut chat, gate, lifecycle, _) = over_budget_chat(
        vec![text_turn("never")],
        Err(ScriptedError("summary refused".into())),
    );
    gate.release.notify_one();

    let err = tokio::time::timeout(STUCK, chat.run("go"))
        .await
        .unwrap()
        .expect_err("the summary fails");

    assert!(matches!(err.kind(), EngineError::Context(_)), "{err:?}");
    lifecycle.assert_closed_once_by("on_run_error");
    assert_nothing_lost(chat.messages());
}

/// Defers every call, so the run suspends on its first step.
struct DeferAll;

#[async_trait]
impl ChatMiddleware for DeferAll {
    async fn on_before_tool_call(&self, _call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        ToolDecision::Defer {
            reason: "needs approval".into(),
        }
    }
}

/// A resumed run does not compact, even with the history over budget:
/// the history ends on the suspended step's tool calls.
#[tokio::test]
async fn resume_does_not_run_the_pre_run_compaction() {
    let gate = SummarizerGate::default();
    let summarizer = GatedSummarizer {
        gate: gate.clone(),
        inner: ScriptedModel::with_turns([summary_turn()]),
    };
    // The seeded turns and the kickoff fit in the budget; the suspended
    // call's arguments put the history over it, with a prefix a
    // compaction would have to summarize.
    let suspending_turn: ScriptedTurn = Ok(vec![
        Ok(StreamChunk::ToolCallStarted {
            call_id: "t1".into(),
            name: "big".into(),
        }),
        Ok(StreamChunk::ToolCallFinished {
            call_id: "t1".into(),
            name: "big".into(),
            args: json!({ "padding": "x".repeat(4_000) }),
        }),
        Ok(StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
            model: None,
        }),
    ]);
    let (model, _) = RecordingModel::new(vec![suspending_turn, text_turn("done")]);
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .history(
            History::builder(1_000)
                .preserve_n_last(1)
                .strategy(Box::new(SummarizeStrategy::new(Arc::new(summarizer)))),
        )
        .middleware(Arc::new(DeferAll))
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let outcome = chat.run("go").await.unwrap();
    assert!(matches!(outcome.finish_reason, FinishReason::Suspended));
    assert_eq!(gate.requests(), 0, "the kickoff fit in the budget");

    let outcome = tokio::time::timeout(
        STUCK,
        chat.resume([("t1".to_string(), ResumeDecision::Approve)]),
    )
    .await
    .expect("resume is stuck on a summary")
    .unwrap();

    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(gate.requests(), 0, "resume does not compact");
}
