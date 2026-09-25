//! In-run context management through the public `Conversation`
//! surface: compaction between iterations
//! (`ConversationBuilder::compact_between_iterations`) and recovery from
//! a provider context-window overflow
//! (`ConversationBuilder::recover_from_context_overflow`).
//!
//! Budgets use the default `CharTokenizer` (`len() / 4`). Overflow is
//! simulated with `ScriptedError("context_overflow")`, which reports
//! `ProviderError::is_context_overflow() == true`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ailoop::{
    AssistantBlock, ChatMiddleware, ChatRequest, CompletionModel, Conversation, EngineError,
    FinishReason, History, Message, RunError, RunId, StreamChunk, ToolDefinition,
    ToolResultContent, Usage, UserBlock,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel, ScriptedTurn};
use ailoop_tools::{ToolContext, ToolDyn};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Value, json};

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
    fn name(&self) -> String {
        "big".into()
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
    async fn on_chat_request(&self, _: &RunId, _: &ailoop::StepId, _: &mut ChatRequest) {
        self.chat_requests.fetch_add(1, Ordering::SeqCst);
    }
    async fn on_run_error(
        &self,
        _: &RunId,
        _: &(dyn std::error::Error + Send + Sync),
        _: &[Message],
    ) {
        self.run_errors.fetch_add(1, Ordering::SeqCst);
    }
}

fn tool_call_turn(id: &str) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::ToolCallStarted {
            id: id.into(),
            name: "big".into(),
        }),
        Ok(StreamChunk::ToolCallFinished {
            id: id.into(),
            name: "big".into(),
            args: json!({}),
        }),
        Ok(StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
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
    M: CompletionModel + Send + Sync,
    M::Error: ailoop::ProviderError,
{
    for i in 0..2 {
        chat.history_push(Message::user(format!("{i}{}", "q".repeat(399))));
        chat.history_push(Message::assistant_text(format!("{i}{}", "a".repeat(399))));
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
                    if let AssistantBlock::ToolCall { id, .. } = b {
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
                                    AssistantBlock::ToolCall { id, .. } if id == call_id)));
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
        .with_history(History::builder(1_000).preserve_n_last(1))
        .compact_between_iterations(true)
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let chunks = collect(&mut chat, "go").await;
    let chunks: Vec<StreamChunk> = chunks.into_iter().map(|c| c.expect("no error")).collect();

    let run_id = match &chunks[0] {
        StreamChunk::RunStarted { run_id } => run_id.clone(),
        other => panic!("no pre-run compaction expected, got {other:?}"),
    };
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

    let history = chat.history_messages();
    assert_eq!(history.len(), 4, "kickoff + pair + final text");
    assert!(matches!(&history[0], Message::User { .. }));
    assert_no_orphans(history);
}

#[tokio::test]
async fn does_not_compact_between_iterations_by_default() {
    let (model, sizes) = RecordingModel::new(vec![tool_call_turn("t1"), text_turn("done")]);
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .with_history(History::builder(1_000).preserve_n_last(1))
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let chunks = collect(&mut chat, "go").await;
    assert!(compacted_counts(&chunks).is_empty());
    assert_eq!(*sizes.lock().unwrap(), vec![5, 7]);
    assert_eq!(chat.history_messages().len(), 8);
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
        .with_history(History::builder(100_000).preserve_n_last(1))
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

    let history = chat.history_messages();
    assert_eq!(history.len(), 2, "kickoff + reply");
    assert!(matches!(&history[1], Message::Assistant { .. }));
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
        .with_history(History::builder(100_000).preserve_n_last(1))
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let outcome = chat.run("go").await.expect("recovered");
    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(outcome.new_messages.len(), 3);
    assert_eq!(*sizes.lock().unwrap(), vec![5, 7, 3]);

    let history = chat.history_messages();
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
        .with_history(History::builder(100_000).preserve_n_last(1))
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);
    let before: Vec<String> = chat
        .history_messages()
        .iter()
        .map(|m| format!("{m:?}"))
        .collect();

    let err = chat.run("go").await.expect_err("overflow persists");
    match err.into_kind() {
        EngineError::ContextOverflow(ScriptedError(msg)) => {
            assert!(msg.contains("context_overflow"))
        }
        other => panic!("expected ContextOverflow, got {other:?}"),
    }
    assert_eq!(*sizes.lock().unwrap(), vec![5, 7, 3]);
    assert_eq!(counters.run_errors.load(Ordering::SeqCst), 1);

    let history = chat.history_messages();
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
        .with_history(History::builder(100_000).preserve_n_last(1))
        .build()
        .unwrap();

    let err = chat.run("go").await.expect_err("cannot shrink");
    assert!(
        matches!(err.kind(), EngineError::ContextOverflow(_)),
        "{err:?}"
    );
    assert_eq!(*sizes.lock().unwrap(), vec![1]);
    assert_eq!(chat.history_messages().len(), 1);
}

#[tokio::test]
async fn overflow_is_a_model_error_when_recovery_is_off() {
    let (model, sizes) = RecordingModel::new(vec![overflow_turn(), text_turn("unused")]);
    let mut chat = Conversation::builder(model)
        .with_history(History::builder(100_000).preserve_n_last(1))
        .recover_from_context_overflow(false)
        .build()
        .unwrap();
    seed_prior_turns(&mut chat);

    let err = chat.run("go").await.expect_err("no recovery");
    assert!(matches!(err.kind(), EngineError::Model(_)), "{err:?}");
    assert_eq!(*sizes.lock().unwrap(), vec![5]);
    assert_eq!(chat.history_messages().len(), 5);
}

/// Dropping the stream mid-run discards the run's changes, including a
/// compaction it already performed, as before in-run compaction existed.
#[tokio::test]
async fn dropping_the_stream_mid_run_rolls_back_history() {
    let (model, _) = RecordingModel::new(vec![tool_call_turn("t1"), text_turn("done")]);
    let mut chat = Conversation::builder(model)
        .tool(BigTool)
        .with_history(History::builder(1_000).preserve_n_last(1))
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

    assert_eq!(chat.history_messages().len(), 5, "prior turns + kickoff");
}
