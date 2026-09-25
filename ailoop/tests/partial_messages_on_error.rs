//! A run that ends in `Err` rolls the conversation history back, but
//! the steps it completed before failing (including tools that already
//! ran) travel in `RunError::partial_messages`, and `on_run_error`
//! receives the same list. The step that failed is never included, so
//! the partial list never holds a `tool_use` without its `tool_result`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ailoop::{
    AssistantBlock, ChatMiddleware, ChatRequest, Conversation, EngineError, FinishReason, Message,
    RunConfig, RunError, RunErrorInfo, StepInfo, StreamChunk, ToolContext, ToolDefinition, ToolDyn,
    ToolRegistry, ToolResultContent, Usage, UserBlock, advanced::run_chat,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel, ScriptedTurn};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

/// Tool with a side effect: counts its executions.
struct Counter {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ToolDyn for Counter {
    fn name(&self) -> String {
        "bump".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new("bump", "stub", json!({"type":"object"}), vec![])
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResultContent {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        ToolResultContent::text(format!("bumped {n}"))
    }
}

/// Records what `on_run_error` receives and the size of every request.
#[derive(Default)]
struct Spy {
    run_errors: Mutex<Vec<Vec<Message>>>,
    request_sizes: Mutex<Vec<usize>>,
}

#[async_trait]
impl ChatMiddleware for Spy {
    async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
        self.request_sizes.lock().unwrap().push(req.messages.len());
    }
    async fn on_run_error(&self, run: &RunErrorInfo<'_>) {
        self.run_errors
            .lock()
            .unwrap()
            .push(run.partial_messages.to_vec());
    }
}

fn finished(reason: FinishReason) -> StreamChunk {
    StreamChunk::TurnFinished {
        reason,
        usage: Usage::default(),
        service_tier: None,
    }
}

fn tool_turn(id: &str) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::TextDelta {
            delta: format!("calling {id}"),
        }),
        Ok(StreamChunk::ToolCallStarted {
            call_id: id.into(),
            name: "bump".into(),
        }),
        Ok(StreamChunk::ToolCallFinished {
            call_id: id.into(),
            name: "bump".into(),
            args: json!({}),
        }),
        Ok(finished(FinishReason::ToolUse)),
    ])
}

/// Streams some text, then the connection drops.
fn broken_turn() -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::TextDelta {
            delta: "half an ans".into(),
        }),
        Err(ScriptedError("connection dropped".into())),
    ])
}

fn text_turn(text: &str) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::TextDelta { delta: text.into() }),
        Ok(finished(FinishReason::EndTurn)),
    ])
}

fn build(turns: Vec<ScriptedTurn>) -> (Conversation<ScriptedModel>, Arc<Spy>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let spy = Arc::new(Spy::default());
    let chat = Conversation::builder(ScriptedModel::with_turns(turns))
        .tool_dyn(Arc::new(Counter {
            calls: calls.clone(),
        }))
        .middleware(spy.clone())
        .build()
        .expect("build");
    (chat, spy, calls)
}

/// Every `tool_use` is answered in the next message and every
/// `tool_result` answers the previous one.
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

fn tool_call_ids(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .flat_map(|m| match m {
            Message::Assistant { blocks } => blocks.as_slice(),
            _ => &[],
        })
        .filter_map(|b| match b {
            AssistantBlock::ToolCall { call_id: id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect()
}

fn debug(messages: &[Message]) -> Vec<String> {
    messages.iter().map(|m| format!("{m:?}")).collect()
}

fn assert_two_complete_steps(err: &RunError<ScriptedError>) {
    assert!(
        matches!(err.kind(), EngineError::Model(ScriptedError(msg)) if msg == "connection dropped"),
        "{err:?}"
    );
    let partial = err.partial_messages();
    assert_eq!(
        partial.len(),
        4,
        "assistant/user pair per step: {partial:?}"
    );
    assert!(matches!(partial[0], Message::Assistant { .. }));
    assert!(matches!(partial[1], Message::User { .. }));
    assert_eq!(tool_call_ids(partial), ["toolu_1", "toolu_2"]);
    assert_no_orphans(partial);
    assert!(
        !format!("{partial:?}").contains("half an ans"),
        "the failed step must not leak into the partial list"
    );
}

#[tokio::test]
async fn error_in_third_step_carries_the_two_completed_steps() {
    let (mut chat, spy, calls) = build(vec![
        tool_turn("toolu_1"),
        tool_turn("toolu_2"),
        broken_turn(),
    ]);

    let err = chat.run("go").await.expect_err("third step fails");

    assert_two_complete_steps(&err);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        err.to_string(),
        "model error: scripted error: connection dropped"
    );

    let seen = spy.run_errors.lock().unwrap();
    assert_eq!(seen.len(), 1, "on_run_error fires once");
    assert_eq!(debug(&seen[0]), debug(err.partial_messages()));
}

#[tokio::test]
async fn history_is_rolled_back_after_err() {
    let (mut chat, _spy, _calls) = build(vec![
        text_turn("hello"),
        tool_turn("toolu_1"),
        tool_turn("toolu_2"),
        broken_turn(),
    ]);
    chat.run("first").await.expect("first run succeeds");
    let before = debug(chat.messages());

    let err = chat.run("second").await.expect_err("third step fails");
    assert_two_complete_steps(&err);

    let history = chat.messages();
    assert_eq!(history.len(), before.len() + 1, "prior turns + kickoff");
    assert_eq!(debug(&history[..before.len()]), before);
    assert!(matches!(history.last(), Some(Message::User { .. })));
}

#[tokio::test]
async fn reapplying_partial_messages_leaves_a_valid_history() {
    let (mut chat, spy, calls) = build(vec![
        tool_turn("toolu_1"),
        tool_turn("toolu_2"),
        broken_turn(),
        text_turn("done"),
    ]);

    let err = chat.run("go").await.expect_err("third step fails");
    chat.extend_messages(err.partial_messages().iter().cloned());

    let history = chat.messages();
    assert_eq!(history.len(), 5, "kickoff + two completed steps");
    assert_eq!(debug(&history[1..]), debug(err.partial_messages()));
    assert_no_orphans(history);

    spy.request_sizes.lock().unwrap().clear();
    let outcome = chat.run("continue").await.expect("run after re-apply");
    assert_eq!(outcome.final_text.as_deref(), Some("done"));
    assert_eq!(
        *spy.request_sizes.lock().unwrap(),
        vec![6],
        "the next request carries the re-applied steps"
    );
    assert_no_orphans(chat.messages());
    assert_eq!(calls.load(Ordering::SeqCst), 2, "no tool ran twice");
}

#[tokio::test]
async fn error_in_first_step_has_no_partial_messages() {
    for first in [
        broken_turn(),
        Err(ScriptedError("permanent: bad auth".into())),
    ] {
        let (mut chat, spy, calls) = build(vec![first]);

        let err = chat.run("go").await.expect_err("first step fails");

        assert!(matches!(err.kind(), EngineError::Model(_)), "{err:?}");
        assert!(err.partial_messages().is_empty(), "{err:?}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(chat.messages().len(), 1, "only the kickoff");
        assert_eq!(*spy.run_errors.lock().unwrap(), vec![Vec::<Message>::new()]);
    }
}

#[tokio::test]
async fn stream_yields_the_partial_messages_in_its_err_item() {
    let (mut chat, _spy, _calls) = build(vec![
        tool_turn("toolu_1"),
        tool_turn("toolu_2"),
        broken_turn(),
    ]);

    let chunks: Vec<_> = chat.stream("go").await.expect("run starts").collect().await;

    let last_step = chunks
        .iter()
        .filter_map(|c| match c {
            Ok(StreamChunk::StepFinished {
                new_messages_so_far,
                ..
            }) => Some(new_messages_so_far.clone()),
            _ => None,
        })
        .next_back()
        .expect("two steps finished");
    let err = chunks
        .into_iter()
        .last()
        .expect("stream yields items")
        .expect_err("last item is the error");
    assert_two_complete_steps(&err);
    assert_eq!(debug(err.partial_messages()), debug(&last_step));
}

#[tokio::test]
async fn run_chat_carries_the_partial_messages_too() {
    let model =
        ScriptedModel::with_turns([tool_turn("toolu_1"), tool_turn("toolu_2"), broken_turn()]);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut registry = ToolRegistry::new();
    registry
        .register(Arc::new(Counter {
            calls: calls.clone(),
        }))
        .unwrap();

    let stream = run_chat(
        &model,
        vec![Message::user("go")],
        &registry,
        RunConfig::default(),
    )
    .await
    .expect("run starts");
    let chunks: Vec<_> = stream.collect().await;

    let err = chunks
        .into_iter()
        .last()
        .expect("stream yields items")
        .expect_err("last item is the error");
    assert_two_complete_steps(&err);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn question_mark_still_converts_into_engine_error() {
    async fn propagate(
        chat: &mut Conversation<ScriptedModel>,
    ) -> Result<(), EngineError<ScriptedError>> {
        chat.run("go").await?;
        Ok(())
    }

    let (mut chat, _spy, _calls) = build(vec![broken_turn()]);
    let err = propagate(&mut chat).await.expect_err("fails");
    assert!(matches!(err, EngineError::Model(_)));
}
