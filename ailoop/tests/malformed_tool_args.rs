//! A tool call whose arguments are not valid JSON (typically cut off by
//! `max_tokens`) must never run. The engine records the call with `{}`,
//! answers it with an error `tool_result`, skips every tool hook and
//! lets the run continue so the model can retry.

use std::sync::{Arc, Mutex};

use ailoop::{
    AssistantBlock, ChatMiddleware, Conversation, FinishReason, Message, RunId, StepId,
    StreamChunk, ToolContext, ToolDecision, ToolDefinition, ToolDyn, ToolResultBlock,
    ToolResultContent, Usage, UserBlock,
};
use ailoop_core::testing::ScriptedModel;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Tool that records the args of every execution.
struct Recording {
    calls: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl ToolDyn for Recording {
    fn name(&self) -> String {
        "write_file".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "write_file",
            "stub",
            json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
            vec![],
        )
    }
    async fn call(&self, args: Value, _ctx: &ToolContext) -> ToolResultContent {
        self.calls.lock().unwrap().push(args);
        ToolResultContent::text("written")
    }
}

/// Records tool hooks by name and every `ToolResult` seen by `on_chunk`.
#[derive(Default)]
struct Spy {
    hooks: Mutex<Vec<String>>,
    results: Mutex<Vec<(String, ToolResultContent)>>,
}

#[async_trait]
impl ChatMiddleware for Spy {
    async fn on_chunk(&self, chunk: &StreamChunk) {
        if let StreamChunk::ToolResult {
            call_id, content, ..
        } = chunk
        {
            self.results
                .lock()
                .unwrap()
                .push((call_id.clone(), content.clone()));
        }
    }
    async fn on_before_tool_call(
        &self,
        _: &RunId,
        _: &StepId,
        name: &str,
        _: &Value,
    ) -> ToolDecision {
        self.hooks.lock().unwrap().push(format!("before:{name}"));
        ToolDecision::Continue
    }
    async fn on_after_tool_call(
        &self,
        _: &RunId,
        _: &StepId,
        name: &str,
        _: &Value,
        _: &ToolResultContent,
    ) {
        self.hooks.lock().unwrap().push(format!("after:{name}"));
    }
}

fn started(id: &str) -> StreamChunk {
    StreamChunk::ToolCallStarted {
        id: id.into(),
        name: "write_file".into(),
    }
}

fn finished(reason: FinishReason) -> StreamChunk {
    StreamChunk::TurnFinished {
        reason,
        usage: Usage::default(),
        service_tier: None,
    }
}

fn text_turn(text: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::TextDelta { delta: text.into() },
        finished(FinishReason::EndTurn),
    ]
}

fn result_text(content: &ToolResultContent) -> &str {
    match &content.blocks[0] {
        ToolResultBlock::Text { text } => text,
        other => panic!("expected text block, got {other:?}"),
    }
}

fn build(
    model: ScriptedModel,
) -> (
    Conversation<ScriptedModel>,
    Arc<Spy>,
    Arc<Mutex<Vec<Value>>>,
) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let spy = Arc::new(Spy::default());
    let chat = Conversation::builder(model)
        .tool_dyn(Arc::new(Recording {
            calls: calls.clone(),
        }))
        .middleware(spy.clone())
        .build()
        .expect("build");
    (chat, spy, calls)
}

#[tokio::test]
async fn malformed_call_is_not_executed_and_answered_with_error() {
    let raw = r#"{"path": "a.txt", "content": "hel"#;
    let model = ScriptedModel::new([
        vec![
            started("toolu_1"),
            StreamChunk::tool_call_from_raw_args("toolu_1", "write_file", raw),
            finished(FinishReason::ToolUse),
        ],
        text_turn("retrying later"),
    ]);
    let (mut chat, spy, calls) = build(model);

    let outcome = chat.run("go").await.expect("run");

    // The run went on to the next turn.
    assert!(
        matches!(outcome.finish_reason, FinishReason::EndTurn),
        "unexpected finish: {:?}",
        outcome.finish_reason
    );
    assert_eq!(outcome.final_text.as_deref(), Some("retrying later"));

    // Nothing ran and no tool hook fired.
    assert!(calls.lock().unwrap().is_empty(), "tool must not run");
    assert!(
        spy.hooks.lock().unwrap().is_empty(),
        "tool hooks must not fire: {:?}",
        spy.hooks.lock().unwrap()
    );

    // The synthesized error reached on_chunk.
    let results = spy.results.lock().unwrap();
    assert_eq!(results.len(), 1);
    let (call_id, content) = &results[0];
    assert_eq!(call_id, "toolu_1");
    assert!(content.is_error);
    let text = result_text(content);
    assert!(
        text.starts_with("Invalid JSON arguments for tool 'write_file'"),
        "{text}"
    );
    let wrapper: Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert_eq!(wrapper, json!({ "INVALID_JSON": raw }));

    // History: tool_use with `{}` paired with its error tool_result,
    // then the follow-up assistant turn.
    let msgs = &outcome.new_messages;
    assert_eq!(msgs.len(), 3, "{msgs:?}");
    match &msgs[0] {
        Message::Assistant { blocks } => match blocks.as_slice() {
            [AssistantBlock::ToolCall { id, name, args, .. }] => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "write_file");
                assert_eq!(args, &json!({}));
            }
            other => panic!("expected a single ToolCall, got {other:?}"),
        },
        other => panic!("expected assistant message, got {other:?}"),
    }
    match &msgs[1] {
        Message::User { blocks } => match blocks.as_slice() {
            [
                UserBlock::ToolResult {
                    call_id, content, ..
                },
            ] => {
                assert_eq!(call_id, "toolu_1");
                assert!(content.is_error);
            }
            other => panic!("expected a single ToolResult, got {other:?}"),
        },
        other => panic!("expected user message, got {other:?}"),
    }
    assert!(matches!(&msgs[2], Message::Assistant { .. }));
}

/// A valid and a malformed call in the same turn: only the valid one
/// runs, and both results land in call order.
#[tokio::test]
async fn mixed_turn_runs_only_the_valid_call() {
    let model = ScriptedModel::new([
        vec![
            started("toolu_1"),
            StreamChunk::tool_call_from_raw_args("toolu_1", "write_file", r#"{"path":"#),
            started("toolu_2"),
            StreamChunk::tool_call_from_raw_args("toolu_2", "write_file", r#"{"path":"b.txt"}"#),
            finished(FinishReason::ToolUse),
        ],
        text_turn("done"),
    ]);
    let (mut chat, spy, calls) = build(model);

    let outcome = chat.run("go").await.expect("run");

    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(*calls.lock().unwrap(), vec![json!({"path": "b.txt"})]);
    assert_eq!(
        *spy.hooks.lock().unwrap(),
        vec!["before:write_file", "after:write_file"]
    );

    let results = spy.results.lock().unwrap();
    let ids: Vec<(&str, bool)> = results
        .iter()
        .map(|(id, c)| (id.as_str(), c.is_error))
        .collect();
    assert_eq!(ids, vec![("toolu_1", true), ("toolu_2", false)]);

    match &outcome.new_messages[1] {
        Message::User { blocks } => {
            let ids: Vec<&str> = blocks
                .iter()
                .map(|b| match b {
                    UserBlock::ToolResult { call_id, .. } => call_id.as_str(),
                    other => panic!("expected ToolResult, got {other:?}"),
                })
                .collect();
            assert_eq!(ids, vec!["toolu_1", "toolu_2"]);
        }
        other => panic!("expected user message, got {other:?}"),
    }
}

/// Truncation by `max_tokens` keeps the current continuation rule: the
/// run ends with `MaxTokens`, but history still pairs the call with its
/// error result so the next run is consistent.
#[tokio::test]
async fn malformed_call_on_max_tokens_ends_run_with_consistent_history() {
    let model = ScriptedModel::new([vec![
        started("toolu_1"),
        StreamChunk::tool_call_from_raw_args("toolu_1", "write_file", r#"{"path":"a"#),
        finished(FinishReason::MaxTokens),
    ]]);
    let (mut chat, _spy, calls) = build(model);

    let outcome = chat.run("go").await.expect("run");

    assert!(matches!(outcome.finish_reason, FinishReason::MaxTokens));
    assert!(calls.lock().unwrap().is_empty());
    match outcome.new_messages.last() {
        Some(Message::User { blocks }) => assert!(matches!(
            blocks.as_slice(),
            [UserBlock::ToolResult { call_id, content, .. }] if call_id == "toolu_1" && content.is_error
        )),
        other => panic!("expected trailing tool_result, got {other:?}"),
    }
}
