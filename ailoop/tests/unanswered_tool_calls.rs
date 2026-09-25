//! A run that aborts part-way through a step still answers every tool
//! call of that step: calls that completed keep their result, and the
//! calls the abort stopped get an error result plus a `ToolResult`
//! chunk. The history never holds a `tool_use` without its
//! `tool_result`, so the conversation can go on after the abort.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ailoop::{
    AbortReason, AssistantBlock, CancellationToken, ChatMiddleware, ChatRequest, CompletionModel,
    Conversation, FinishReason, Message, RunOptions, StreamChunk, ToolCallInfo, ToolDecision,
    ToolDefinition, ToolResultBlock, ToolResultContent, Usage, UserBlock,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel};
use ailoop_tools::{ToolContext, ToolDyn};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Value, json};

fn definition(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        name,
        "stub",
        json!({"type":"object","properties":{},"required":[]}),
        vec![],
    )
}

struct GetWeather;

#[async_trait]
impl ToolDyn for GetWeather {
    fn name(&self) -> &str {
        "get_weather"
    }
    fn tool_definition(&self) -> ToolDefinition {
        definition("get_weather")
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        ToolResultContent::text("sunny")
    }
}

/// Never returns; cancels `cancel` first when set, so the run aborts
/// while this tool is in flight.
struct Hang {
    cancel: Option<CancellationToken>,
}

#[async_trait]
impl ToolDyn for Hang {
    fn name(&self) -> &str {
        "hang"
    }
    fn tool_definition(&self) -> ToolDefinition {
        definition("hang")
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        std::future::pending().await
    }
}

/// Records the `call_id` of every `ToolResult` chunk.
#[derive(Default)]
struct ToolResultChunks(Mutex<Vec<String>>);

impl ToolResultChunks {
    fn ids(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatMiddleware for ToolResultChunks {
    async fn on_chunk(&self, chunk: &StreamChunk) {
        if let StreamChunk::ToolResult { call_id, .. } = chunk {
            self.0.lock().unwrap().push(call_id.clone());
        }
    }
}

fn call(id: &str, name: &str) -> StreamChunk {
    StreamChunk::ToolCallFinished {
        call_id: id.into(),
        name: name.into(),
        args: json!({}),
    }
}

fn turn_finished(reason: FinishReason) -> StreamChunk {
    StreamChunk::TurnFinished {
        reason,
        usage: Usage::default(),
        service_tier: None,
    }
}

/// One turn calling `first` then `get_weather`.
fn two_calls(first: &str) -> Vec<StreamChunk> {
    vec![
        call("toolu_a", first),
        call("toolu_b", "get_weather"),
        turn_finished(FinishReason::ToolUse),
    ]
}

fn end_turn() -> Vec<StreamChunk> {
    vec![
        StreamChunk::TextDelta { delta: "ok".into() },
        turn_finished(FinishReason::EndTurn),
    ]
}

/// The `(call_id, content)` of every tool result in `messages`.
fn tool_results(messages: &[Message]) -> Vec<(String, ToolResultContent)> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::User { blocks } => Some(blocks),
            _ => None,
        })
        .flatten()
        .filter_map(|b| match b {
            UserBlock::ToolResult {
                call_id, content, ..
            } => Some((call_id.clone(), content.clone())),
            _ => None,
        })
        .collect()
}

/// What a provider checks on replay: every assistant message with tool
/// calls is followed by a user message answering exactly those calls.
fn check_pairing(messages: &[Message]) -> Result<(), String> {
    for (i, message) in messages.iter().enumerate() {
        let Message::Assistant { blocks } = message else {
            continue;
        };
        let calls: Vec<&str> = blocks
            .iter()
            .filter_map(|b| match b {
                AssistantBlock::ToolCall { call_id: id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        if calls.is_empty() {
            continue;
        }
        let answered: Vec<String> = tool_results(&messages[i + 1..(i + 2).min(messages.len())])
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        if answered != calls {
            return Err(format!(
                "message {i}: tool_use {calls:?} answered by {answered:?}"
            ));
        }
    }
    Ok(())
}

fn assert_not_run(content: &ToolResultContent, reason: &str) {
    assert!(content.is_error, "a call that did not run is an error");
    assert_eq!(
        content.blocks,
        vec![ToolResultBlock::Text {
            text: format!("Tool not run: the run was aborted ({reason})"),
        }]
    );
}

#[tokio::test]
async fn timeout_during_first_call_answers_both_calls() {
    let timeout = Duration::from_millis(20);
    let chunks = Arc::new(ToolResultChunks::default());
    let mut chat = Conversation::builder(ScriptedModel::new([two_calls("hang")]))
        .tool(Hang { cancel: None })
        .tool(GetWeather)
        .middleware(chunks.clone())
        .build()
        .expect("build");

    let outcome = chat
        .run_with_options("hi", RunOptions::new().timeout(timeout))
        .await
        .expect("aborts are Ok");

    assert!(matches!(
        outcome.finish_reason,
        FinishReason::Aborted(AbortReason::Timeout(_))
    ));
    let results = tool_results(&outcome.new_messages);
    let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["toolu_a", "toolu_b"]);
    for (_, content) in &results {
        assert_not_run(content, &AbortReason::Timeout(timeout).to_string());
    }
    assert_eq!(chunks.ids(), ["toolu_a", "toolu_b"]);
    check_pairing(chat.messages()).unwrap();
}

#[tokio::test]
async fn cancellation_during_first_call_answers_both_calls() {
    let token = CancellationToken::new();
    let chunks = Arc::new(ToolResultChunks::default());
    let mut chat = Conversation::builder(ScriptedModel::new([two_calls("hang")]))
        .tool(Hang {
            cancel: Some(token.clone()),
        })
        .tool(GetWeather)
        .middleware(chunks.clone())
        .build()
        .expect("build");

    let outcome = chat
        .run_with_options("hi", RunOptions::new().cancellation(token))
        .await
        .expect("aborts are Ok");

    assert!(matches!(
        outcome.finish_reason,
        FinishReason::Aborted(AbortReason::Cancelled)
    ));
    let results = tool_results(&outcome.new_messages);
    let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["toolu_a", "toolu_b"]);
    for (_, content) in &results {
        assert_not_run(content, "cancelled by caller");
    }
    assert_eq!(chunks.ids(), ["toolu_a", "toolu_b"]);
    check_pairing(chat.messages()).unwrap();
}

/// Streams one finished tool call, then never ends.
struct StallsAfterToolCall;

#[async_trait]
impl CompletionModel for StallsAfterToolCall {
    type Error = ScriptedError;

    fn name(&self) -> &str {
        "stalls"
    }

    fn model(&self) -> &str {
        "stalls"
    }

    async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        Ok(futures::stream::iter([Ok(call("toolu_a", "get_weather"))])
            .chain(futures::stream::pending())
            .boxed())
    }
}

#[tokio::test]
async fn abort_mid_stream_answers_finished_tool_calls() {
    let chunks = Arc::new(ToolResultChunks::default());
    let mut chat = Conversation::builder(StallsAfterToolCall)
        .tool(GetWeather)
        .middleware(chunks.clone())
        .build()
        .expect("build");

    let outcome = chat
        .run_with_options("hi", RunOptions::new().timeout(Duration::from_millis(20)))
        .await
        .expect("aborts are Ok");

    assert!(matches!(
        outcome.finish_reason,
        FinishReason::Aborted(AbortReason::Timeout(_))
    ));
    let results = tool_results(&outcome.new_messages);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "toolu_a");
    assert!(results[0].1.is_error);
    assert_eq!(chunks.ids(), ["toolu_a"]);
    check_pairing(chat.messages()).unwrap();
}

/// Rejects a request whose history leaves a `tool_use` unanswered, the
/// way providers do; otherwise plays the scripted turns.
struct StrictModel(ScriptedModel);

#[async_trait]
impl CompletionModel for StrictModel {
    type Error = ScriptedError;

    fn name(&self) -> &str {
        self.0.name()
    }

    fn model(&self) -> &str {
        self.0.model()
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        check_pairing(&req.messages).map_err(ScriptedError)?;
        self.0.chat_stream(req).await
    }
}

#[tokio::test]
async fn conversation_continues_after_terminate_mid_step() {
    struct TerminateFirst;

    #[async_trait]
    impl ChatMiddleware for TerminateFirst {
        async fn on_before_tool_call(&self, call: &ToolCallInfo, _: &Value) -> ToolDecision {
            if call.call_id == "toolu_a" {
                ToolDecision::Terminate {
                    reason: "policy".into(),
                }
            } else {
                ToolDecision::Continue
            }
        }
    }

    let model = StrictModel(ScriptedModel::new([two_calls("get_weather"), end_turn()]));
    let mut chat = Conversation::builder(model)
        .tool(GetWeather)
        .middleware(Arc::new(TerminateFirst))
        .build()
        .expect("build");

    let outcome = chat.run("hi").await.expect("aborts are Ok");
    assert!(matches!(
        outcome.finish_reason,
        FinishReason::Aborted(AbortReason::ToolTerminated { .. })
    ));
    check_pairing(chat.messages()).unwrap();

    let next = chat
        .run("go on")
        .await
        .expect("history after the abort is a valid request");
    assert!(matches!(next.finish_reason, FinishReason::EndTurn));
}
