//! Integration test that drives the engine end-to-end and asserts the
//! exact sequence of `ChatMiddleware` hook invocations. The unit tests
//! in `tracing_middleware.rs` call hooks directly, so they cannot catch
//! a regression where the engine stops firing a hook or fires it out of
//! order. This test does.

use std::sync::{Arc, Mutex};

use ailoop::{Message, ToolDefinition, ToolResultContent, advanced::run_chat};
use ailoop_core::testing::{ScriptedError, ScriptedModel};
use ailoop_core::{
    ChatMiddleware, ChatRequest, FinishReason, HookAction, RunConfig, RunId, StepId, StreamChunk,
    ToolDecision, Usage,
};
use ailoop_tools::{ToolRegistry, registry::ToolDyn};
use futures::StreamExt;
use serde_json::{Value, json};

/// Records the name of every `ChatMiddleware` hook the engine invokes,
/// in order. `on_chunk` entries also carry the `StreamChunk` variant so
/// the assertion can distinguish a `RunStarted` chunk from a `ToolResult`
/// chunk without a separate Vec.
#[derive(Default, Clone)]
struct RecordingMiddleware {
    log: Arc<Mutex<Vec<String>>>,
}

impl RecordingMiddleware {
    fn new() -> Self {
        Self::default()
    }

    fn entries(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }

    fn push(&self, label: impl Into<String>) {
        self.log.lock().unwrap().push(label.into());
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for RecordingMiddleware {
    async fn on_run_start(
        &self,
        _run_id: &RunId,
        _messages: &[Message],
        _config: &RunConfig,
    ) -> HookAction {
        self.push("on_run_start");
        HookAction::Continue
    }

    async fn on_chat_request(&self, _run_id: &RunId, _step_id: &StepId, _req: &mut ChatRequest) {
        self.push("on_chat_request");
    }

    async fn on_chunk(&self, chunk: &StreamChunk) {
        let label = match chunk {
            StreamChunk::RunStarted { .. } => "on_chunk:RunStarted",
            StreamChunk::StepStarted { .. } => "on_chunk:StepStarted",
            StreamChunk::TextDelta { .. } => "on_chunk:TextDelta",
            StreamChunk::ReasoningDelta { .. } => "on_chunk:ReasoningDelta",
            StreamChunk::ReasoningEnd { .. } => "on_chunk:ReasoningEnd",
            StreamChunk::RedactedReasoningBlock { .. } => "on_chunk:RedactedReasoningBlock",
            StreamChunk::ToolCallStart { .. } => "on_chunk:ToolCallStart",
            StreamChunk::ToolCallArgsDelta { .. } => "on_chunk:ToolCallArgsDelta",
            StreamChunk::ToolCallEnd { .. } => "on_chunk:ToolCallEnd",
            StreamChunk::TurnFinished { .. } => "on_chunk:TurnFinished",
            StreamChunk::ToolResult { .. } => "on_chunk:ToolResult",
            StreamChunk::StepFinished { .. } => "on_chunk:StepFinished",
            StreamChunk::RunFinished { .. } => "on_chunk:RunFinished",
            StreamChunk::HistoryCompacted { .. } => "on_chunk:HistoryCompacted",
            _ => "on_chunk:Unknown",
        };
        self.push(label);
    }

    async fn on_before_tool_call(
        &self,
        _run_id: &RunId,
        _step_id: &StepId,
        _name: &str,
        _args: &Value,
    ) -> ToolDecision {
        self.push("on_before_tool_call");
        ToolDecision::Continue
    }

    async fn on_after_tool_call(
        &self,
        _run_id: &RunId,
        _step_id: &StepId,
        _name: &str,
        _args: &Value,
        _result: &ToolResultContent,
    ) {
        self.push("on_after_tool_call");
    }

    async fn on_run_finished(
        &self,
        _run_id: &RunId,
        _reason: &FinishReason,
        _usage: &Usage,
        _new_messages: &[Message],
    ) {
        self.push("on_run_finished");
    }

    async fn on_run_error(&self, _run_id: &RunId, _err: &(dyn std::error::Error + Send + Sync)) {
        self.push("on_run_error");
    }
}

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
    async fn call(&self, _: Value) -> ToolResultContent {
        ToolResultContent::Text("sunny".into())
    }
}

/// Two-turn run (turn 1 issues a tool call, turn 2 ends the run).
/// Asserts the engine fires every hook on `RecordingMiddleware` in the
/// expected order. If the engine drops a hook or reorders one, this
/// catches it where the unit tests in `tracing_middleware.rs` cannot.
#[tokio::test]
async fn engine_invokes_middleware_hooks_in_order() {
    let turn1 = vec![
        StreamChunk::TextDelta {
            delta: "let me check ".into(),
        },
        StreamChunk::ToolCallStart {
            id: "toolu_1".into(),
            name: "get_weather".into(),
        },
        StreamChunk::ToolCallEnd {
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
    let turn2 = vec![
        StreamChunk::TextDelta {
            delta: "it's sunny".into(),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        },
    ];

    let model = ScriptedModel::new([turn1, turn2]);
    let recorder = RecordingMiddleware::new();
    let mw: Arc<dyn ChatMiddleware> = Arc::new(recorder.clone());

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(GetWeather)).unwrap();

    let mut config = RunConfig::default();
    config.middlewares = vec![mw];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let _: Vec<_> = stream.collect().await;

    let entries = recorder.entries();
    let expected = vec![
        "on_run_start",
        "on_chunk:RunStarted",
        "on_chunk:StepStarted",
        "on_chat_request",
        "on_chunk:TextDelta",
        "on_chunk:ToolCallStart",
        "on_chunk:ToolCallEnd",
        "on_chunk:TurnFinished",
        "on_before_tool_call",
        "on_after_tool_call",
        "on_chunk:ToolResult",
        "on_chunk:StepFinished",
        "on_chunk:StepStarted",
        "on_chat_request",
        "on_chunk:TextDelta",
        "on_chunk:TurnFinished",
        "on_chunk:StepFinished",
        "on_run_finished",
        "on_chunk:RunFinished",
    ];
    assert_eq!(entries, expected, "hook lifecycle deviated from contract");
}

/// A mid-stream `Err` from the model must surface through `on_run_error`
/// (and not `on_run_finished`). Uses the `with_turns` API to script a
/// stream that emits one Ok chunk and then fails — the SSE-drop scenario
/// a future `RetryingModel<M>` will need to test against.
#[tokio::test]
async fn mid_stream_error_fires_on_run_error_not_on_run_finished() {
    let model = ScriptedModel::with_turns([Ok(vec![
        Ok(StreamChunk::TextDelta {
            delta: "partial".into(),
        }),
        Err(ScriptedError("connection dropped".into())),
    ])]);
    let recorder = RecordingMiddleware::new();
    let mw: Arc<dyn ChatMiddleware> = Arc::new(recorder.clone());
    let registry = ToolRegistry::new();

    let mut config = RunConfig::default();
    config.middlewares = vec![mw];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start (stream opens before mid-stream Err)");
    let chunks: Vec<_> = stream.collect().await;

    assert!(
        chunks.iter().any(|c| c.is_err()),
        "expected the mid-stream Err to propagate to the engine consumer"
    );

    let entries = recorder.entries();
    assert!(
        entries.contains(&"on_run_error".to_string()),
        "expected on_run_error to fire on mid-stream Err, got: {entries:?}"
    );
    assert!(
        !entries.contains(&"on_run_finished".to_string()),
        "on_run_finished must not fire on the error path, got: {entries:?}"
    );
}
