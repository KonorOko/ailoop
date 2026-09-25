//! Integration test that drives the engine end-to-end and asserts the
//! exact sequence of `ChatMiddleware` hook invocations. The unit tests
//! in `tracing_middleware.rs` call hooks directly, so they cannot catch
//! a regression where the engine stops firing a hook or fires it out of
//! order. This test does.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ailoop::{Message, ToolDefinition, ToolResultContent, advanced::run_chat};
use ailoop_core::testing::{ScriptedError, ScriptedModel};
use ailoop_core::{
    ChatMiddleware, ChatRequest, FinishReason, HookAction, RunConfig, RunId, StepId, StreamChunk,
    ToolCallInfo, ToolDecision, Usage,
};
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
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
    async fn on_run_started(
        &self,
        _run_id: &RunId,
        _messages: &[Message],
        _config: &RunConfig,
    ) -> HookAction {
        self.push("on_run_started");
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
            StreamChunk::ReasoningFinished { .. } => "on_chunk:ReasoningFinished",
            StreamChunk::RedactedReasoningBlock { .. } => "on_chunk:RedactedReasoningBlock",
            StreamChunk::ToolCallStarted { .. } => "on_chunk:ToolCallStarted",
            StreamChunk::ToolCallArgsDelta { .. } => "on_chunk:ToolCallArgsDelta",
            StreamChunk::ToolCallFinished { .. } => "on_chunk:ToolCallFinished",
            StreamChunk::TurnFinished { .. } => "on_chunk:TurnFinished",
            StreamChunk::ToolResult { .. } => "on_chunk:ToolResult",
            StreamChunk::StepFinished { .. } => "on_chunk:StepFinished",
            StreamChunk::RunFinished { .. } => "on_chunk:RunFinished",
            StreamChunk::HistoryCompacted { .. } => "on_chunk:HistoryCompacted",
            _ => "on_chunk:Unknown",
        };
        self.push(label);
    }

    async fn on_before_tool_call(&self, _call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        self.push("on_before_tool_call");
        ToolDecision::Continue
    }

    async fn on_after_tool_call(
        &self,
        _call: &ToolCallInfo,
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

    async fn on_run_error(
        &self,
        _run_id: &RunId,
        _err: &(dyn std::error::Error + Send + Sync),
        _usage: &Usage,
        _: &[Message],
    ) {
        self.push("on_run_error");
    }

    fn on_run_dropped(&self, _run_id: &RunId) {
        self.push("on_run_dropped");
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
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        ToolResultContent::text("sunny")
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
        "on_run_started",
        "on_chunk:RunStarted",
        "on_chunk:StepStarted",
        "on_chat_request",
        "on_chunk:TextDelta",
        "on_chunk:ToolCallStarted",
        "on_chunk:ToolCallFinished",
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
    assert!(
        !entries.contains(&"on_run_dropped".to_string()),
        "on_run_dropped must not fire once on_run_error closed the run, got: {entries:?}"
    );
}

fn count(entries: &[String], hook: &str) -> usize {
    entries.iter().filter(|e| *e == hook).count()
}

fn text_turn(text: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::TextDelta { delta: text.into() },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        },
    ]
}

/// Polls `stream` until it has nothing ready for a while, i.e. the
/// engine is parked on a hook that never resolves.
async fn drain_until_stalled<S: futures::Stream + Unpin>(stream: &mut S) {
    while let Ok(Some(_)) = tokio::time::timeout(Duration::from_millis(50), stream.next()).await {}
}

/// A caller that drops the stream mid-run gets neither `on_run_finished`
/// nor `on_run_error`; `on_run_dropped` is the only closing hook, and it
/// fires exactly once per middleware.
#[tokio::test]
async fn dropping_the_stream_mid_run_fires_on_run_dropped_once() {
    let model = ScriptedModel::new([text_turn("hello")]);
    let recorders = [RecordingMiddleware::new(), RecordingMiddleware::new()];
    let registry = ToolRegistry::new();

    let mut config = RunConfig::default();
    config.middlewares = recorders
        .iter()
        .map(|r| Arc::new(r.clone()) as Arc<dyn ChatMiddleware>)
        .collect();

    let mut stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    while let Some(chunk) = stream.next().await {
        if matches!(chunk.unwrap(), StreamChunk::TextDelta { .. }) {
            break;
        }
    }
    drop(stream);

    for recorder in &recorders {
        let entries = recorder.entries();
        assert_eq!(count(&entries, "on_run_dropped"), 1, "{entries:?}");
        assert_eq!(count(&entries, "on_run_finished"), 0, "{entries:?}");
        assert_eq!(count(&entries, "on_run_error"), 0, "{entries:?}");
        assert_eq!(entries.last().map(String::as_str), Some("on_run_dropped"));
    }
}

/// A stream dropped before its first poll never started a run, so no
/// hook fires at all.
#[tokio::test]
async fn dropping_an_unpolled_stream_fires_nothing() {
    let model = ScriptedModel::new([text_turn("hello")]);
    let recorder = RecordingMiddleware::new();
    let registry = ToolRegistry::new();

    let mut config = RunConfig::default();
    config.middlewares = vec![Arc::new(recorder.clone())];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    drop(stream);

    assert!(recorder.entries().is_empty(), "{:?}", recorder.entries());
}

/// Parks forever in one lifecycle hook, recording that it got there.
struct StallingMiddleware {
    in_started: bool,
    log: RecordingMiddleware,
}

#[async_trait::async_trait]
impl ChatMiddleware for StallingMiddleware {
    async fn on_run_started(
        &self,
        _run_id: &RunId,
        _messages: &[Message],
        _config: &RunConfig,
    ) -> HookAction {
        self.log.push("on_run_started");
        if self.in_started {
            std::future::pending::<()>().await;
        }
        HookAction::Continue
    }

    async fn on_run_finished(
        &self,
        _run_id: &RunId,
        _reason: &FinishReason,
        _usage: &Usage,
        _new_messages: &[Message],
    ) {
        self.log.push("on_run_finished");
        if !self.in_started {
            std::future::pending::<()>().await;
        }
    }

    fn on_run_dropped(&self, _run_id: &RunId) {
        self.log.push("on_run_dropped");
    }
}

/// A drop that lands while the engine is firing `on_run_finished` only
/// reaches the middlewares it has not called yet: the ones before (and
/// the one it was awaiting) already had their closing hook.
#[tokio::test]
async fn drop_during_closing_hooks_only_reaches_the_rest() {
    let model = ScriptedModel::new([text_turn("hello")]);
    let before = RecordingMiddleware::new();
    let stalled = RecordingMiddleware::new();
    let after = RecordingMiddleware::new();
    let registry = ToolRegistry::new();

    let mut config = RunConfig::default();
    config.middlewares = vec![
        Arc::new(before.clone()),
        Arc::new(StallingMiddleware {
            in_started: false,
            log: stalled.clone(),
        }),
        Arc::new(after.clone()),
    ];

    let mut stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    drain_until_stalled(&mut stream).await;
    drop(stream);

    let (before, stalled, after) = (before.entries(), stalled.entries(), after.entries());
    assert_eq!(count(&before, "on_run_finished"), 1, "{before:?}");
    assert_eq!(count(&before, "on_run_dropped"), 0, "{before:?}");
    assert_eq!(count(&stalled, "on_run_finished"), 1, "{stalled:?}");
    assert_eq!(count(&stalled, "on_run_dropped"), 0, "{stalled:?}");
    assert_eq!(count(&after, "on_run_finished"), 0, "{after:?}");
    assert_eq!(count(&after, "on_run_dropped"), 1, "{after:?}");
}

/// A drop while `on_run_started` is still running reaches every
/// middleware, including the ones the engine had not started yet.
#[tokio::test]
async fn drop_during_on_run_started_reaches_every_middleware() {
    let model = ScriptedModel::new([text_turn("hello")]);
    let stalled = RecordingMiddleware::new();
    let after = RecordingMiddleware::new();
    let registry = ToolRegistry::new();

    let mut config = RunConfig::default();
    config.middlewares = vec![
        Arc::new(StallingMiddleware {
            in_started: true,
            log: stalled.clone(),
        }),
        Arc::new(after.clone()),
    ];

    let mut stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    drain_until_stalled(&mut stream).await;
    drop(stream);

    assert_eq!(stalled.entries(), ["on_run_started", "on_run_dropped"]);
    assert_eq!(after.entries(), ["on_run_dropped"]);
}
