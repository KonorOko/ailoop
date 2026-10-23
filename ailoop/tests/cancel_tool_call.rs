//! `RunControl::cancel_tool`: one tool call is stopped and the run goes
//! on. What the model sees depends on when the cancel lands (before the
//! call's turn, during its `before` hooks, or while it runs), sibling
//! calls are untouched, and a run abort is still an abort.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ailoop::{
    AbortReason, CancellationToken, ChatMiddleware, FinishReason, Message, RunConfig, RunControl,
    StreamChunk, ToolCallInfo, ToolDecision, ToolDefinition, ToolResultContent, Usage,
    advanced::run_chat,
};
use ailoop_core::testing::ScriptedModel;
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::sync::Notify;

const NOT_RUN: &str = "Tool not run: cancelled by the caller";
const WHILE_RUNNING: &str =
    "Tool cancelled by the caller while running; it may have partly completed";

fn tool_calls_turn(calls: &[(&str, &str)]) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();
    for (id, name) in calls {
        chunks.push(StreamChunk::ToolCallStarted {
            call_id: (*id).into(),
            name: (*name).into(),
        });
        chunks.push(StreamChunk::ToolCallFinished {
            call_id: (*id).into(),
            name: (*name).into(),
            args: json!({}),
        });
    }
    chunks.push(StreamChunk::TurnFinished {
        reason: FinishReason::ToolUse,
        usage: Usage::default(),
        service_tier: None,
    });
    chunks
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

fn definition(name: &str) -> ToolDefinition {
    ToolDefinition::new(name, "test tool", json!({"type": "object"}), vec![])
}

/// Signals when it starts, keeps its token, then hangs without ever
/// looking at the token: only the engine dropping it can stop it.
#[derive(Default)]
struct Hang {
    started: Notify,
    token: Mutex<Option<CancellationToken>>,
}

#[async_trait::async_trait]
impl ToolDyn for Hang {
    fn name(&self) -> &str {
        "hang"
    }
    fn tool_definition(&self) -> ToolDefinition {
        definition("hang")
    }
    async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
        *self.token.lock().unwrap() = Some(ctx.cancellation().clone());
        self.started.notify_one();
        tokio::time::sleep(Duration::from_secs(60)).await;
        ToolResultContent::text("never")
    }
}

/// Counts its invocations, keeps its token, and answers at once; with
/// `cancel` set, cancels that call id first.
#[derive(Default)]
struct Quick {
    name: &'static str,
    safe: bool,
    calls: AtomicUsize,
    token: Mutex<Option<CancellationToken>>,
    cancel: Option<(RunControl, &'static str)>,
}

impl Quick {
    fn named(name: &'static str) -> Self {
        Self {
            name,
            ..Self::default()
        }
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl ToolDyn for Quick {
    fn name(&self) -> &str {
        self.name
    }
    fn tool_definition(&self) -> ToolDefinition {
        definition(self.name)
    }
    fn concurrency_safe(&self) -> bool {
        self.safe
    }
    async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.token.lock().unwrap() = Some(ctx.cancellation().clone());
        if let Some((control, id)) = &self.cancel {
            assert!(control.cancel_tool(id), "{id} should be cancellable");
        }
        ToolResultContent::text("ok")
    }
}

/// Records which call ids the tool hooks saw, and what the `after`
/// hook saw as the result. With `cancel_in_before`, cancels the call
/// from its own `before` hook, as if the user cancelled while an
/// approval was being asked.
#[derive(Default)]
struct Hooks {
    before: Mutex<Vec<String>>,
    after: Mutex<Vec<(String, ToolResultContent)>>,
    cancel_in_before: Option<RunControl>,
}

#[async_trait::async_trait]
impl ChatMiddleware for Hooks {
    async fn on_before_tool_call(&self, call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        self.before.lock().unwrap().push(call.call_id.clone());
        if let Some(control) = &self.cancel_in_before {
            control.cancel_tool(&call.call_id);
        }
        ToolDecision::Continue
    }
    async fn on_after_tool_call(
        &self,
        call: &ToolCallInfo,
        _args: &Value,
        result: &ToolResultContent,
    ) {
        self.after
            .lock()
            .unwrap()
            .push((call.call_id.clone(), result.clone()));
    }
}

fn config(control: &RunControl, middlewares: Vec<Arc<dyn ChatMiddleware>>) -> RunConfig {
    let mut config = RunConfig::default();
    config.control = Some(control.clone());
    config.middlewares = middlewares;
    config
}

async fn collect(
    model: &ScriptedModel,
    registry: &ToolRegistry,
    config: RunConfig,
) -> Vec<StreamChunk> {
    run_chat(model, vec![Message::user("hi")], registry, config)
        .await
        .expect("run starts")
        .map(|c| c.expect("no error"))
        .collect()
        .await
}

fn result_of<'a>(chunks: &'a [StreamChunk], call: &str) -> &'a ToolResultContent {
    chunks
        .iter()
        .find_map(|c| match c {
            StreamChunk::ToolResult {
                call_id, content, ..
            } if call_id == call => Some(content),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no ToolResult for {call}"))
}

fn finish_reason(chunks: &[StreamChunk]) -> FinishReason {
    match chunks.last() {
        Some(StreamChunk::RunFinished { reason, .. }) => reason.clone(),
        other => panic!("stream must end in RunFinished, got {other:?}"),
    }
}

fn assert_error(content: &ToolResultContent, text: &str) {
    assert!(
        content.is_error,
        "expected an error result, got {content:?}"
    );
    assert_eq!(content.as_text(), Some(text));
}

/// Cancelling a running call drops it (the tool never checks its
/// token), fires its token, and the run goes on to the next turn while
/// the sibling call runs normally with a token of its own.
#[tokio::test]
async fn cancel_while_running_stops_only_that_call() {
    let control = RunControl::new();
    let hang = Arc::new(Hang::default());
    let quick = Arc::new(Quick::named("quick"));
    let mut registry = ToolRegistry::new();
    registry.register(hang.clone()).unwrap();
    registry.register(quick.clone()).unwrap();
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "hang"), ("t2", "quick")]),
        text_turn("done"),
    ]);

    let canceller = {
        let (control, hang) = (control.clone(), hang.clone());
        tokio::spawn(async move {
            hang.started.notified().await;
            assert!(control.cancel_tool("t1"));
        })
    };
    let hooks = Arc::new(Hooks::default());
    let chunks = collect(&model, &registry, config(&control, vec![hooks.clone()])).await;
    canceller.await.unwrap();

    assert_error(result_of(&chunks, "t1"), WHILE_RUNNING);
    assert!(hang.token.lock().unwrap().as_ref().unwrap().is_cancelled());
    assert_eq!(result_of(&chunks, "t2").as_text(), Some("ok"));
    assert!(!quick.token.lock().unwrap().as_ref().unwrap().is_cancelled());
    let after = hooks.after.lock().unwrap();
    assert_eq!(after[0].0, "t1");
    assert_eq!(after[0].1.as_text(), Some(WHILE_RUNNING));
    assert!(matches!(finish_reason(&chunks), FinishReason::EndTurn));
}

/// A call cancelled before its turn comes is never invoked and no tool
/// hook fires for it; the call that cancelled it finishes normally.
#[tokio::test]
async fn cancel_before_the_calls_turn_skips_it_and_its_hooks() {
    let control = RunControl::new();
    let first = Arc::new(Quick {
        cancel: Some((control.clone(), "t2")),
        ..Quick::named("first")
    });
    let second = Arc::new(Quick::named("second"));
    let mut registry = ToolRegistry::new();
    registry.register(first.clone()).unwrap();
    registry.register(second.clone()).unwrap();
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "first"), ("t2", "second")]),
        text_turn("done"),
    ]);
    let hooks = Arc::new(Hooks::default());
    let chunks = collect(&model, &registry, config(&control, vec![hooks.clone()])).await;

    assert_eq!(result_of(&chunks, "t1").as_text(), Some("ok"));
    assert_error(result_of(&chunks, "t2"), NOT_RUN);
    assert_eq!(second.calls(), 0);
    assert_eq!(*hooks.before.lock().unwrap(), ["t1"]);
    assert_eq!(hooks.after.lock().unwrap().len(), 1);
    assert!(matches!(finish_reason(&chunks), FinishReason::EndTurn));
}

/// A cancel during the `before` hooks (an approval being asked) does
/// not run the tool, and the `after` hooks see the synthesized result.
#[tokio::test]
async fn cancel_during_before_hooks_skips_the_tool_but_pairs_hooks() {
    let control = RunControl::new();
    let quick = Arc::new(Quick::named("quick"));
    let mut registry = ToolRegistry::new();
    registry.register(quick.clone()).unwrap();
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "quick")]), text_turn("done")]);
    let hooks = Arc::new(Hooks {
        cancel_in_before: Some(control.clone()),
        ..Hooks::default()
    });
    let chunks = collect(&model, &registry, config(&control, vec![hooks.clone()])).await;

    assert_eq!(quick.calls(), 0);
    assert_error(result_of(&chunks, "t1"), NOT_RUN);
    let after = hooks.after.lock().unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].1.as_text(), Some(NOT_RUN));
    assert!(matches!(finish_reason(&chunks), FinishReason::EndTurn));
}

/// `cancel_tool` is a no-op for an unknown id, a call that already has
/// its result, and any call once the run is over.
#[tokio::test]
async fn cancel_tool_is_false_outside_the_step_in_flight() {
    let control = RunControl::new();
    // `second` tries to cancel `t1`, which already has its result.
    struct CancelFinished(RunControl, Mutex<Option<bool>>);
    #[async_trait::async_trait]
    impl ToolDyn for CancelFinished {
        fn name(&self) -> &str {
            "second"
        }
        fn tool_definition(&self) -> ToolDefinition {
            definition("second")
        }
        async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
            *self.1.lock().unwrap() =
                Some(self.0.cancel_tool("t1") || self.0.cancel_tool("unknown"));
            ToolResultContent::text("ok")
        }
    }
    let probe = Arc::new(CancelFinished(control.clone(), Mutex::new(None)));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Quick::named("first"))).unwrap();
    registry.register(probe.clone()).unwrap();
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "first"), ("t2", "second")]),
        text_turn("done"),
    ]);
    let chunks = collect(&model, &registry, config(&control, vec![])).await;

    assert_eq!(*probe.1.lock().unwrap(), Some(false));
    assert!(matches!(finish_reason(&chunks), FinishReason::EndTurn));
    assert!(!control.cancel_tool("t2"), "the run is over");
}

/// Cancelling the caller's token while a tool runs is still a run
/// abort, not a per-call cancel.
#[tokio::test]
async fn caller_cancellation_still_aborts_the_run() {
    let control = RunControl::new();
    let hang = Arc::new(Hang::default());
    let mut registry = ToolRegistry::new();
    registry.register(hang.clone()).unwrap();
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "hang")]), text_turn("done")]);
    let caller = CancellationToken::new();
    let mut config = config(&control, vec![]);
    config.cancellation = Some(caller.clone());

    let canceller = {
        let (caller, hang) = (caller.clone(), hang.clone());
        tokio::spawn(async move {
            hang.started.notified().await;
            caller.cancel();
        })
    };
    let chunks = collect(&model, &registry, config).await;
    canceller.await.unwrap();

    assert_eq!(
        finish_reason(&chunks),
        FinishReason::Aborted(AbortReason::Cancelled)
    );
    assert!(!control.cancel_tool("t1"));
}

/// Cancels `target` from the `before` hook of `on`.
struct CancelAtGate {
    control: RunControl,
    on: &'static str,
    target: &'static str,
}

#[async_trait::async_trait]
impl ChatMiddleware for CancelAtGate {
    async fn on_before_tool_call(&self, call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        if call.call_id == self.on {
            assert!(self.control.cancel_tool(self.target));
        }
        ToolDecision::Continue
    }
}

/// A call of a batch cancelled after its gates passed, but before the
/// batch started running, never ran: the model is told so rather than
/// that it may have partly completed.
#[tokio::test]
async fn a_batch_call_cancelled_before_it_starts_is_not_run() {
    let control = RunControl::new();
    let quick = Arc::new(Quick {
        safe: true,
        ..Quick::named("quick")
    });
    let mut registry = ToolRegistry::new();
    registry.register(quick.clone()).unwrap();
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "quick"), ("t2", "quick")]),
        text_turn("done"),
    ]);
    let gate = Arc::new(CancelAtGate {
        control: control.clone(),
        on: "t2",
        target: "t1",
    });
    let mut config = config(&control, vec![gate]);
    config.tool_concurrency = 2;
    let chunks = collect(&model, &registry, config).await;

    assert_error(result_of(&chunks, "t1"), NOT_RUN);
    assert_eq!(result_of(&chunks, "t2").as_text(), Some("ok"));
    assert_eq!(quick.calls(), 1);
}
