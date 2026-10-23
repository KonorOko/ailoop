//! `tool_concurrency`: consecutive calls to `concurrency_safe` tools run
//! together, up to the limit; any other call runs alone as a barrier.
//! Gates still run one call at a time in the model's order before the
//! batch executes, and `after` hooks, `ToolResult` chunks and history
//! follow the model's order once the batch is done.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ailoop::{
    AbortReason, ChatMiddleware, Conversation, FinishReason, Message, RunConfig, RunControl,
    RunOptions, StreamChunk, ToolCallInfo, ToolDecision, ToolDefinition, ToolResultContent, Usage,
    UserBlock, advanced::run_chat,
};
use ailoop_core::testing::ScriptedModel;
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
use futures::StreamExt;
use serde_json::{Value, json};

/// Shared record of what ran, for every tool and hook of a test.
#[derive(Default)]
struct Trace {
    events: Mutex<Vec<String>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl Trace {
    fn log(&self, event: String) {
        self.events.lock().unwrap().push(event);
    }
    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
    fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }
}

/// Answers with its call id after `delay`, logging start and end and
/// tracking how many calls are running at once. With `cancel` set,
/// cancels that call id once it starts.
struct Probe {
    name: &'static str,
    safe: bool,
    delay: Duration,
    trace: Arc<Trace>,
    cancel: Option<(RunControl, &'static str)>,
}

impl Probe {
    fn new(name: &'static str, safe: bool, delay: Duration, trace: &Arc<Trace>) -> Arc<Self> {
        Arc::new(Self {
            name,
            safe,
            delay,
            trace: trace.clone(),
            cancel: None,
        })
    }
}

#[async_trait::async_trait]
impl ToolDyn for Probe {
    fn name(&self) -> &str {
        self.name
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(self.name, "probe", json!({"type": "object"}), vec![])
    }
    fn concurrency_safe(&self) -> bool {
        self.safe
    }
    async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
        let id = ctx.call_id().to_string();
        self.trace.log(format!("start:{id}"));
        let now = self.trace.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.trace.max_in_flight.fetch_max(now, Ordering::SeqCst);
        if let Some((control, target)) = &self.cancel {
            assert!(control.cancel_tool(target));
        }
        tokio::time::sleep(self.delay).await;
        self.trace.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.trace.log(format!("end:{id}"));
        ToolResultContent::text(id)
    }
}

/// Logs the tool hooks into the shared trace; with `terminate_on` set,
/// ends the run at that call's gate.
struct Hooks {
    trace: Arc<Trace>,
    terminate_on: Option<&'static str>,
}

#[async_trait::async_trait]
impl ChatMiddleware for Hooks {
    async fn on_before_tool_call(&self, call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        self.trace.log(format!("before:{}", call.call_id));
        if self.terminate_on == Some(call.call_id.as_str()) {
            return ToolDecision::Terminate {
                reason: "policy".into(),
            };
        }
        ToolDecision::Continue
    }
    async fn on_after_tool_call(&self, call: &ToolCallInfo, _: &Value, _: &ToolResultContent) {
        self.trace.log(format!("after:{}", call.call_id));
    }
}

fn calls_turn(calls: &[(&str, &str)]) -> Vec<StreamChunk> {
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

fn done_turn() -> Vec<StreamChunk> {
    vec![
        StreamChunk::TextDelta {
            delta: "done".into(),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        },
    ]
}

async fn run(
    calls: &[(&str, &str)],
    tools: Vec<Arc<dyn ToolDyn>>,
    concurrency: usize,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
    control: Option<RunControl>,
) -> Vec<StreamChunk> {
    let mut registry = ToolRegistry::new();
    for tool in tools {
        registry.register(tool).unwrap();
    }
    let model = ScriptedModel::new([calls_turn(calls), done_turn()]);
    let mut config = RunConfig::default();
    config.tool_concurrency = concurrency;
    config.middlewares = middlewares;
    config.control = control;
    run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run starts")
        .map(|c| c.expect("no error"))
        .collect()
        .await
}

fn result_ids(chunks: &[StreamChunk]) -> Vec<String> {
    chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::ToolResult { call_id, .. } => Some(call_id.clone()),
            _ => None,
        })
        .collect()
}

fn result_text(chunks: &[StreamChunk], id: &str) -> Option<String> {
    chunks.iter().find_map(|c| match c {
        StreamChunk::ToolResult {
            call_id, content, ..
        } if call_id == id => content.as_text().map(str::to_string),
        _ => None,
    })
}

fn finish_reason(chunks: &[StreamChunk]) -> FinishReason {
    match chunks.last() {
        Some(StreamChunk::RunFinished { reason, .. }) => reason.clone(),
        other => panic!("stream must end in RunFinished, got {other:?}"),
    }
}

const SLOW: Duration = Duration::from_millis(40);

/// Three safe calls run together; results come back in the model's
/// order even though the first call is the slowest.
#[tokio::test]
async fn safe_calls_run_together_and_answer_in_model_order() {
    let trace = Arc::new(Trace::default());
    let slow = Probe::new("slow", true, SLOW * 2, &trace);
    let fast = Probe::new("fast", true, Duration::ZERO, &trace);
    let chunks = run(
        &[("t1", "slow"), ("t2", "fast"), ("t3", "fast")],
        vec![slow, fast],
        3,
        vec![],
        None,
    )
    .await;

    assert_eq!(trace.max_in_flight(), 3);
    assert_eq!(result_ids(&chunks), ["t1", "t2", "t3"]);
    assert_eq!(result_text(&chunks, "t2").as_deref(), Some("t2"));
    let Some(StreamChunk::RunFinished { new_messages, .. }) = chunks.last() else {
        panic!("no RunFinished");
    };
    let results: Vec<&str> = new_messages
        .iter()
        .flat_map(|m| match m {
            Message::User { blocks } => blocks.as_slice(),
            _ => &[],
        })
        .filter_map(|b| match b {
            UserBlock::ToolResult { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(results, ["t1", "t2", "t3"]);
}

/// No more than `tool_concurrency` calls run at once.
#[tokio::test]
async fn concurrency_limit_is_respected() {
    let trace = Arc::new(Trace::default());
    let probe = Probe::new("read", true, SLOW, &trace);
    run(
        &[
            ("t1", "read"),
            ("t2", "read"),
            ("t3", "read"),
            ("t4", "read"),
        ],
        vec![probe],
        2,
        vec![],
        None,
    )
    .await;
    assert_eq!(trace.max_in_flight(), 2);
}

/// A call to a tool that is not concurrency-safe runs alone, splitting
/// the safe calls around it into separate batches.
#[tokio::test]
async fn unsafe_call_is_a_barrier() {
    let trace = Arc::new(Trace::default());
    let read = Probe::new("read", true, SLOW, &trace);
    let write = Probe::new("write", false, SLOW, &trace);
    run(
        &[("t1", "read"), ("t2", "write"), ("t3", "read")],
        vec![read, write],
        3,
        vec![],
        None,
    )
    .await;
    assert_eq!(trace.max_in_flight(), 1);
    assert_eq!(
        trace
            .events()
            .into_iter()
            .filter(|e| e.starts_with("start") || e.starts_with("end"))
            .collect::<Vec<_>>(),
        [
            "start:t1", "end:t1", "start:t2", "end:t2", "start:t3", "end:t3"
        ]
    );
}

/// With the default of 1, safe tools still run one at a time.
#[tokio::test]
async fn default_concurrency_is_sequential() {
    let trace = Arc::new(Trace::default());
    let probe = Probe::new("read", true, SLOW, &trace);
    run(
        &[("t1", "read"), ("t2", "read")],
        vec![probe],
        1,
        vec![],
        None,
    )
    .await;
    assert_eq!(trace.max_in_flight(), 1);
}

/// Every gate of a batch runs, in order, before any call of it starts;
/// the `after` hooks follow in order once every call has ended.
#[tokio::test]
async fn gates_run_in_order_before_the_batch_and_after_hooks_follow_it() {
    let trace = Arc::new(Trace::default());
    let slow = Probe::new("slow", true, SLOW, &trace);
    let fast = Probe::new("fast", true, Duration::ZERO, &trace);
    let hooks = Arc::new(Hooks {
        trace: trace.clone(),
        terminate_on: None,
    });
    run(
        &[("t1", "slow"), ("t2", "fast")],
        vec![slow, fast],
        2,
        vec![hooks],
        None,
    )
    .await;

    let events = trace.events();
    let at = |e: &str| events.iter().position(|x| x == e).unwrap();
    assert_eq!(&events[..2], ["before:t1", "before:t2"]);
    assert!(at("end:t1") < at("after:t1") && at("end:t2") < at("after:t1"));
    assert!(at("after:t1") < at("after:t2"));
}

/// A `Terminate` at the second gate of a batch ends the run before any
/// call of the batch runs.
#[tokio::test]
async fn terminate_at_a_gate_runs_no_call_of_the_batch() {
    let trace = Arc::new(Trace::default());
    let probe = Probe::new("read", true, Duration::ZERO, &trace);
    let hooks = Arc::new(Hooks {
        trace: trace.clone(),
        terminate_on: Some("t2"),
    });
    let chunks = run(
        &[("t1", "read"), ("t2", "read")],
        vec![probe],
        2,
        vec![hooks],
        None,
    )
    .await;

    assert!(!trace.events().iter().any(|e| e.starts_with("start")));
    assert_eq!(result_ids(&chunks), ["t1", "t2"]);
    assert!(
        result_text(&chunks, "t1")
            .unwrap()
            .starts_with("Tool not run: the run was aborted")
    );
    assert!(matches!(
        finish_reason(&chunks),
        FinishReason::Aborted(AbortReason::ToolTerminated { .. })
    ));
}

/// `cancel_tool` inside a batch stops only the targeted call.
#[tokio::test]
async fn cancel_tool_in_a_batch_stops_only_that_call() {
    let trace = Arc::new(Trace::default());
    let control = RunControl::new();
    let hang = Probe::new("hang", true, Duration::from_secs(60), &trace);
    let canceller = Arc::new(Probe {
        name: "cancel",
        safe: true,
        delay: SLOW,
        trace: trace.clone(),
        cancel: Some((control.clone(), "t1")),
    });
    let chunks = run(
        &[("t1", "hang"), ("t2", "cancel")],
        vec![hang, canceller],
        2,
        vec![],
        Some(control),
    )
    .await;

    assert_eq!(
        result_text(&chunks, "t1").as_deref(),
        Some("Tool cancelled by the caller while running; it may have partly completed")
    );
    assert_eq!(result_text(&chunks, "t2").as_deref(), Some("t2"));
    assert!(matches!(finish_reason(&chunks), FinishReason::EndTurn));
}

/// `ConversationBuilder::tool_concurrency` reaches the engine.
#[tokio::test]
async fn conversation_builder_sets_tool_concurrency() {
    let trace = Arc::new(Trace::default());
    let mut chat = Conversation::builder(ScriptedModel::new([
        calls_turn(&[("t1", "read"), ("t2", "read")]),
        done_turn(),
    ]))
    .tool_dyn(Probe::new("read", true, SLOW, &trace))
    .tool_concurrency(2)
    .build()
    .expect("build");
    chat.run_with_options("hi", RunOptions::new())
        .await
        .expect("run");
    assert_eq!(trace.max_in_flight(), 2);
}

/// `t1` waits until `t3` has started.
struct WaitsForThird {
    started: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait]
impl ToolDyn for WaitsForThird {
    fn name(&self) -> &str {
        "read"
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new("read", "probe", json!({"type": "object"}), vec![])
    }
    fn concurrency_safe(&self) -> bool {
        true
    }
    async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
        match ctx.call_id() {
            "t1" => self.started.notified().await,
            "t3" => self.started.notify_one(),
            _ => {}
        }
        ToolResultContent::text(ctx.call_id())
    }
}

/// A finished call frees its slot for the next one even while an
/// earlier call of the batch is still running: with a limit of 2, `t3`
/// starts once `t2` is done, without waiting for `t1`.
#[tokio::test]
async fn a_slow_call_does_not_hold_back_the_rest_of_the_batch() {
    let tool = Arc::new(WaitsForThird {
        started: Arc::new(tokio::sync::Notify::new()),
    });
    let chunks = tokio::time::timeout(
        Duration::from_secs(5),
        run(
            &[("t1", "read"), ("t2", "read"), ("t3", "read")],
            vec![tool],
            2,
            vec![],
            None,
        ),
    )
    .await
    .expect("t3 should start while t1 waits for it");
    assert_eq!(result_ids(&chunks), ["t1", "t2", "t3"]);
}
