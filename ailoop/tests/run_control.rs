//! `RunControl` inbox: messages pushed from outside a run enter it
//! when a turn closes — after the turn's tool results, in the same user
//! message — keep the run going when the model was about to stop, and
//! are never lost: undelivered ones stay queued, and a dropped run puts
//! back what it took.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ailoop::{
    AbortReason, ChatMiddleware, ChatRequest, ContinueDecision, Conversation, ConversationBuilder,
    FinishReason, Inbound, Message, RunConfig, RunControl, RunOptions, StepId, StepInfo,
    StreamChunk, ToolDefinition, ToolResultContent, TurnEndInfo, Usage, UserBlock,
    advanced::run_chat,
};
use ailoop_core::testing::ScriptedModel;
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
use futures::StreamExt;
use serde_json::{Value, json};

const STEER: &str = "answer in English instead";

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

fn tool_call_turn(id: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCallStarted {
            call_id: id.into(),
            name: "push".into(),
        },
        StreamChunk::ToolCallFinished {
            call_id: id.into(),
            name: "push".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ]
}

/// A tool that, while it runs, has the user push a message: the case
/// of a correction typed while the agent works.
struct PushWhileRunning(RunControl);

#[async_trait::async_trait]
impl ToolDyn for PushWhileRunning {
    fn name(&self) -> &str {
        "push"
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new("push", "does nothing", json!({"type": "object"}), vec![])
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        self.0.push(Inbound::text("m1", STEER));
        ToolResultContent::text("ok")
    }
}

/// Pushes `message` the first time the model is asked for a turn, so
/// it is queued while that turn streams.
struct PushOnFirstRequest {
    control: RunControl,
    message: Mutex<Option<Inbound>>,
}

impl PushOnFirstRequest {
    fn new(control: &RunControl, message: Inbound) -> Arc<Self> {
        Arc::new(Self {
            control: control.clone(),
            message: Mutex::new(Some(message)),
        })
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for PushOnFirstRequest {
    async fn on_chat_request(&self, _step: &StepInfo, _req: &mut ChatRequest) {
        if let Some(message) = self.message.lock().unwrap().take() {
            self.control.push(message);
        }
    }
}

/// Counts `on_turn_end` calls and always lets the run stop; optionally
/// pushes a message while it runs.
#[derive(Default)]
struct Gate {
    calls: AtomicUsize,
    push_on_call: Option<RunControl>,
}

#[async_trait::async_trait]
impl ChatMiddleware for Gate {
    async fn on_turn_end(&self, _turn: &TurnEndInfo<'_>) -> ContinueDecision {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(control) = &self.push_on_call {
            control.push(Inbound::text("late", "too late for this run"));
        }
        ContinueDecision::Stop
    }
}

/// Captures the messages of every request the model receives.
#[derive(Default)]
struct RequestLog(Mutex<Vec<Vec<Message>>>);

#[async_trait::async_trait]
impl ChatMiddleware for RequestLog {
    async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
        self.0.lock().unwrap().push(req.messages.clone());
    }
}

async fn run(
    model: &ScriptedModel,
    registry: &ToolRegistry,
    control: &RunControl,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
    max_iterations: usize,
) -> Vec<StreamChunk> {
    let mut config = RunConfig::default();
    config.middlewares = middlewares;
    config.max_iterations = max_iterations;
    config.control = Some(control.clone());
    run_chat(model, vec![Message::user("hi")], registry, config)
        .await
        .expect("run starts")
        .map(|c| c.expect("no error"))
        .collect()
        .await
}

fn registry_with_pusher(control: &RunControl) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry
        .register(Arc::new(PushWhileRunning(control.clone())))
        .unwrap();
    registry
}

fn step_ids(chunks: &[StreamChunk]) -> Vec<StepId> {
    chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::StepStarted { step_id, .. } => Some(step_id.clone()),
            _ => None,
        })
        .collect()
}

fn deliveries(chunks: &[StreamChunk]) -> Vec<(StepId, Vec<String>)> {
    chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::InboundDelivered { step_id, ids, .. } => {
                Some((step_id.clone(), ids.clone()))
            }
            _ => None,
        })
        .collect()
}

fn finished(chunks: &[StreamChunk]) -> (FinishReason, Vec<Message>) {
    match chunks.last() {
        Some(StreamChunk::RunFinished {
            reason,
            new_messages,
            ..
        }) => (reason.clone(), new_messages.clone()),
        other => panic!("stream must end in RunFinished, got {other:?}"),
    }
}

fn user_blocks(message: &Message) -> &[UserBlock] {
    match message {
        Message::User { blocks } => blocks,
        other => panic!("expected a user message, got {other:?}"),
    }
}

fn is_text(block: &UserBlock, expected: &str) -> bool {
    matches!(block, UserBlock::Text { text, .. } if text == expected)
}

/// A message pushed while a tool runs enters when that turn closes:
/// behind the tool result, in the same user message, and the next
/// request carries it.
#[tokio::test]
async fn message_pushed_during_a_tool_follows_its_result() {
    let control = RunControl::new();
    let model = ScriptedModel::new([tool_call_turn("t1"), text_turn("done")]);
    let log = Arc::new(RequestLog::default());
    let chunks = run(
        &model,
        &registry_with_pusher(&control),
        &control,
        vec![log.clone()],
        10,
    )
    .await;

    let steps = step_ids(&chunks);
    assert_eq!(deliveries(&chunks), [(steps[0].clone(), vec!["m1".into()])]);

    let requests = log.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let blocks = user_blocks(requests[1].last().unwrap());
    assert_eq!(blocks.len(), 2);
    assert!(matches!(&blocks[0], UserBlock::ToolResult { call_id, .. } if call_id == "t1"));
    assert!(is_text(&blocks[1], STEER));

    let (reason, new_messages) = finished(&chunks);
    assert!(matches!(reason, FinishReason::EndTurn));
    assert!(new_messages.iter().any(
        |m| matches!(m, Message::User { blocks } if blocks.iter().any(|b| is_text(b, STEER)))
    ));
    assert!(control.is_empty());
}

/// The delivery chunk comes after the step's tool results and before
/// its `StepFinished`.
#[tokio::test]
async fn delivery_chunk_sits_between_tool_results_and_step_finished() {
    let control = RunControl::new();
    let model = ScriptedModel::new([tool_call_turn("t1"), text_turn("done")]);
    let chunks = run(
        &model,
        &registry_with_pusher(&control),
        &control,
        vec![],
        10,
    )
    .await;

    let position = |pred: &dyn Fn(&StreamChunk) -> bool| chunks.iter().position(pred).unwrap();
    let result = position(&|c| matches!(c, StreamChunk::ToolResult { .. }));
    let delivered = position(&|c| matches!(c, StreamChunk::InboundDelivered { .. }));
    let step_finished = position(&|c| matches!(c, StreamChunk::StepFinished { .. }));
    assert!(result < delivered && delivered < step_finished);
}

/// When the model was about to stop and something arrived, the run
/// continues without asking `on_turn_end`; the gate runs once the model
/// stops again with nothing queued.
#[tokio::test]
async fn delivery_keeps_the_run_going_and_skips_the_gate_that_turn() {
    let control = RunControl::new();
    let model = ScriptedModel::new([text_turn("first"), text_turn("second")]);
    let gate = Arc::new(Gate::default());
    let log = Arc::new(RequestLog::default());
    let chunks = run(
        &model,
        &ToolRegistry::new(),
        &control,
        vec![
            PushOnFirstRequest::new(&control, Inbound::text("m1", STEER)),
            gate.clone(),
            log.clone(),
        ],
        10,
    )
    .await;

    assert_eq!(step_ids(&chunks).len(), 2);
    assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
    let requests = log.0.lock().unwrap();
    let last = requests[1].last().unwrap();
    assert!(is_text(&user_blocks(last)[0], STEER));
    assert!(matches!(finished(&chunks).0, FinishReason::EndTurn));
}

/// The continuation a delivery causes counts against `max_iterations`;
/// at the cap the run aborts with the delivered message kept.
#[tokio::test]
async fn delivery_continuation_counts_against_max_iterations() {
    let control = RunControl::new();
    let model = ScriptedModel::new([text_turn("first"), text_turn("second")]);
    let chunks = run(
        &model,
        &ToolRegistry::new(),
        &control,
        vec![PushOnFirstRequest::new(
            &control,
            Inbound::text("m1", STEER),
        )],
        1,
    )
    .await;

    let (reason, new_messages) = finished(&chunks);
    assert_eq!(
        reason,
        FinishReason::Aborted(AbortReason::MaxIterations(1)),
        "the delivery continued past the cap"
    );
    assert!(is_text(
        &user_blocks(new_messages.last().unwrap())[0],
        STEER
    ));
}

/// A message with no blocks is reported but does not keep the run
/// going by itself: the gate still runs and the run stops.
#[tokio::test]
async fn empty_message_is_reported_without_continuing() {
    let control = RunControl::new();
    let model = ScriptedModel::new([text_turn("only")]);
    let gate = Arc::new(Gate::default());
    let chunks = run(
        &model,
        &ToolRegistry::new(),
        &control,
        vec![
            PushOnFirstRequest::new(&control, Inbound::new("empty", vec![])),
            gate.clone(),
        ],
        10,
    )
    .await;

    assert_eq!(step_ids(&chunks).len(), 1);
    assert_eq!(deliveries(&chunks)[0].1, ["empty"]);
    assert_eq!(gate.calls.load(Ordering::SeqCst), 1);
    assert!(matches!(finished(&chunks).0, FinishReason::EndTurn));
}

/// What arrives after the last drain (here, while the gate runs) stays
/// queued when the run ends.
#[tokio::test]
async fn message_arriving_during_the_gate_stays_queued() {
    let control = RunControl::new();
    let model = ScriptedModel::new([text_turn("only")]);
    let gate = Arc::new(Gate {
        push_on_call: Some(control.clone()),
        ..Gate::default()
    });
    let chunks = run(&model, &ToolRegistry::new(), &control, vec![gate], 10).await;

    assert!(deliveries(&chunks).is_empty());
    assert!(matches!(finished(&chunks).0, FinishReason::EndTurn));
    let pending = control.take_pending();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, "late");
}

/// Dropping the stream after a delivery puts the taken messages back
/// at the front of the queue, ahead of anything pushed since.
#[tokio::test]
async fn dropped_run_puts_taken_messages_back() {
    let control = RunControl::new();
    let model = ScriptedModel::new([tool_call_turn("t1"), text_turn("done")]);
    let registry = registry_with_pusher(&control);
    let mut config = RunConfig::default();
    config.control = Some(control.clone());
    let mut stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run starts");
    while let Some(chunk) = stream.next().await {
        if matches!(
            chunk.expect("no error"),
            StreamChunk::InboundDelivered { .. }
        ) {
            break;
        }
    }
    control.push(Inbound::text("m2", "and keep it short"));
    drop(stream);

    let ids: Vec<String> = control.take_pending().into_iter().map(|m| m.id).collect();
    assert_eq!(ids, ["m1", "m2"]);
}

/// Through `Conversation`, delivered messages are committed to history
/// like the rest of the run and survive a snapshot round trip.
#[tokio::test]
async fn conversation_commits_delivered_messages() {
    let control = RunControl::new();
    let turns = || ScriptedModel::new([tool_call_turn("t1"), text_turn("done")]);
    let mut chat = Conversation::builder(turns())
        .tool_dyn(Arc::new(PushWhileRunning(control.clone())))
        .build()
        .expect("build");
    let outcome = chat
        .run_with_options("hi", RunOptions::new().control(control.clone()))
        .await
        .expect("run");
    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));

    let has_steer = |history: &[Message]| {
        history.iter().any(
            |m| matches!(m, Message::User { blocks } if blocks.iter().any(|b| is_text(b, STEER))),
        )
    };
    assert!(has_steer(chat.messages()));
    let restored = ConversationBuilder::from_snapshot(turns(), chat.snapshot())
        .build()
        .expect("build");
    assert!(has_steer(restored.messages()));
}

/// Signals when `on_run_finished` starts, then never returns: a slow
/// closing hook the caller gives up on.
#[derive(Default)]
struct HangOnFinish {
    entered: tokio::sync::Notify,
}

#[async_trait::async_trait]
impl ChatMiddleware for HangOnFinish {
    async fn on_run_finished(&self, _run: &ailoop::RunFinishedInfo<'_>) {
        self.entered.notify_one();
        std::future::pending::<()>().await;
    }
}

/// A run dropped while its closing hooks run never handed its
/// `RunFinished` over: the conversation rolls it back, so the messages
/// it delivered go back in the queue instead of being lost.
#[tokio::test]
async fn drop_during_closing_hooks_puts_delivered_messages_back() {
    let control = RunControl::new();
    let hang = Arc::new(HangOnFinish::default());
    let mut chat = Conversation::builder(ScriptedModel::new([
        tool_call_turn("t1"),
        text_turn("done"),
    ]))
    .tool_dyn(Arc::new(PushWhileRunning(control.clone())))
    .middleware(hang.clone())
    .build()
    .expect("build");
    let mut stream = chat
        .stream_with_options("hi", RunOptions::new().control(control.clone()))
        .await
        .expect("starts");
    loop {
        tokio::select! {
            biased;
            _ = hang.entered.notified() => break,
            chunk = stream.next() => {
                let chunk = chunk.expect("the run reaches its closing hooks");
                assert!(!matches!(chunk, Ok(StreamChunk::RunFinished { .. })));
            }
        }
    }
    drop(stream);

    let has_steer = chat
        .messages()
        .iter()
        .any(|m| matches!(m, Message::User { blocks } if blocks.iter().any(|b| is_text(b, STEER))));
    assert!(!has_steer, "the run was rolled back");
    let ids: Vec<String> = control.take_pending().into_iter().map(|m| m.id).collect();
    assert_eq!(ids, ["m1"]);
}
