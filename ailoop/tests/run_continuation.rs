//! `ChatMiddleware::on_turn_end` continuation hook: a completion gate
//! that sends the model back to work stays inside one run — continuous
//! iterations, accumulated usage, a single `RunFinished`, and the
//! injected user message recorded in history.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ailoop::{
    AbortReason, ChatMiddleware, ChatRequest, ContinueDecision, Conversation, FinishReason,
    HookAction, Message, RunConfig, RunId, StepId, StreamChunk, ToolDefinition, ToolResultContent,
    Usage, UserBlock, advanced::run_chat,
};
use ailoop_core::testing::ScriptedModel;
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
use futures::StreamExt;
use serde_json::{Value, json};

const RETRY_PROMPT: &str = "tests are failing, fix them";

fn usage(input: u32, output: u32) -> Usage {
    let mut usage = Usage::default();
    usage.input_tokens = input;
    usage.output_tokens = output;
    usage
}

fn text_turn(text: &str, reason: FinishReason, usage: Usage) -> Vec<StreamChunk> {
    vec![
        StreamChunk::TextDelta { delta: text.into() },
        StreamChunk::TurnFinished {
            reason,
            usage,
            service_tier: None,
        },
    ]
}

fn tool_call_turn(id: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCallStarted {
            id: id.into(),
            name: "noop".into(),
        },
        StreamChunk::ToolCallFinished {
            id: id.into(),
            name: "noop".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ]
}

struct Noop;

#[async_trait::async_trait]
impl ToolDyn for Noop {
    fn name(&self) -> String {
        "noop".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new("noop", "does nothing", json!({"type": "object"}), vec![])
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        ToolResultContent::text("ok")
    }
}

/// Asks to continue the first `fail_times` turn ends, then stops.
/// Records how often it was asked and the reason it saw each time.
#[derive(Default)]
struct Gate {
    fail_times: usize,
    calls: AtomicUsize,
    reasons: Mutex<Vec<FinishReason>>,
}

impl Gate {
    fn failing(fail_times: usize) -> Arc<Self> {
        Arc::new(Self {
            fail_times,
            ..Self::default()
        })
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for Gate {
    async fn on_turn_end(
        &self,
        _run_id: &RunId,
        _step_id: &StepId,
        reason: &FinishReason,
        _new_messages: &[Message],
    ) -> ContinueDecision {
        self.reasons.lock().unwrap().push(reason.clone());
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n < self.fail_times {
            ContinueDecision::continue_with(RETRY_PROMPT)
        } else {
            ContinueDecision::Stop
        }
    }
}

/// Captures the messages of every request the model receives.
#[derive(Default)]
struct RequestLog(Mutex<Vec<Vec<Message>>>);

#[async_trait::async_trait]
impl ChatMiddleware for RequestLog {
    async fn on_chat_request(&self, _run_id: &RunId, _step_id: &StepId, req: &mut ChatRequest) {
        self.0.lock().unwrap().push(req.messages.clone());
    }
}

async fn run(
    model: &ScriptedModel,
    registry: &ToolRegistry,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
    max_iterations: usize,
) -> Vec<StreamChunk> {
    let mut config = RunConfig::default();
    config.middlewares = middlewares;
    config.max_iterations = max_iterations;
    run_chat(model, vec![Message::user("hi")], registry, config)
        .await
        .expect("run starts")
        .map(|c| c.expect("no error"))
        .collect()
        .await
}

fn step_iterations(chunks: &[StreamChunk]) -> Vec<usize> {
    chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::StepStarted { iteration, .. } => Some(*iteration),
            _ => None,
        })
        .collect()
}

fn run_finished(chunks: &[StreamChunk]) -> (&FinishReason, &Usage, &[Message]) {
    let finished: Vec<_> = chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::RunFinished {
                reason,
                usage,
                new_messages,
                ..
            } => Some((reason, usage, new_messages.as_slice())),
            _ => None,
        })
        .collect();
    assert_eq!(finished.len(), 1, "exactly one RunFinished");
    assert!(
        matches!(chunks.last(), Some(StreamChunk::RunFinished { .. })),
        "RunFinished is the last chunk"
    );
    finished[0]
}

fn is_injected(message: &Message) -> bool {
    matches!(
        message,
        Message::User { blocks, .. }
            if matches!(blocks.as_slice(), [UserBlock::Text { text, .. }] if text == RETRY_PROMPT)
    )
}

#[tokio::test]
async fn gate_failing_once_continues_within_the_same_run() {
    let model = ScriptedModel::new([
        text_turn("done", FinishReason::EndTurn, usage(10, 2)),
        text_turn("really done", FinishReason::EndTurn, usage(20, 3)),
    ]);
    let gate = Gate::failing(1);
    let requests = Arc::new(RequestLog::default());
    let chunks = run(
        &model,
        &ToolRegistry::new(),
        vec![gate.clone(), requests.clone()],
        10,
    )
    .await;

    assert_eq!(step_iterations(&chunks), vec![0, 1]);
    let (reason, usage, new_messages) = run_finished(&chunks);
    assert!(matches!(reason, FinishReason::EndTurn));
    assert_eq!((usage.input_tokens, usage.output_tokens), (30, 5));
    assert_eq!(new_messages.len(), 3);
    assert!(matches!(new_messages[0], Message::Assistant { .. }));
    assert!(is_injected(&new_messages[1]));
    assert!(matches!(new_messages[2], Message::Assistant { .. }));
    assert_eq!(gate.calls(), 2);

    // The injected message reaches the model on the continuation turn.
    let requests = requests.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].last().is_some_and(is_injected));

    // The step that asked to continue already reports the injected
    // message, like it reports its tool results.
    let first_step = chunks.iter().find_map(|c| match c {
        StreamChunk::StepFinished {
            iteration: 0,
            new_messages_so_far,
            ..
        } => Some(new_messages_so_far.clone()),
        _ => None,
    });
    assert!(first_step.unwrap().last().is_some_and(is_injected));
}

#[tokio::test]
async fn gate_continuation_is_committed_to_conversation_history() {
    let model = ScriptedModel::new([
        text_turn("done", FinishReason::EndTurn, usage(10, 2)),
        text_turn("really done", FinishReason::EndTurn, usage(20, 3)),
    ]);
    let mut chat = Conversation::builder(model)
        .middleware(Gate::failing(1))
        .build()
        .unwrap();

    let outcome = chat.run("hi").await.unwrap();

    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(outcome.final_text.as_deref(), Some("really done"));
    let history = chat.history_messages();
    assert_eq!(
        history.len(),
        4,
        "kickoff + assistant + injected + assistant"
    );
    assert!(is_injected(&history[2]));
}

#[tokio::test]
async fn gate_that_never_passes_stops_at_max_iterations() {
    let model =
        ScriptedModel::new((0..5).map(|_| text_turn("done", FinishReason::EndTurn, usage(1, 1))));
    let gate = Gate::failing(usize::MAX);
    let chunks = run(&model, &ToolRegistry::new(), vec![gate.clone()], 3).await;

    assert_eq!(step_iterations(&chunks), vec![0, 1, 2]);
    let (reason, usage, new_messages) = run_finished(&chunks);
    assert!(matches!(
        reason,
        FinishReason::Aborted(AbortReason::MaxIterations(3))
    ));
    assert_eq!(usage.input_tokens, 3);
    assert_eq!(gate.calls(), 3);
    // Every assistant turn is followed by the gate's message.
    assert_eq!(new_messages.len(), 6);
    assert!(new_messages.iter().skip(1).step_by(2).all(is_injected));
}

/// Implements no hook, so it relies on the `on_turn_end` default.
struct Passive;
impl ChatMiddleware for Passive {}

#[tokio::test]
async fn default_hook_leaves_the_run_unchanged() {
    for middlewares in [vec![], vec![Arc::new(Passive) as Arc<dyn ChatMiddleware>]] {
        let model = ScriptedModel::new([
            text_turn("done", FinishReason::EndTurn, usage(10, 2)),
            text_turn("unused", FinishReason::EndTurn, usage(10, 2)),
        ]);
        let chunks = run(&model, &ToolRegistry::new(), middlewares, 10).await;

        let kinds: Vec<_> = chunks
            .iter()
            .map(|c| match c {
                StreamChunk::RunStarted { .. } => "RunStarted",
                StreamChunk::StepStarted { .. } => "StepStarted",
                StreamChunk::TextDelta { .. } => "TextDelta",
                StreamChunk::StepFinished { .. } => "StepFinished",
                StreamChunk::RunFinished { .. } => "RunFinished",
                other => panic!("unexpected chunk {other:?}"),
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "RunStarted",
                "StepStarted",
                "TextDelta",
                "StepFinished",
                "RunFinished"
            ]
        );
        let (reason, _, new_messages) = run_finished(&chunks);
        assert!(matches!(reason, FinishReason::EndTurn));
        assert_eq!(new_messages.len(), 1);
    }
}

#[tokio::test]
async fn first_continue_wins_and_later_gates_are_skipped() {
    let model = ScriptedModel::new([
        text_turn("done", FinishReason::EndTurn, usage(1, 1)),
        text_turn("really done", FinishReason::EndTurn, usage(1, 1)),
    ]);
    let first = Gate::failing(1);
    let second = Gate::failing(0);
    let chunks = run(
        &model,
        &ToolRegistry::new(),
        vec![first.clone(), second.clone()],
        10,
    )
    .await;

    let (reason, _, new_messages) = run_finished(&chunks);
    assert!(matches!(reason, FinishReason::EndTurn));
    assert_eq!(new_messages.len(), 3);
    assert_eq!(first.calls(), 2);
    // Skipped on the first turn end, asked on the second.
    assert_eq!(second.calls(), 1);
}

#[tokio::test]
async fn gate_sees_every_natural_finish_reason_but_not_tool_use() {
    let model = ScriptedModel::new([
        tool_call_turn("t1"),
        text_turn("partial", FinishReason::MaxTokens, usage(1, 1)),
        text_turn("done", FinishReason::EndTurn, usage(1, 1)),
    ]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Noop)).unwrap();
    let gate = Gate::failing(1);
    let chunks = run(&model, &registry, vec![gate.clone()], 10).await;

    assert_eq!(step_iterations(&chunks), vec![0, 1, 2]);
    let reasons = gate.reasons.lock().unwrap();
    assert!(matches!(
        reasons.as_slice(),
        [FinishReason::MaxTokens, FinishReason::EndTurn]
    ));
}

/// Terminates every run before the first model call.
struct Terminate;

#[async_trait::async_trait]
impl ChatMiddleware for Terminate {
    async fn on_run_started(
        &self,
        _run_id: &RunId,
        _messages: &[Message],
        _config: &RunConfig,
    ) -> HookAction {
        HookAction::Terminate {
            reason: "no".into(),
        }
    }
}

#[tokio::test]
async fn gate_is_not_asked_on_aborted_runs() {
    let model = ScriptedModel::new([text_turn("done", FinishReason::EndTurn, usage(1, 1))]);
    let gate = Gate::failing(usize::MAX);
    let chunks = run(
        &model,
        &ToolRegistry::new(),
        vec![Arc::new(Terminate), gate.clone()],
        10,
    )
    .await;

    let (reason, _, _) = run_finished(&chunks);
    assert!(matches!(reason, FinishReason::Aborted(_)));
    assert_eq!(gate.calls(), 0);
}

/// A turn that stops for `MaxTokens` after completing a tool call: the
/// injected blocks join the tool results in one user message instead
/// of producing two user messages in a row.
#[tokio::test]
async fn injected_blocks_share_the_tool_result_message() {
    let mut turn = tool_call_turn("t1");
    turn.pop();
    turn.push(StreamChunk::TurnFinished {
        reason: FinishReason::MaxTokens,
        usage: Usage::default(),
        service_tier: None,
    });
    let model = ScriptedModel::new([turn, text_turn("done", FinishReason::EndTurn, usage(1, 1))]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Noop)).unwrap();
    let gate = Gate::failing(1);
    let chunks = run(&model, &registry, vec![gate], 10).await;

    let (_, _, new_messages) = run_finished(&chunks);
    assert_eq!(new_messages.len(), 3);
    match &new_messages[1] {
        Message::User { blocks, .. } => {
            assert!(matches!(blocks[0], UserBlock::ToolResult { .. }));
            assert!(matches!(&blocks[1], UserBlock::Text { text, .. } if text == RETRY_PROMPT));
        }
        other => panic!("expected a user message, got {other:?}"),
    }
}

/// Returns `Continue` with no blocks, which would otherwise continue
/// the run without a new user message.
struct EmptyContinue(AtomicUsize);

#[async_trait::async_trait]
impl ChatMiddleware for EmptyContinue {
    async fn on_turn_end(
        &self,
        _run_id: &RunId,
        _step_id: &StepId,
        _reason: &FinishReason,
        _new_messages: &[Message],
    ) -> ContinueDecision {
        self.0.fetch_add(1, Ordering::SeqCst);
        ContinueDecision::Continue { blocks: vec![] }
    }
}

/// An empty `Continue` counts as `Stop`: the run finishes instead of
/// sending a request that ends on the assistant's turn, and the next
/// middleware is still asked.
#[tokio::test]
async fn empty_continue_counts_as_stop() {
    let model = ScriptedModel::new([
        text_turn("done", FinishReason::EndTurn, usage(1, 1)),
        text_turn("unreachable", FinishReason::EndTurn, usage(1, 1)),
    ]);
    let registry = ToolRegistry::new();
    let empty = Arc::new(EmptyContinue(AtomicUsize::new(0)));
    let gate = Gate::failing(0);
    let chunks = run(&model, &registry, vec![empty.clone(), gate.clone()], 10).await;

    let (reason, _, new_messages) = run_finished(&chunks);
    assert!(matches!(reason, FinishReason::EndTurn));
    assert_eq!(step_iterations(&chunks), vec![0]);
    assert_eq!(new_messages.len(), 1);
    assert_eq!(empty.0.load(Ordering::SeqCst), 1);
    assert_eq!(gate.calls(), 1, "the next middleware is still asked");
}
