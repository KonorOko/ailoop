//! `ToolDecision::Defer`: a gate suspends the run mid-step instead of
//! waiting for an approval, the conversation keeps the suspension (in
//! its snapshot too), and `resume` finishes the step with a decision per
//! deferred call: approved calls run with the stored arguments, denied
//! ones are answered, waiting ones take their gates, and nothing that
//! already ran runs again.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ailoop::{
    CallState, ChatMiddleware, ChatRequest, Conversation, ConversationBuilder,
    ConversationSnapshot, EngineError, FinishReason, Message, ResumeDecision, ResumeError, RunId,
    RunOptions, RunStartInfo, StepInfo, StreamChunk, SubAgentTool, ToolCallInfo, ToolDecision,
    ToolDefinition, ToolResultContent, Usage, UserBlock,
};
use ailoop_core::testing::ScriptedModel;
use ailoop_tools::{ToolContext, ToolDyn};
use futures::StreamExt;
use serde_json::{Value, json};

/// Counts its calls and keeps the arguments of the last one.
struct Probe {
    name: &'static str,
    safe: bool,
    calls: AtomicUsize,
    last_args: Mutex<Option<Value>>,
}

impl Probe {
    fn new(name: &'static str) -> Arc<Self> {
        Self::with_safety(name, false)
    }
    fn with_safety(name: &'static str, safe: bool) -> Arc<Self> {
        Arc::new(Self {
            name,
            safe,
            calls: AtomicUsize::new(0),
            last_args: Mutex::new(None),
        })
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
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
    async fn call(&self, args: Value, ctx: &ToolContext) -> ToolResultContent {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.last_args.lock().unwrap() = Some(args);
        ToolResultContent::text(format!("{} ran", ctx.call_id()))
    }
}

/// Defers every call to a tool in `defer`, stamps `"seen": true` into
/// the arguments first (as a sanitizer would), and logs the tool hooks
/// and the runs it sees.
#[derive(Default)]
struct Gate {
    defer: HashSet<&'static str>,
    log: Mutex<Vec<String>>,
    resumed_from: Mutex<Vec<Option<RunId>>>,
    finished: AtomicUsize,
    requests: Mutex<Vec<Vec<Message>>>,
}

impl Gate {
    fn deferring(names: &[&'static str]) -> Arc<Self> {
        Arc::new(Self {
            defer: names.iter().copied().collect(),
            ..Self::default()
        })
    }
    fn log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for Gate {
    async fn on_run_started(&self, run: &RunStartInfo<'_>) -> ailoop::HookAction {
        self.resumed_from
            .lock()
            .unwrap()
            .push(run.config.resumed_from.clone());
        ailoop::HookAction::Continue
    }
    async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
        self.requests.lock().unwrap().push(req.messages.clone());
    }
    async fn on_before_tool_call_mut(&self, _call: &ToolCallInfo, args: &mut Value) {
        args["seen"] = json!(true);
    }
    async fn on_before_tool_call(&self, call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        self.log
            .lock()
            .unwrap()
            .push(format!("before:{}", call.call_id));
        if self.defer.contains(call.name.as_str()) {
            return ToolDecision::Defer {
                reason: format!("{} needs approval", call.name),
            };
        }
        ToolDecision::Continue
    }
    async fn on_after_tool_call(&self, call: &ToolCallInfo, _: &Value, _: &ToolResultContent) {
        self.log
            .lock()
            .unwrap()
            .push(format!("after:{}", call.call_id));
    }
    async fn on_run_finished(&self, _run: &ailoop::RunFinishedInfo<'_>) {
        self.finished.fetch_add(1, Ordering::SeqCst);
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
            args: json!({"target": *id}),
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

struct Setup {
    chat: Conversation<ScriptedModel>,
    gate: Arc<Gate>,
    read: Arc<Probe>,
    delete: Arc<Probe>,
}

fn setup(turns: Vec<Vec<StreamChunk>>, defer: &[&'static str], concurrency: usize) -> Setup {
    let gate = Gate::deferring(defer);
    let read = Probe::with_safety("read", concurrency > 1);
    let delete = Probe::with_safety("delete", concurrency > 1);
    let chat = Conversation::builder(ScriptedModel::new(turns))
        .tool_dyn(read.clone())
        .tool_dyn(delete.clone())
        .middleware(gate.clone())
        .tool_concurrency(concurrency)
        .build()
        .expect("build");
    Setup {
        chat,
        gate,
        read,
        delete,
    }
}

fn states(chat: &Conversation<ScriptedModel>) -> Vec<(String, &'static str)> {
    chat.suspension()
        .expect("suspended")
        .calls
        .iter()
        .map(|c| {
            let state = match c.state {
                CallState::Answered { .. } => "answered",
                CallState::Deferred { .. } => "deferred",
                CallState::Waiting => "waiting",
                _ => "other",
            };
            (c.call_id.clone(), state)
        })
        .collect()
}

fn tool_result_ids(message: &Message) -> Vec<String> {
    match message {
        Message::User { blocks } => blocks
            .iter()
            .filter_map(|b| match b {
                UserBlock::ToolResult { call_id, .. } => Some(call_id.clone()),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

fn result_text(history: &[Message], id: &str) -> Option<String> {
    history.iter().find_map(|m| match m {
        Message::User { blocks } => blocks.iter().find_map(|b| match b {
            UserBlock::ToolResult {
                call_id, content, ..
            } if call_id == id => content.as_text().map(str::to_string),
            _ => None,
        }),
        _ => None,
    })
}

/// Every `tool_use` in the history has its `tool_result` right after.
fn assert_no_orphans(history: &[Message]) {
    for (i, message) in history.iter().enumerate() {
        if let Message::Assistant { blocks } = message {
            let ids: Vec<String> = blocks
                .iter()
                .filter_map(|b| match b {
                    ailoop::AssistantBlock::ToolCall { call_id, .. } => Some(call_id.clone()),
                    _ => None,
                })
                .collect();
            if !ids.is_empty() {
                let next = history.get(i + 1).expect("tool calls need results");
                assert_eq!(tool_result_ids(next), ids);
            }
        }
    }
}

/// A deferred call suspends the run: the calls before it keep their
/// result, the ones after it wait with their gates not yet run, and the
/// conversation refuses a new run until it is resumed.
#[tokio::test]
async fn defer_suspends_the_run_mid_step() {
    let mut s = setup(
        vec![calls_turn(&[
            ("t1", "read"),
            ("t2", "delete"),
            ("t3", "read"),
        ])],
        &["delete"],
        1,
    );
    let mut stream = s.chat.stream("clean up").await.expect("starts");
    let mut kinds = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk.expect("no error") {
            StreamChunk::RunSuspended { .. } => kinds.push("suspended"),
            StreamChunk::RunFinished { reason, .. } => {
                assert_eq!(reason, FinishReason::Suspended);
                kinds.push("finished");
            }
            _ => {}
        }
    }
    drop(stream);

    assert_eq!(kinds, ["suspended", "finished"]);
    assert_eq!(
        states(&s.chat),
        [
            ("t1".into(), "answered"),
            ("t2".into(), "deferred"),
            ("t3".into(), "waiting")
        ]
    );
    let pending: Vec<_> = s.chat.suspension().unwrap().pending().collect();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].args, json!({"target": "t2", "seen": true}));
    assert_eq!((s.read.calls(), s.delete.calls()), (1, 0));
    assert_eq!(s.gate.log(), ["before:t1", "after:t1", "before:t2"]);
    assert_eq!(s.gate.finished.load(Ordering::SeqCst), 1);
    assert!(matches!(
        s.chat.messages().last(),
        Some(Message::Assistant { .. })
    ));

    let err = s.chat.run("anything else?").await.unwrap_err();
    assert!(matches!(err.kind(), EngineError::Suspended));
}

/// Approving runs the call with the stored arguments, without its
/// `before` hooks; the waiting call takes its gates; the call that
/// already ran does not run again; the model gets every result of the
/// step in one message, in order.
#[tokio::test]
async fn approve_runs_the_stored_call_and_finishes_the_step() {
    let mut s = setup(
        vec![
            calls_turn(&[("t1", "read"), ("t2", "delete"), ("t3", "read")]),
            text_turn("done"),
        ],
        &["delete"],
        1,
    );
    let first = s.chat.run("clean up").await.expect("runs");
    assert_eq!(first.finish_reason, FinishReason::Suspended);

    let outcome = s
        .chat
        .resume([("t2".into(), ResumeDecision::Approve)])
        .await
        .expect("resumes");

    assert_eq!(outcome.finish_reason, FinishReason::EndTurn);
    assert!(s.chat.suspension().is_none());
    assert_eq!((s.read.calls(), s.delete.calls()), (2, 1));
    assert_eq!(
        *s.delete.last_args.lock().unwrap(),
        Some(json!({"target": "t2", "seen": true}))
    );
    assert_eq!(
        s.gate.log(),
        [
            "before:t1",
            "after:t1",
            "before:t2",
            "after:t2",
            "before:t3",
            "after:t3"
        ]
    );
    let requests = s.gate.requests.lock().unwrap();
    assert_eq!(requests.len(), 2, "the resumed step makes no request");
    assert_eq!(
        tool_result_ids(requests[1].last().unwrap()),
        ["t1", "t2", "t3"]
    );
    assert_eq!(
        *s.gate.resumed_from.lock().unwrap(),
        [None, Some(first.run_id)]
    );
    assert_no_orphans(s.chat.messages());
}

/// Denying answers the call without running it; its `after` hooks
/// still fire.
#[tokio::test]
async fn deny_answers_the_call_without_running_it() {
    let mut s = setup(
        vec![calls_turn(&[("t1", "delete")]), text_turn("ok")],
        &["delete"],
        1,
    );
    s.chat.run("clean up").await.expect("runs");
    s.chat
        .resume([("t1".into(), ResumeDecision::deny("not today"))])
        .await
        .expect("resumes");

    assert_eq!(s.delete.calls(), 0);
    assert_eq!(
        result_text(s.chat.messages(), "t1").as_deref(),
        Some("Tool denied: not today")
    );
    assert_eq!(s.gate.log(), ["before:t1", "after:t1"]);
    assert_no_orphans(s.chat.messages());
}

/// A waiting call deferred in turn suspends the resumed run again; a
/// second resume finishes the step.
#[tokio::test]
async fn a_waiting_call_can_suspend_the_resumed_run() {
    let mut s = setup(
        vec![
            calls_turn(&[("t1", "delete"), ("t2", "delete")]),
            text_turn("ok"),
        ],
        &["delete"],
        1,
    );
    s.chat.run("clean up").await.expect("runs");
    let second = s
        .chat
        .resume([("t1".into(), ResumeDecision::Approve)])
        .await
        .expect("resumes");
    assert_eq!(second.finish_reason, FinishReason::Suspended);
    assert_eq!(
        states(&s.chat),
        [("t1".into(), "answered"), ("t2".into(), "deferred")]
    );

    s.chat
        .resume([("t2".into(), ResumeDecision::Approve)])
        .await
        .expect("resumes");
    assert_eq!(s.delete.calls(), 2);
    assert!(s.chat.suspension().is_none());
    assert_no_orphans(s.chat.messages());
}

/// In a batch of concurrency-safe calls, every deferral of the batch
/// lands in one suspension and the other calls of the batch run.
#[tokio::test]
async fn a_batch_can_defer_several_calls() {
    let mut s = setup(
        vec![calls_turn(&[
            ("t1", "read"),
            ("t2", "delete"),
            ("t3", "delete"),
        ])],
        &["delete"],
        3,
    );
    s.chat.run("clean up").await.expect("runs");
    let pending: Vec<_> = s
        .chat
        .suspension()
        .unwrap()
        .pending()
        .map(|c| c.call_id.clone())
        .collect();
    assert_eq!(pending, ["t2", "t3"]);
    assert_eq!(s.read.calls(), 1);
}

/// The suspension survives a snapshot through JSON and a new
/// conversation built from it, and resumes there.
#[tokio::test]
async fn suspension_survives_a_snapshot_round_trip() {
    let mut s = setup(
        vec![calls_turn(&[("t1", "read"), ("t2", "delete")])],
        &["delete"],
        1,
    );
    s.chat.run("clean up").await.expect("runs");
    let json = serde_json::to_string(&s.chat.snapshot()).unwrap();
    drop(s);

    let snapshot: ConversationSnapshot = serde_json::from_str(&json).unwrap();
    let gate = Gate::deferring(&["delete"]);
    let read = Probe::new("read");
    let delete = Probe::new("delete");
    let mut chat =
        ConversationBuilder::from_snapshot(ScriptedModel::new([text_turn("ok")]), snapshot)
            .tool_dyn(read.clone())
            .tool_dyn(delete.clone())
            .middleware(gate.clone())
            .build()
            .expect("build");
    assert!(chat.suspension().is_some());

    let outcome = chat
        .resume([("t2".into(), ResumeDecision::Approve)])
        .await
        .expect("resumes");
    assert_eq!(outcome.finish_reason, FinishReason::EndTurn);
    assert_eq!((read.calls(), delete.calls()), (0, 1));
    assert_eq!(
        result_text(chat.messages(), "t1").as_deref(),
        Some("t1 ran")
    );
    assert_no_orphans(chat.messages());
}

/// Decisions that do not match the deferred calls fail before anything
/// runs, and the suspension is kept.
#[tokio::test]
async fn mismatched_decisions_are_rejected() {
    let mut s = setup(vec![calls_turn(&[("t1", "delete")])], &["delete"], 1);
    s.chat.run("clean up").await.expect("runs");

    let missing = s.chat.resume([]).await.unwrap_err();
    assert!(matches!(
        missing.kind(),
        EngineError::Resume(ResumeError::MissingDecision(id)) if id == "t1"
    ));
    let extra = s
        .chat
        .resume([
            ("t1".into(), ResumeDecision::Approve),
            ("t9".into(), ResumeDecision::Approve),
        ])
        .await
        .unwrap_err();
    assert!(matches!(
        extra.kind(),
        EngineError::Resume(ResumeError::NotDeferred(id)) if id == "t9"
    ));
    assert!(s.chat.suspension().is_some());
    assert_eq!(s.delete.calls(), 0);
}

/// Abandoning closes the step without a model call: what has no result
/// gets "not run", the history is valid again, and the next run starts.
#[tokio::test]
async fn abandon_closes_the_step_and_unblocks_the_conversation() {
    let mut s = setup(
        vec![
            calls_turn(&[("t1", "read"), ("t2", "delete"), ("t3", "read")]),
            text_turn("fresh start"),
        ],
        &["delete"],
        1,
    );
    s.chat.run("clean up").await.expect("runs");
    s.chat.abandon_suspension("dismissed").expect("abandons");

    assert!(s.chat.suspension().is_none());
    assert_eq!(
        result_text(s.chat.messages(), "t1").as_deref(),
        Some("t1 ran")
    );
    assert_eq!(
        result_text(s.chat.messages(), "t2").as_deref(),
        Some("Tool not run: dismissed")
    );
    assert_eq!(
        result_text(s.chat.messages(), "t3").as_deref(),
        Some("Tool not run: dismissed")
    );
    assert_no_orphans(s.chat.messages());

    let next = s.chat.run("never mind").await.expect("runs");
    assert_eq!(next.final_text.as_deref(), Some("fresh start"));
    assert_eq!(
        s.chat.abandon_suspension("again").unwrap_err(),
        ResumeError::NotSuspended
    );
}

/// A dropped resume stream puts the suspension back.
#[tokio::test]
async fn dropped_resume_keeps_the_suspension() {
    let mut s = setup(
        vec![calls_turn(&[("t1", "delete")]), text_turn("ok")],
        &["delete"],
        1,
    );
    s.chat.run("clean up").await.expect("runs");
    let stream = s
        .chat
        .resume_stream([("t1".into(), ResumeDecision::Approve)], RunOptions::new())
        .await
        .expect("starts");
    drop(stream);
    assert!(s.chat.suspension().is_some());
}

/// A sub-agent whose run suspends reports an error to the parent and
/// stays usable for the next call.
#[tokio::test]
async fn suspended_sub_agent_is_abandoned_and_reports_an_error() {
    let child = Conversation::builder(ScriptedModel::new([
        calls_turn(&[("c1", "delete")]),
        text_turn("child ok"),
    ]))
    .tool_dyn(Probe::new("delete"))
    .middleware(Gate::deferring(&["delete"]))
    .build()
    .expect("child");
    let researcher = SubAgentTool::new("researcher", "delegate", child);

    let call = |id: &str| {
        vec![
            StreamChunk::ToolCallStarted {
                call_id: id.into(),
                name: "researcher".into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: id.into(),
                name: "researcher".into(),
                args: json!({"prompt": "go"}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
                service_tier: None,
            },
        ]
    };
    let mut parent = Conversation::builder(ScriptedModel::new([
        call("p1"),
        call("p2"),
        text_turn("parent done"),
    ]))
    .tool(researcher)
    .build()
    .expect("parent");
    parent.run("research").await.expect("runs");

    let first = result_text(parent.messages(), "p1").unwrap();
    assert!(first.contains("cannot defer"), "got {first}");
    assert_eq!(
        result_text(parent.messages(), "p2").as_deref(),
        Some("child ok")
    );
}

/// A resumed run that aborts before it gets to the suspended step (a
/// caller token already cancelled, a `Terminate` in `on_run_started`)
/// still closes that step: every call gets a result, so the history
/// has no `tool_use` left without one and the next run can go out.
#[tokio::test]
async fn abort_before_the_resumed_step_still_closes_it() {
    let mut s = setup(
        vec![
            calls_turn(&[("t1", "read"), ("t2", "delete"), ("t3", "read")]),
            text_turn("ok"),
        ],
        &["delete"],
        1,
    );
    s.chat.run("clean up").await.expect("runs");
    assert!(s.chat.suspension().is_some());

    let cancelled = tokio_util::sync::CancellationToken::new();
    cancelled.cancel();
    let outcome = s
        .chat
        .resume_with_options(
            [("t2".into(), ResumeDecision::Approve)],
            RunOptions::new().cancellation(cancelled),
        )
        .await
        .expect("an abort is not an error");

    assert!(matches!(outcome.finish_reason, FinishReason::Aborted(_)));
    assert!(s.chat.suspension().is_none());
    assert_eq!(s.delete.calls(), 0, "the approved call never ran");
    assert_no_orphans(s.chat.messages());
    assert_eq!(
        result_text(s.chat.messages(), "t1").as_deref(),
        Some("t1 ran")
    );
    assert!(
        result_text(s.chat.messages(), "t2")
            .unwrap()
            .starts_with("Tool not run")
    );
    assert_eq!(s.gate.finished.load(Ordering::SeqCst), 2);

    s.chat
        .run("what now?")
        .await
        .expect("the conversation goes on");
    assert_no_orphans(s.chat.messages());
}

/// A call with malformed arguments after the deferred one is answered
/// with its error in the suspension, not left waiting: resuming must
/// not run the tool with the empty arguments that stand in for the
/// ones the model failed to produce.
#[tokio::test]
async fn a_malformed_call_after_a_deferral_is_answered_not_run() {
    let mut turn = calls_turn(&[("t1", "delete")]);
    turn.insert(
        2,
        StreamChunk::ToolCallMalformed {
            call_id: "t2".into(),
            name: "read".into(),
            raw: "{not json".into(),
            error: "expected value".into(),
        },
    );
    let mut s = setup(vec![turn, text_turn("done")], &["delete"], 1);
    let mut stream = s.chat.stream("clean up").await.expect("starts");
    let mut results = Vec::new();
    while let Some(chunk) = stream.next().await {
        if let StreamChunk::ToolResult { call_id, .. } = chunk.expect("no error") {
            results.push(call_id);
        }
    }
    drop(stream);
    assert_eq!(
        results,
        ["t2"],
        "the rejection goes out as the run suspends"
    );
    assert_eq!(
        states(&s.chat),
        [("t1".into(), "deferred"), ("t2".into(), "answered")]
    );

    s.chat
        .resume([("t1".into(), ResumeDecision::Approve)])
        .await
        .expect("resumes");
    assert_eq!((s.delete.calls(), s.read.calls()), (1, 0));
    assert_no_orphans(s.chat.messages());
    assert!(
        result_text(s.chat.messages(), "t2")
            .unwrap()
            .contains("{not json")
    );
}

/// Cancels the run from inside a call's `after` hook and then hangs, so
/// the abort lands while the batch is in phase 3.
struct CancelInAfter {
    token: tokio_util::sync::CancellationToken,
}

#[async_trait::async_trait]
impl ChatMiddleware for CancelInAfter {
    async fn on_after_tool_call(&self, _: &ToolCallInfo, _: &Value, _: &ToolResultContent) {
        self.token.cancel();
        std::future::pending::<()>().await;
    }
}

/// An abort while a batch runs its `after` hooks still answers the
/// batch's deferred call, so the step has a result for every call.
#[tokio::test]
async fn abort_in_after_hooks_answers_a_deferred_call_of_the_batch() {
    let token = tokio_util::sync::CancellationToken::new();
    let gate = Gate::deferring(&["delete"]);
    let mut chat = Conversation::builder(ScriptedModel::new(vec![calls_turn(&[
        ("t1", "delete"),
        ("t2", "read"),
    ])]))
    .tool_dyn(Probe::with_safety("read", true))
    .tool_dyn(Probe::with_safety("delete", true))
    .middleware(gate)
    .middleware(Arc::new(CancelInAfter {
        token: token.clone(),
    }))
    .tool_concurrency(2)
    .build()
    .expect("build");

    let outcome = chat
        .run_with_options("clean up", RunOptions::new().cancellation(token))
        .await
        .expect("an abort is not an error");

    assert!(matches!(outcome.finish_reason, FinishReason::Aborted(_)));
    assert!(chat.suspension().is_none());
    assert_no_orphans(chat.messages());
    assert!(
        result_text(chat.messages(), "t1")
            .unwrap()
            .starts_with("Tool not run")
    );
}
