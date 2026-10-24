//! `RunControl::background_tool`: a running call to a backgroundable
//! tool is answered at once with a provisional result and keeps running
//! outside the run; its real result reaches the run while the run takes
//! it, and the `BackgroundConfig::on_result` callback always.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ailoop::{
    AbortReason, BackgroundConfig, BackgroundToolResult, CallState, CancellationToken,
    ChatMiddleware, Conversation, FinishReason, Inbound, Message, RunConfig, RunControl,
    RunOptions, Sanitize, StreamChunk, TimeoutTool, ToolCallInfo, ToolDecision, ToolDefinition,
    ToolProgress, ToolResultBlock, ToolResultContent, Usage, UserBlock, advanced::run_chat,
};
use ailoop_core::testing::ScriptedModel;
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tokio::time::Instant;

const WHILE_RUNNING: &str =
    "Tool cancelled by the caller while running; it may have partly completed";
const CANCELLED_IN_BACKGROUND: &str =
    "Background tool call cancelled by the caller; it may have partly completed";

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

/// Sleeps for `duration` without looking at its token (only the engine
/// or its task dropping it stops it early), then answers
/// `"<name> done"`.
struct Job {
    name: &'static str,
    background: bool,
    safe: bool,
    duration: Duration,
    /// Reports "before", waits for `release`, reports "after".
    progress: bool,
    release: Notify,
    panics: bool,
    started: Notify,
    token: Mutex<Option<CancellationToken>>,
    finished: AtomicBool,
}

impl Job {
    fn new(name: &'static str, secs: u64) -> Self {
        Self {
            name,
            background: true,
            safe: false,
            duration: Duration::from_secs(secs),
            progress: false,
            release: Notify::new(),
            panics: false,
            started: Notify::new(),
            token: Mutex::new(None),
            finished: AtomicBool::new(false),
        }
    }
    fn token(&self) -> CancellationToken {
        self.token.lock().unwrap().clone().expect("job started")
    }
    fn finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl ToolDyn for Job {
    fn name(&self) -> &str {
        self.name
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(self.name, "test job", json!({"type": "object"}), vec![])
    }
    fn concurrency_safe(&self) -> bool {
        self.safe
    }
    fn backgroundable(&self) -> bool {
        self.background
    }
    async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
        *self.token.lock().unwrap() = Some(ctx.cancellation().clone());
        self.started.notify_one();
        if self.progress {
            ctx.progress(ToolProgress::new().with_message("before"));
            self.release.notified().await;
            ctx.progress(ToolProgress::new().with_message("after"));
        } else {
            tokio::time::sleep(self.duration).await;
        }
        if self.panics {
            panic!("boom");
        }
        self.finished.store(true, Ordering::SeqCst);
        ToolResultContent::text(format!("{} done", self.name))
    }
}

/// Records what the read-only `after` hook saw, per call.
#[derive(Default)]
struct After(Mutex<Vec<(String, ToolResultContent)>>);

#[async_trait::async_trait]
impl ChatMiddleware for After {
    async fn on_after_tool_call(
        &self,
        call: &ToolCallInfo,
        _args: &Value,
        result: &ToolResultContent,
    ) {
        self.0
            .lock()
            .unwrap()
            .push((call.call_id.clone(), result.clone()));
    }
}

fn callback() -> (
    BackgroundConfig,
    mpsc::UnboundedReceiver<BackgroundToolResult>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let config = BackgroundConfig::new().on_result(move |result| {
        let _ = tx.send(result);
    });
    (config, rx)
}

fn config(control: &RunControl, background: BackgroundConfig) -> RunConfig {
    let mut config = RunConfig::default();
    config.control = Some(control.clone());
    config.background = background;
    config
}

fn tools<const N: usize>(jobs: [Arc<Job>; N]) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for job in jobs {
        registry.register(job).unwrap();
    }
    registry
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

/// Moves `call` once `job` has started, after `after`.
fn move_when_started(
    control: &RunControl,
    job: &Arc<Job>,
    call: &'static str,
    after: Duration,
) -> tokio::task::JoinHandle<Instant> {
    let (control, job) = (control.clone(), job.clone());
    tokio::spawn(async move {
        job.started.notified().await;
        tokio::time::sleep(after).await;
        assert!(control.background_tool(call), "{call} should be movable");
        Instant::now()
    })
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

fn position(chunks: &[StreamChunk], f: impl Fn(&StreamChunk) -> bool) -> Option<usize> {
    chunks.iter().position(f)
}

fn backgrounded(chunks: &[StreamChunk], call: &str) -> Option<usize> {
    position(
        chunks,
        |c| matches!(c, StreamChunk::ToolBackgrounded { call_id, .. } if call_id == call),
    )
}

fn tool_result_at(chunks: &[StreamChunk], call: &str) -> usize {
    position(
        chunks,
        |c| matches!(c, StreamChunk::ToolResult { call_id, .. } if call_id == call),
    )
    .unwrap_or_else(|| panic!("no ToolResult for {call}"))
}

fn finish(chunks: &[StreamChunk]) -> (FinishReason, Vec<Message>) {
    match chunks.last() {
        Some(StreamChunk::RunFinished {
            reason,
            new_messages,
            ..
        }) => (reason.clone(), new_messages.clone()),
        other => panic!("stream must end in RunFinished, got {other:?}"),
    }
}

fn notice(call: &str) -> String {
    BackgroundConfig::default_notice(&ToolCallInfo::new(
        Default::default(),
        Default::default(),
        call,
        "job",
    ))
}

/// The user blocks of the run's messages, flattened, as text: tool
/// results as `result <id>: <text>`, text blocks as they are.
fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::User { blocks } => Some(blocks),
            _ => None,
        })
        .flatten()
        .map(|block| match block {
            UserBlock::ToolResult {
                call_id, content, ..
            } => format!(
                "result {call_id}: {}",
                content.as_text().unwrap_or_default()
            ),
            UserBlock::Text { text, .. } => text.clone(),
            other => format!("{other:?}"),
        })
        .collect()
}

/// The proposal's headline case: a 30 s tool moved at 2 s frees the
/// step at once, the message pushed while it ran enters the same turn
/// close, the history keeps the provisional result, and the real one,
/// arriving after the run ended, goes to the callback.
#[tokio::test(start_paused = true)]
async fn moved_call_frees_the_step_and_delivers_the_inbox_in_the_same_turn() {
    for _ in 0..20 {
        let control = RunControl::new();
        let job = Arc::new(Job::new("job", 30));
        let registry = tools([job.clone()]);
        let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
        let (background, mut results) = callback();
        let start = Instant::now();
        let mover = {
            let (control, job) = (control.clone(), job.clone());
            tokio::spawn(async move {
                job.started.notified().await;
                tokio::time::sleep(Duration::from_secs(2)).await;
                control.push(Inbound::text("steer", "stop that"));
                assert!(control.background_tool("t1"));
                Instant::now()
            })
        };

        let chunks = collect(&model, &registry, config(&control, background)).await;
        let ended = Instant::now();
        let moved = mover.await.unwrap();
        assert_eq!(moved - start, Duration::from_secs(2));
        assert!(
            ended - moved < Duration::from_secs(1),
            "run waited for the tool"
        );

        // Provisional result, flagged first.
        let at = tool_result_at(&chunks, "t1");
        assert_eq!(backgrounded(&chunks, "t1"), Some(at - 1));
        assert_eq!(
            result_of(&chunks, "t1").as_text(),
            Some(notice("t1").as_str())
        );
        assert!(!result_of(&chunks, "t1").is_error);
        // The steer entered when that same step closed.
        let step_of_result = match &chunks[at] {
            StreamChunk::ToolResult { step_id, .. } => step_id.clone(),
            _ => unreachable!(),
        };
        assert!(chunks.iter().any(|c| matches!(
            c,
            StreamChunk::InboundDelivered { step_id, ids, .. }
                if *step_id == step_of_result && ids == &["steer"]
        )));
        let (reason, new_messages) = finish(&chunks);
        assert!(matches!(reason, FinishReason::EndTurn));
        assert_eq!(
            user_texts(&new_messages),
            [format!("result t1: {}", notice("t1")), "stop that".into()]
        );

        // The tool went on; its result reaches the callback only.
        assert!(!job.token().is_cancelled());
        let result = results.recv().await.expect("callback");
        assert_eq!(Instant::now() - start, Duration::from_secs(30));
        assert!(job.finished());
        assert_eq!(result.call_id, "t1");
        assert_eq!(result.name, "job");
        assert_eq!(result.content.as_text(), Some("job done"));
        assert!(!result.delivered && !result.cancelled);
        assert!(control.is_empty());
        assert!(results.try_recv().is_err(), "reported once");
    }
}

/// A result that arrives while the run is still going enters it as a
/// message with the call id, after `BackgroundToolFinished`, and the
/// callback says it was delivered.
#[tokio::test(start_paused = true)]
async fn real_result_enters_the_run_while_it_is_alive() {
    for _ in 0..20 {
        let control = RunControl::new();
        let job = Arc::new(Job::new("job", 5));
        let mut wait = Job::new("wait", 10);
        wait.background = false;
        let wait = Arc::new(wait);
        let registry = tools([job.clone(), wait.clone()]);
        let model = ScriptedModel::new([
            tool_calls_turn(&[("t1", "job")]),
            tool_calls_turn(&[("t2", "wait")]),
            text_turn("done"),
        ]);
        let (background, mut results) = callback();
        let mover = move_when_started(&control, &job, "t1", Duration::ZERO);

        let chunks = collect(&model, &registry, config(&control, background)).await;
        mover.await.unwrap();

        let result = results.try_recv().expect("reported during the run");
        assert!(result.delivered);
        assert_eq!(result.content.as_text(), Some("job done"));
        let finished = position(&chunks, |c| {
            matches!(c, StreamChunk::BackgroundToolFinished { call_id, content, .. }
                if call_id == "t1" && content.as_text() == Some("job done"))
        })
        .expect("BackgroundToolFinished");
        let delivered = position(
            &chunks,
            |c| matches!(c, StreamChunk::InboundDelivered { ids, .. } if ids == &["t1"]),
        )
        .expect("InboundDelivered");
        assert!(tool_result_at(&chunks, "t2") < finished && finished < delivered);
        let (_, new_messages) = finish(&chunks);
        assert_eq!(
            user_texts(&new_messages),
            [
                format!("result t1: {}", notice("t1")),
                "result t2: wait done".into(),
                "Result of background tool call `job` (call id t1):".into(),
                "job done".into(),
            ]
        );
        assert!(results.try_recv().is_err(), "reported once");
    }
}

/// The real result goes through the run's output transforms on both
/// paths, while the read-only `after` hooks see only the provisional
/// result.
#[tokio::test(start_paused = true)]
async fn real_result_goes_through_after_mut_hooks_on_both_paths() {
    let redact = || {
        Arc::new(Sanitize::new().on_tool_result(|_, result| {
            for block in &mut result.blocks {
                if let ToolResultBlock::Text { text } = block {
                    *text = text.replace("done", "[redacted]");
                }
            }
        }))
    };
    // Delivered to the run.
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 5));
    let mut wait = Job::new("wait", 10);
    wait.background = false;
    let registry = tools([job.clone(), Arc::new(wait)]);
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "job")]),
        tool_calls_turn(&[("t2", "wait")]),
        text_turn("done"),
    ]);
    let (background, mut results) = callback();
    let after = Arc::new(After::default());
    let mut cfg = config(&control, background);
    cfg.middlewares = vec![redact(), after.clone()];
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    let chunks = collect(&model, &registry, cfg).await;
    mover.await.unwrap();
    let result = results.try_recv().unwrap();
    assert!(result.delivered);
    assert_eq!(result.content.as_text(), Some("job [redacted]"));
    assert!(chunks.iter().any(|c| matches!(c,
        StreamChunk::BackgroundToolFinished { content, .. } if content.as_text() == Some("job [redacted]"))));
    let (_, new_messages) = finish(&chunks);
    assert!(user_texts(&new_messages).contains(&"job [redacted]".to_string()));
    let t1: Vec<_> = after
        .0
        .lock()
        .unwrap()
        .iter()
        .filter(|(id, _)| id == "t1")
        .cloned()
        .collect();
    assert_eq!(
        t1.len(),
        1,
        "after hooks run once, on the provisional result"
    );
    assert_eq!(t1[0].1.as_text(), Some(notice("t1").as_str()));

    // Reported to the callback only.
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 30));
    let registry = tools([job.clone()]);
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    let (background, mut results) = callback();
    let mut cfg = config(&control, background);
    cfg.middlewares = vec![redact()];
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    collect(&model, &registry, cfg).await;
    mover.await.unwrap();
    let result = results.recv().await.unwrap();
    assert!(!result.delivered);
    assert_eq!(result.content.as_text(), Some("job [redacted]"));
}

/// Cancelling the run does not reach a moved call; `cancel_background`
/// does, even though the tool never looks at its token.
#[tokio::test(start_paused = true)]
async fn ending_or_cancelling_the_run_leaves_the_moved_call_running() {
    for _ in 0..20 {
        let control = RunControl::new();
        let job = Arc::new(Job::new("job", 60));
        let mut wait = Job::new("wait", 10);
        wait.background = false;
        let wait = Arc::new(wait);
        let registry = tools([job.clone(), wait.clone()]);
        let model = ScriptedModel::new([
            tool_calls_turn(&[("t1", "job")]),
            tool_calls_turn(&[("t2", "wait")]),
            text_turn("done"),
        ]);
        let (background, mut results) = callback();
        let run_token = CancellationToken::new();
        let mut cfg = config(&control, background);
        cfg.cancellation = Some(run_token.clone());
        let mover = move_when_started(&control, &job, "t1", Duration::from_secs(1));
        let canceller = {
            let (wait, run_token) = (wait.clone(), run_token.clone());
            tokio::spawn(async move {
                wait.started.notified().await;
                run_token.cancel();
            })
        };

        let chunks = collect(&model, &registry, cfg).await;
        mover.await.unwrap();
        canceller.await.unwrap();
        assert!(matches!(
            finish(&chunks).0,
            FinishReason::Aborted(AbortReason::Cancelled)
        ));
        assert!(wait.token().is_cancelled());
        assert!(
            !job.token().is_cancelled(),
            "the run's abort reached the moved call"
        );
        assert!(!control.cancel_tool("t1"));

        tokio::time::sleep(Duration::from_secs(5)).await;
        assert!(control.cancel_background("t1"));
        let result = results.recv().await.unwrap();
        assert!(job.token().is_cancelled());
        assert!(!job.finished(), "the tool future was dropped");
        assert!(result.cancelled && !result.delivered);
        assert!(result.content.is_error);
        assert_eq!(result.content.as_text(), Some(CANCELLED_IN_BACKGROUND));
        assert!(!control.cancel_background("t1"), "already finished");
        assert!(!control.background_tool("t1"));
    }
}

/// A tool that does not opt in cannot be moved: the call is waited for
/// as before.
#[tokio::test(start_paused = true)]
async fn tool_without_the_flag_cannot_be_moved() {
    let control = RunControl::new();
    let mut job = Job::new("job", 30);
    job.background = false;
    let job = Arc::new(job);
    let registry = tools([job.clone()]);
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    let (background, mut results) = callback();
    let mover = {
        let (control, job) = (control.clone(), job.clone());
        tokio::spawn(async move {
            job.started.notified().await;
            control.background_tool("t1")
        })
    };
    let start = Instant::now();
    let chunks = collect(&model, &registry, config(&control, background)).await;
    assert!(!mover.await.unwrap());
    assert_eq!(Instant::now() - start, Duration::from_secs(30));
    assert_eq!(result_of(&chunks, "t1").as_text(), Some("job done"));
    assert_eq!(backgrounded(&chunks, "t1"), None);
    assert!(results.try_recv().is_err());
}

/// `background_tool` is `false` for an unknown call, a call that is not
/// running yet or already answered, and a call moved before.
#[tokio::test(start_paused = true)]
async fn background_tool_is_false_outside_a_running_call() {
    let control = RunControl::new();
    assert!(!control.background_tool("t1"));
    let job = Arc::new(Job::new("job", 30));
    let registry = tools([job.clone()]);
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    let mover = {
        let (control, job) = (control.clone(), job.clone());
        tokio::spawn(async move {
            job.started.notified().await;
            assert!(control.background_tool("t1"));
            assert!(!control.background_tool("t1"), "already moved");
            assert!(!control.cancel_tool("t1"), "moved calls are not the run's");
        })
    };
    collect(&model, &registry, config(&control, BackgroundConfig::new())).await;
    mover.await.unwrap();
    assert!(!control.background_tool("t1"));
}

/// Aborting the run keeps a moved call running and answered with its
/// provisional result; its unmoved sibling is cancelled as before.
#[tokio::test(start_paused = true)]
async fn abort_keeps_moved_calls_running_and_cancels_the_rest() {
    for _ in 0..20 {
        let control = RunControl::new();
        let mut a = Job::new("a", 30);
        a.safe = true;
        let mut b = Job::new("b", 30);
        b.safe = true;
        let (a, b) = (Arc::new(a), Arc::new(b));
        let registry = tools([a.clone(), b.clone()]);
        let model = ScriptedModel::new([
            tool_calls_turn(&[("t1", "a"), ("t2", "b")]),
            text_turn("ok"),
        ]);
        let (background, mut results) = callback();
        let mut cfg = config(&control, background);
        cfg.tool_concurrency = 2;
        cfg.timeout = Some(Duration::from_secs(5));
        let mover = move_when_started(&control, &a, "t1", Duration::from_secs(1));

        let chunks = collect(&model, &registry, cfg).await;
        mover.await.unwrap();
        assert!(matches!(
            finish(&chunks).0,
            FinishReason::Aborted(AbortReason::Timeout(_))
        ));
        assert_eq!(
            result_of(&chunks, "t1").as_text(),
            Some(notice("t1").as_str())
        );
        assert!(backgrounded(&chunks, "t1").is_some());
        assert!(result_of(&chunks, "t2").is_error);
        assert!(b.token().is_cancelled());
        assert!(!a.token().is_cancelled());
        let (_, new_messages) = finish(&chunks);
        assert!(user_texts(&new_messages)[0].starts_with("result t1: Moved to the background"));

        let result = results.recv().await.unwrap();
        assert_eq!(result.call_id, "t1");
        assert!(!result.delivered && a.finished());
        assert!(
            results.try_recv().is_err(),
            "the cancelled sibling is not reported"
        );
    }
}

/// In a parallel batch the moved call stops holding the batch: it ends
/// when its sibling does.
#[tokio::test(start_paused = true)]
async fn moved_call_stops_holding_its_batch() {
    for _ in 0..20 {
        let control = RunControl::new();
        let mut slow = Job::new("slow", 30);
        slow.safe = true;
        let mut quick = Job::new("quick", 3);
        quick.safe = true;
        quick.background = false;
        let slow = Arc::new(slow);
        let registry = tools([slow.clone(), Arc::new(quick)]);
        let model = ScriptedModel::new([
            tool_calls_turn(&[("t1", "slow"), ("t2", "quick")]),
            text_turn("ok"),
        ]);
        let mut cfg = config(&control, BackgroundConfig::new());
        cfg.tool_concurrency = 2;
        let mover = move_when_started(&control, &slow, "t1", Duration::from_secs(1));
        let start = Instant::now();
        let chunks = collect(&model, &registry, cfg).await;
        mover.await.unwrap();
        assert_eq!(Instant::now() - start, Duration::from_secs(3));
        assert_eq!(
            result_of(&chunks, "t1").as_text(),
            Some(notice("t1").as_str())
        );
        assert_eq!(result_of(&chunks, "t2").as_text(), Some("quick done"));
        assert!(
            tool_result_at(&chunks, "t1") < tool_result_at(&chunks, "t2"),
            "model's order"
        );
    }
}

/// Before it is moved, a backgroundable call is cancelled by
/// `cancel_tool` as any other call: its tool's token fires, and nothing
/// reaches the callback.
#[tokio::test(start_paused = true)]
async fn cancel_tool_before_the_move_works_as_before() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 30));
    let registry = tools([job.clone()]);
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    let (background, mut results) = callback();
    let canceller = {
        let (control, job) = (control.clone(), job.clone());
        tokio::spawn(async move {
            job.started.notified().await;
            assert!(control.cancel_tool("t1"));
        })
    };
    let chunks = collect(&model, &registry, config(&control, background)).await;
    canceller.await.unwrap();
    assert_eq!(result_of(&chunks, "t1").as_text(), Some(WHILE_RUNNING));
    assert!(job.token().is_cancelled());
    tokio::time::sleep(Duration::from_secs(60)).await;
    assert!(!job.finished());
    assert!(results.try_recv().is_err());
    assert!(!control.background_tool("t1") && !control.cancel_background("t1"));
}

/// The provisional result uses the configured notice and goes through
/// the `after` hooks like any result.
#[tokio::test(start_paused = true)]
async fn provisional_result_uses_the_notice_and_runs_after_hooks() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 30));
    let registry = tools([job.clone()]);
    let mut turn = tool_calls_turn(&[("t1", "job")]);
    if let StreamChunk::ToolCallFinished { args, .. } = &mut turn[1] {
        *args = json!({"target": "docs"});
    }
    let model = ScriptedModel::new([turn, text_turn("ok")]);
    let after = Arc::new(After::default());
    let mut cfg = config(
        &control,
        BackgroundConfig::new().notice(|call, args| {
            format!(
                "{} of {} keeps running",
                call.name,
                args["target"].as_str().unwrap()
            )
        }),
    );
    cfg.middlewares = vec![after.clone()];
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    let chunks = collect(&model, &registry, cfg).await;
    mover.await.unwrap();
    assert_eq!(
        result_of(&chunks, "t1").as_text(),
        Some("job of docs keeps running")
    );
    let seen = after.0.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].1.as_text(), Some("job of docs keeps running"));
}

/// A moved call in a step that suspends is answered in the suspension;
/// its real result goes to the callback, since the run ended.
#[tokio::test(start_paused = true)]
async fn moved_call_in_a_suspended_step_is_answered_and_reported_to_the_callback() {
    struct DeferGated;
    #[async_trait::async_trait]
    impl ChatMiddleware for DeferGated {
        async fn on_before_tool_call(&self, call: &ToolCallInfo, _: &Value) -> ToolDecision {
            if call.name == "gated" {
                ToolDecision::Defer {
                    reason: "approve".into(),
                }
            } else {
                ToolDecision::Continue
            }
        }
    }
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 30));
    let mut gated = Job::new("gated", 1);
    gated.background = false;
    let registry = tools([job.clone(), Arc::new(gated)]);
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job"), ("t2", "gated")])]);
    let (background, mut results) = callback();
    let mut cfg = config(&control, background);
    cfg.middlewares = vec![Arc::new(DeferGated)];
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    let chunks = collect(&model, &registry, cfg).await;
    mover.await.unwrap();

    assert!(matches!(finish(&chunks).0, FinishReason::Suspended));
    let suspension = chunks
        .iter()
        .find_map(|c| match c {
            StreamChunk::RunSuspended { suspension, .. } => Some(suspension),
            _ => None,
        })
        .expect("RunSuspended");
    match &suspension.calls[0].state {
        CallState::Answered { content } => {
            assert_eq!(content.as_text(), Some(notice("t1").as_str()))
        }
        other => panic!("expected an answered call, got {other:?}"),
    }
    let result = results.recv().await.unwrap();
    assert!(!result.delivered);
    assert_eq!(result.content.as_text(), Some("job done"));
}

/// Progress a moved call reports is dropped, even while its batch still
/// runs.
#[tokio::test(start_paused = true)]
async fn progress_after_the_move_is_dropped() {
    let control = RunControl::new();
    let mut job = Job::new("job", 0);
    job.safe = true;
    job.progress = true;
    let job = Arc::new(job);
    let mut wait = Job::new("wait", 10);
    wait.safe = true;
    wait.background = false;
    let registry = tools([job.clone(), Arc::new(wait)]);
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "job"), ("t2", "wait")]),
        text_turn("ok"),
    ]);
    let mut cfg = config(&control, BackgroundConfig::new());
    cfg.tool_concurrency = 2;
    let mover = {
        let (control, job) = (control.clone(), job.clone());
        tokio::spawn(async move {
            job.started.notified().await;
            tokio::time::sleep(Duration::from_secs(1)).await;
            assert!(control.background_tool("t1"));
            tokio::time::sleep(Duration::from_secs(1)).await;
            job.release.notify_one();
        })
    };
    let chunks = collect(&model, &registry, cfg).await;
    mover.await.unwrap();
    let messages: Vec<_> = chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::ToolProgress {
                call_id, progress, ..
            } if call_id == "t1" => progress.message.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(messages, ["before"]);
    assert!(job.finished());
}

/// A backgroundable call that panics before it is moved takes the run
/// down, as any tool's panic does.
#[tokio::test(start_paused = true)]
#[should_panic(expected = "boom")]
async fn panic_in_an_unmoved_background_call_reaches_the_run() {
    let control = RunControl::new();
    let mut job = Job::new("job", 1);
    job.panics = true;
    let registry = tools([Arc::new(job)]);
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    collect(&model, &registry, config(&control, BackgroundConfig::new())).await;
}

/// Once moved, a panic is the call's result, reported like any other.
#[tokio::test(start_paused = true)]
async fn panic_in_a_moved_call_is_reported_as_an_error() {
    let control = RunControl::new();
    let mut job = Job::new("job", 30);
    job.panics = true;
    let job = Arc::new(job);
    let registry = tools([job.clone()]);
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    let (background, mut results) = callback();
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    collect(&model, &registry, config(&control, background)).await;
    mover.await.unwrap();
    let result = results.recv().await.unwrap();
    assert!(result.content.is_error);
    assert_eq!(
        result.content.as_text(),
        Some("Tool panicked while running in the background")
    );
}

/// The run's end racing the real result: whichever wins, the result is
/// reported exactly once, and `delivered` matches whether it is in the
/// run's messages. Real time on a multi-threaded runtime, so the two
/// genuinely race.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn result_racing_the_run_end_is_delivered_exactly_once() {
    struct Quick(Notify, AtomicBool);
    #[async_trait::async_trait]
    impl ToolDyn for Quick {
        fn name(&self) -> &str {
            "job"
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new("job", "", json!({"type": "object"}), vec![])
        }
        fn backgroundable(&self) -> bool {
            true
        }
        async fn call(&self, _: Value, _: &ToolContext) -> ToolResultContent {
            self.1.store(true, Ordering::SeqCst);
            self.0.notified().await;
            ToolResultContent::text("raced")
        }
    }
    let (mut delivered, mut not_delivered) = (0, 0);
    for round in 0..300 {
        let control = RunControl::new();
        let quick = Arc::new(Quick(Notify::new(), AtomicBool::new(false)));
        let mut registry = ToolRegistry::new();
        registry.register(quick.clone()).unwrap();
        // A third turn for when the result lands in the last drain and
        // the run goes on to answer it.
        let model = ScriptedModel::new([
            tool_calls_turn(&[("t1", "job")]),
            text_turn("ok"),
            text_turn("ok again"),
        ]);
        let (background, mut results) = callback();
        let mover = {
            let (control, quick) = (control.clone(), quick.clone());
            tokio::spawn(async move {
                while !quick.1.load(Ordering::SeqCst) {
                    tokio::task::yield_now().await;
                }
                assert!(control.background_tool("t1"));
                // Let the run go on, then release the tool at a point
                // that varies with the round.
                for _ in 0..(round % 20) {
                    tokio::task::yield_now().await;
                }
                quick.0.notify_one();
            })
        };
        let chunks = collect(&model, &registry, config(&control, background)).await;
        mover.await.unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), results.recv())
            .await
            .expect("reported")
            .unwrap();
        let in_run = user_texts(&finish(&chunks).1).contains(&"raced".to_string());
        assert_eq!(result.delivered, in_run, "round {round}");
        if in_run {
            delivered += 1;
        } else {
            not_delivered += 1;
        }
        assert!(results.try_recv().is_err(), "round {round}: reported twice");
        assert!(control.is_empty(), "round {round}: left in the queue");
    }
    // Not asserted (timing-dependent), but useful when it is run by hand.
    eprintln!("delivered {delivered}, not delivered {not_delivered}");
}

/// A later run on the same handle may reuse the id of a call an earlier
/// run moved and that is still running: the two calls stay apart.
#[tokio::test(start_paused = true)]
async fn reused_call_id_does_not_mix_up_a_moved_call_and_a_new_one() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 5));
    let other = Arc::new(Job::new("other", 10));
    let registry = tools([job.clone(), other.clone()]);
    let (background, mut results) = callback();

    // The first run moves `t1` and ends while it runs.
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    let mover = move_when_started(&control, &job, "t1", Duration::from_secs(1));
    collect(&model, &registry, config(&control, background.clone())).await;
    mover.await.unwrap();

    // The second run calls another tool under the same id, and waits
    // for it past the end of the first call.
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "other")]), text_turn("ok")]);
    let chunks = collect(&model, &registry, config(&control, background)).await;
    assert_eq!(result_of(&chunks, "t1").as_text(), Some("other done"));
    assert_eq!(backgrounded(&chunks, "t1"), None);

    let result = results.recv().await.unwrap();
    assert_eq!(result.name, "job");
    assert_eq!(result.content.as_text(), Some("job done"));
    assert!(!result.delivered);
    assert!(results.try_recv().is_err());
}

/// `cancel_tool` and `background_tool` on the same running call: the
/// first one wins and the other returns `false`, so a call reported
/// cancelled is never left running in the background.
#[tokio::test(start_paused = true)]
async fn cancel_tool_and_background_tool_exclude_each_other() {
    for cancel_first in [true, false] {
        let control = RunControl::new();
        let job = Arc::new(Job::new("job", 30));
        let registry = tools([job.clone()]);
        let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
        let (background, mut results) = callback();
        let racer = {
            let (control, job) = (control.clone(), job.clone());
            tokio::spawn(async move {
                job.started.notified().await;
                // Both before the engine looks at the call again.
                if cancel_first {
                    (control.cancel_tool("t1"), control.background_tool("t1"))
                } else {
                    let moved = control.background_tool("t1");
                    (control.cancel_tool("t1"), moved)
                }
            })
        };
        let chunks = collect(&model, &registry, config(&control, background)).await;
        let (cancelled, moved) = racer.await.unwrap();
        assert_eq!((cancelled, moved), (cancel_first, !cancel_first));
        tokio::time::sleep(Duration::from_secs(60)).await;
        if cancel_first {
            assert_eq!(result_of(&chunks, "t1").as_text(), Some(WHILE_RUNNING));
            assert!(!job.finished());
            assert!(results.try_recv().is_err());
        } else {
            assert_eq!(
                result_of(&chunks, "t1").as_text(),
                Some(notice("t1").as_str())
            );
            assert!(job.finished());
            assert_eq!(
                results.try_recv().unwrap().content.as_text(),
                Some("job done")
            );
        }
    }
}

/// Runs until `stop` matches a chunk, then drops the stream.
async fn run_until(
    model: &ScriptedModel,
    registry: &ToolRegistry,
    config: RunConfig,
    stop: impl Fn(&StreamChunk) -> bool,
) {
    let mut stream = run_chat(model, vec![Message::user("hi")], registry, config)
        .await
        .expect("run starts");
    while let Some(chunk) = stream.next().await {
        if stop(&chunk.expect("no error")) {
            return;
        }
    }
    panic!("the run ended before the chunk to stop at");
}

/// A run dropped after it took a background result is rolled back, so
/// the result is not delivered: the callback says so, once, and the
/// result is not put back in the queue for a later run to deliver.
#[tokio::test(start_paused = true)]
async fn dropped_run_reports_a_taken_result_as_not_delivered() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 2));
    let mut wait = Job::new("wait", 10);
    wait.background = false;
    let registry = tools([job.clone(), Arc::new(wait)]);
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "job")]),
        tool_calls_turn(&[("t2", "wait")]),
        text_turn("done"),
    ]);
    let (background, mut results) = callback();
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    control.push(Inbound::text("m", "queued"));
    run_until(
        &model,
        &registry,
        config(&control, background),
        |c| matches!(c, StreamChunk::BackgroundToolFinished { call_id, .. } if call_id == "t1"),
    )
    .await;
    mover.await.unwrap();

    let result = results.try_recv().expect("reported when the run dropped");
    assert_eq!(result.content.as_text(), Some("job done"));
    assert!(!result.delivered);
    assert!(results.try_recv().is_err());
    let pending = control.take_pending();
    assert_eq!(
        pending.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        ["m"],
        "queued messages go back, the background result does not"
    );
}

/// A run whose stream is dropped while a call it moved still runs
/// stops taking results: the real result goes to the callback, as not
/// delivered.
#[tokio::test(start_paused = true)]
async fn dropped_run_leaves_a_later_result_to_the_callback() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 5));
    let mut wait = Job::new("wait", 10);
    wait.background = false;
    let registry = tools([job.clone(), Arc::new(wait)]);
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "job")]),
        tool_calls_turn(&[("t2", "wait")]),
        text_turn("done"),
    ]);
    let (background, mut results) = callback();
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    run_until(
        &model,
        &registry,
        config(&control, background),
        |c| matches!(c, StreamChunk::ToolCallStarted { call_id, .. } if call_id == "t2"),
    )
    .await;
    mover.await.unwrap();
    assert!(!job.finished());

    let result = tokio::time::timeout(Duration::from_secs(60), results.recv())
        .await
        .expect("reported to the callback")
        .unwrap();
    assert_eq!(result.content.as_text(), Some("job done"));
    assert!(!result.delivered);
    assert!(results.try_recv().is_err());
    assert!(control.take_pending().is_empty());
}

/// A moved call cancelled while its run is alive: the run takes the
/// cancellation as the call's result, and both the chunk and the
/// message say so.
#[tokio::test(start_paused = true)]
async fn moved_call_cancelled_while_the_run_lives_is_delivered_as_cancelled() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 60));
    let mut wait = Job::new("wait", 10);
    wait.background = false;
    let wait = Arc::new(wait);
    let registry = tools([job.clone(), wait.clone()]);
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "job")]),
        tool_calls_turn(&[("t2", "wait")]),
        text_turn("done"),
    ]);
    let (background, mut results) = callback();
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    let canceller = {
        let (control, wait) = (control.clone(), wait.clone());
        tokio::spawn(async move {
            wait.started.notified().await;
            assert!(control.cancel_background("t1"));
        })
    };
    let chunks = collect(&model, &registry, config(&control, background)).await;
    mover.await.unwrap();
    canceller.await.unwrap();

    let finished = chunks
        .iter()
        .find_map(|c| match c {
            StreamChunk::BackgroundToolFinished {
                call_id,
                name,
                content,
                cancelled,
                ..
            } if call_id == "t1" => Some((name.clone(), content.clone(), *cancelled)),
            _ => None,
        })
        .expect("BackgroundToolFinished");
    assert_eq!(finished.0, "job");
    assert!(finished.1.is_error && finished.2);
    assert_eq!(finished.1.as_text(), Some(CANCELLED_IN_BACKGROUND));
    let (_, new_messages) = finish(&chunks);
    let texts = user_texts(&new_messages);
    assert_eq!(
        texts[texts.len() - 2..],
        [
            "Background tool call `job` (call id t1) was cancelled:".to_string(),
            CANCELLED_IN_BACKGROUND.to_string(),
        ]
    );
    let result = results.try_recv().unwrap();
    assert!(result.cancelled && result.delivered);
}

/// A background result the run takes goes ahead of the messages queued
/// for the same turn close, even those queued before it arrived.
#[tokio::test(start_paused = true)]
async fn background_results_go_ahead_of_queued_messages() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 5));
    let mut wait = Job::new("wait", 10);
    wait.background = false;
    let wait = Arc::new(wait);
    let registry = tools([job.clone(), wait.clone()]);
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "job")]),
        tool_calls_turn(&[("t2", "wait")]),
        text_turn("done"),
    ]);
    let (background, _results) = callback();
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    let pusher = {
        let (control, wait) = (control.clone(), wait.clone());
        tokio::spawn(async move {
            wait.started.notified().await;
            // Before `t1` finishes.
            tokio::time::sleep(Duration::from_secs(1)).await;
            control.push(Inbound::text("m", "queued first"));
        })
    };
    let chunks = collect(&model, &registry, config(&control, background)).await;
    mover.await.unwrap();
    pusher.await.unwrap();

    assert!(chunks.iter().any(|c| matches!(
        c,
        StreamChunk::InboundDelivered { ids, .. } if ids == &["t1", "m"]
    )));
    let (_, new_messages) = finish(&chunks);
    let texts = user_texts(&new_messages);
    assert_eq!(
        texts[texts.len() - 3..],
        [
            "Result of background tool call `job` (call id t1):".to_string(),
            "job done".into(),
            "queued first".into(),
        ]
    );
}

/// Through `Conversation`, `RunOptions::background` sets the notice
/// and the callback of the run.
#[tokio::test(start_paused = true)]
async fn conversation_takes_the_background_config_from_run_options() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 30));
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    let mut chat = Conversation::builder(model)
        .tool_dyn(job.clone())
        .build()
        .expect("build");
    let (background, mut results) = callback();
    let background = background.notice(|call, _| format!("moved {}", call.call_id));
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    let outcome = chat
        .run_with_options(
            "hi",
            RunOptions::new()
                .control(control.clone())
                .background(background),
        )
        .await
        .expect("run");
    mover.await.unwrap();

    assert!(user_texts(&outcome.new_messages).contains(&"result t1: moved t1".to_string()));
    let result = results.recv().await.unwrap();
    assert_eq!(result.content.as_text(), Some("job done"));
    assert!(!result.delivered);
}

/// Reports usage when it starts and again when it ends, `secs` later.
struct Spender {
    secs: u64,
    started: Notify,
}

#[async_trait::async_trait]
impl ToolDyn for Spender {
    fn name(&self) -> &str {
        "spend"
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new("spend", "test spender", json!({"type": "object"}), vec![])
    }
    fn backgroundable(&self) -> bool {
        true
    }
    async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
        ctx.report_usage(Usage::new(10, 1));
        self.started.notify_one();
        tokio::time::sleep(Duration::from_secs(self.secs)).await;
        ctx.report_usage(Usage::new(100, 10));
        ToolResultContent::text("spent")
    }
}

/// A moved call's usage counts towards its run while the run lasts:
/// what it reports after the run finished is not in the run's usage.
#[tokio::test(start_paused = true)]
async fn moved_call_usage_counts_until_the_run_finishes() {
    for (secs, expected) in [(2, Usage::new(110, 11)), (30, Usage::new(10, 1))] {
        let control = RunControl::new();
        let spender = Arc::new(Spender {
            secs,
            started: Notify::new(),
        });
        let mut wait = Job::new("wait", 10);
        wait.background = false;
        let mut registry = tools([Arc::new(wait)]);
        registry.register(spender.clone()).unwrap();
        let model = ScriptedModel::new([
            tool_calls_turn(&[("t1", "spend")]),
            tool_calls_turn(&[("t2", "wait")]),
            text_turn("done"),
        ]);
        let (background, mut results) = callback();
        let mover = {
            let (control, spender) = (control.clone(), spender.clone());
            tokio::spawn(async move {
                spender.started.notified().await;
                assert!(control.background_tool("t1"));
            })
        };
        let chunks = collect(&model, &registry, config(&control, background)).await;
        mover.await.unwrap();
        let usage = match chunks.last() {
            Some(StreamChunk::RunFinished { usage, .. }) => *usage,
            other => panic!("expected RunFinished, got {other:?}"),
        };
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (expected.input_tokens, expected.output_tokens),
            "tool running {secs} s"
        );
        assert_eq!(
            results.recv().await.unwrap().content.as_text(),
            Some("spent")
        );
    }
}

/// Shares a [`Job`] with the test while another wrapper owns the tool.
struct Shared(Arc<Job>);

#[async_trait::async_trait]
impl ToolDyn for Shared {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn tool_definition(&self) -> ToolDefinition {
        self.0.tool_definition()
    }
    fn backgroundable(&self) -> bool {
        self.0.backgroundable()
    }
    async fn call(&self, args: Value, ctx: &ToolContext) -> ToolResultContent {
        self.0.call(args, ctx).await
    }
}

/// A `TimeoutTool` can be moved, and its timeout still stops the call
/// once it is in the background.
#[tokio::test(start_paused = true)]
async fn timeout_still_applies_to_a_moved_call() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 60));
    let mut registry = ToolRegistry::new();
    registry
        .register(Arc::new(TimeoutTool::new(
            Shared(job.clone()),
            Duration::from_secs(5),
        )))
        .unwrap();
    let model = ScriptedModel::new([tool_calls_turn(&[("t1", "job")]), text_turn("ok")]);
    let (background, mut results) = callback();
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    let start = Instant::now();
    let chunks = collect(&model, &registry, config(&control, background)).await;
    mover.await.unwrap();
    assert_eq!(
        result_of(&chunks, "t1").as_text(),
        Some(notice("t1").as_str())
    );

    let result = results.recv().await.unwrap();
    assert_eq!(Instant::now() - start, Duration::from_secs(5));
    assert!(result.content.is_error && !result.cancelled && !result.delivered);
    assert_eq!(
        result.content.as_text(),
        Some("Tool 'job' timed out after 5s")
    );
    assert!(!job.finished());
}

/// Asks for one more turn the first time a turn would end the run.
#[derive(Default)]
struct ContinueOnce(AtomicBool);

#[async_trait::async_trait]
impl ChatMiddleware for ContinueOnce {
    async fn on_turn_end(&self, _turn: &ailoop::TurnEndInfo<'_>) -> ailoop::ContinueDecision {
        if self.0.swap(true, Ordering::SeqCst) {
            ailoop::ContinueDecision::Stop
        } else {
            ailoop::ContinueDecision::Continue {
                blocks: vec![UserBlock::text("go on")],
            }
        }
    }
}

/// A turn that would end the run and finds nothing to deliver does not
/// stop the run taking background results when `on_turn_end` keeps it
/// going: a result arriving in a later turn enters the run.
#[tokio::test(start_paused = true)]
async fn run_kept_going_by_on_turn_end_still_takes_background_results() {
    let control = RunControl::new();
    let job = Arc::new(Job::new("job", 5));
    let mut wait = Job::new("wait", 10);
    wait.background = false;
    let registry = tools([job.clone(), Arc::new(wait)]);
    let model = ScriptedModel::new([
        tool_calls_turn(&[("t1", "job")]),
        // Would end the run, with `t1` still running.
        text_turn("waiting"),
        tool_calls_turn(&[("t2", "wait")]),
        text_turn("done"),
    ]);
    let (background, mut results) = callback();
    let mut cfg = config(&control, background);
    cfg.middlewares = vec![Arc::new(ContinueOnce::default())];
    let mover = move_when_started(&control, &job, "t1", Duration::ZERO);
    let chunks = collect(&model, &registry, cfg).await;
    mover.await.unwrap();

    assert!(chunks.iter().any(|c| matches!(
        c,
        StreamChunk::InboundDelivered { ids, .. } if ids == &["t1"]
    )));
    let result = results.try_recv().expect("reported when the run ended");
    assert!(result.delivered);
    assert_eq!(result.content.as_text(), Some("job done"));
}
