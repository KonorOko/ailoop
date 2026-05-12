//! Integration tests for the per-run timeout and external cancellation
//! primitives on `RunConfig`. They live alongside the engine tests
//! because they exercise the abort-signal plumbing through real awaits
//! (HTTP setup, SSE chunks, tool execution) where a unit test that
//! pokes `select!` directly would miss interaction bugs.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use ailoop::{Message, ToolDefinition, ToolResultContent, advanced::run_chat};
use ailoop_core::testing::{ScriptedError, ScriptedModel};
use ailoop_core::{
    CancellationToken, ChatMiddleware, ChatRequest, CompletionModel, FinishReason, HookAction,
    RunConfig, RunId, StepId, StreamChunk, ToolDecision, Usage,
};
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Value, json};

/// Mock `CompletionModel` whose `chat_stream` never resolves. Used to
/// trigger the abort signal deterministically — the engine sits in
/// `chat_stream().await`, abort wins the `select!`, run terminates.
struct BlockingModel;

#[async_trait]
impl CompletionModel for BlockingModel {
    type Error = ScriptedError;

    fn name(&self) -> &str {
        "blocking"
    }

    fn model(&self) -> &str {
        "blocking"
    }

    async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        std::future::pending().await
    }
}

/// Tool used by the "preserve tools_result on abort" test. The first
/// call returns successfully; the second call cancels the token and
/// then awaits forever, ensuring the abort path runs while the tool
/// future is still in flight.
struct CancelOnSecondCall {
    token: CancellationToken,
    calls: AtomicUsize,
}

#[async_trait]
impl ToolDyn for CancelOnSecondCall {
    fn name(&self) -> String {
        "cancel_on_second".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "cancel_on_second",
            "test tool",
            json!({"type":"object","properties":{},"required":[]}),
            vec![],
        )
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            ToolResultContent::text("first")
        } else {
            self.token.cancel();
            std::future::pending::<()>().await;
            unreachable!("cancellation should drop this future before it returns")
        }
    }
}

/// Records `on_run_finished` calls and the reason given. Used to verify
/// that abort paths still fire the lifecycle hook (analogous to the
/// `RecordingMiddleware` in `middleware_lifecycle.rs`, trimmed to what
/// these tests need).
#[derive(Default)]
struct RunFinishedRecorder {
    finished: AtomicUsize,
    last_reason: Mutex<Option<FinishReason>>,
}

#[async_trait]
impl ChatMiddleware for RunFinishedRecorder {
    async fn on_run_finished(
        &self,
        _run_id: &RunId,
        reason: &FinishReason,
        _usage: &Usage,
        _new_messages: &[Message],
    ) {
        self.finished.fetch_add(1, Ordering::SeqCst);
        *self.last_reason.lock().unwrap() = Some(reason.clone());
    }
}

fn run_finished<E>(chunks: &[Result<StreamChunk, E>]) -> &StreamChunk {
    chunks
        .iter()
        .find_map(|c| match c {
            Ok(chunk @ StreamChunk::RunFinished { .. }) => Some(chunk),
            _ => None,
        })
        .expect("expected RunFinished chunk")
}

#[tokio::test]
async fn timeout_aborts_run_during_chat_stream() {
    let registry = ToolRegistry::new();
    let mut config = RunConfig::default();
    config.timeout = Some(Duration::from_millis(30));

    let stream = run_chat(&BlockingModel, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream.collect().await;

    match run_finished(&chunks) {
        StreamChunk::RunFinished { reason, .. } => match reason {
            FinishReason::Aborted(msg) => assert!(
                msg.starts_with("timeout exceeded"),
                "unexpected abort reason: {msg}"
            ),
            other => panic!("expected Aborted reason, got {other:?}"),
        },
        other => panic!("expected RunFinished, got {other:?}"),
    }
}

#[tokio::test]
async fn cancellation_aborts_run_externally() {
    let token = CancellationToken::new();
    let token_for_task = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        token_for_task.cancel();
    });

    let registry = ToolRegistry::new();
    let mut config = RunConfig::default();
    config.cancellation = Some(token);

    let stream = run_chat(&BlockingModel, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream.collect().await;

    match run_finished(&chunks) {
        StreamChunk::RunFinished { reason, .. } => match reason {
            FinishReason::Aborted(msg) => assert_eq!(msg, "cancelled by caller"),
            other => panic!("expected Aborted reason, got {other:?}"),
        },
        other => panic!("expected RunFinished, got {other:?}"),
    }
}

#[tokio::test]
async fn no_timeout_or_cancellation_leaves_run_unaffected() {
    let model = ScriptedModel::new([vec![
        StreamChunk::TextDelta { delta: "hi".into() },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        },
    ]]);
    let registry = ToolRegistry::new();

    let stream = run_chat(
        &model,
        vec![Message::user("hi")],
        &registry,
        RunConfig::default(),
    )
    .await
    .expect("run_chat should start");
    let chunks: Vec<_> = stream.collect().await;

    match run_finished(&chunks) {
        StreamChunk::RunFinished { reason, .. } => assert!(
            matches!(reason, FinishReason::EndTurn),
            "expected EndTurn, got {reason:?}"
        ),
        other => panic!("expected RunFinished, got {other:?}"),
    }
}

#[tokio::test]
async fn cancellation_wins_over_timeout_when_both_configured() {
    // Both fire essentially simultaneously, so the abort future's
    // `biased` ordering is what determines the winner. Cancel must win
    // so callers can rely on the "cancelled by caller" reason.
    let token = CancellationToken::new();
    token.cancel();

    let registry = ToolRegistry::new();
    let mut config = RunConfig::default();
    config.timeout = Some(Duration::from_millis(0));
    config.cancellation = Some(token);

    let stream = run_chat(&BlockingModel, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream.collect().await;

    match run_finished(&chunks) {
        StreamChunk::RunFinished { reason, .. } => match reason {
            FinishReason::Aborted(msg) => assert_eq!(
                msg, "cancelled by caller",
                "expected cancellation to outrace timeout, got {msg}"
            ),
            other => panic!("expected Aborted reason, got {other:?}"),
        },
        other => panic!("expected RunFinished, got {other:?}"),
    }
}

#[tokio::test]
async fn abort_during_tool_loop_preserves_prior_tool_results() {
    use ailoop_core::{AssistantBlock, UserBlock};

    // Two-tool turn: the first tool runs to completion and pushes its
    // result to `tools_result`; the second tool fires the cancellation
    // and then hangs. The engine must persist the first tool's result
    // in `RunFinished.new_messages` so history is consistent on resume.
    let turn = vec![
        StreamChunk::ToolCallStarted {
            id: "toolu_a".into(),
            name: "cancel_on_second".into(),
        },
        StreamChunk::ToolCallFinished {
            id: "toolu_a".into(),
            name: "cancel_on_second".into(),
            args: json!({}),
        },
        StreamChunk::ToolCallStarted {
            id: "toolu_b".into(),
            name: "cancel_on_second".into(),
        },
        StreamChunk::ToolCallFinished {
            id: "toolu_b".into(),
            name: "cancel_on_second".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let model = ScriptedModel::new([turn]);

    let token = CancellationToken::new();
    let tool = Arc::new(CancelOnSecondCall {
        token: token.clone(),
        calls: AtomicUsize::new(0),
    });
    let mut registry = ToolRegistry::new();
    registry.register(tool).unwrap();

    let mut config = RunConfig::default();
    config.cancellation = Some(token);

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream.collect().await;

    let new_messages = match run_finished(&chunks) {
        StreamChunk::RunFinished {
            reason: FinishReason::Aborted(_),
            new_messages,
            ..
        } => new_messages,
        other => panic!("expected RunFinished{{Aborted}}, got {other:?}"),
    };

    let assistant_tool_call_ids: Vec<&str> = new_messages
        .iter()
        .filter_map(|m| match m {
            Message::Assistant { blocks } => Some(blocks),
            _ => None,
        })
        .flat_map(|blocks| blocks.iter())
        .filter_map(|b| match b {
            AssistantBlock::ToolCall { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(assistant_tool_call_ids, vec!["toolu_a", "toolu_b"]);

    let tool_result_ids: Vec<&str> = new_messages
        .iter()
        .filter_map(|m| match m {
            Message::User { blocks } => Some(blocks),
            _ => None,
        })
        .flat_map(|blocks| blocks.iter())
        .filter_map(|b| match b {
            UserBlock::ToolResult { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_result_ids.contains(&"toolu_a"),
        "first tool's ToolResult must be preserved, got {tool_result_ids:?}"
    );
}

#[tokio::test]
async fn on_run_finished_fires_when_run_aborts_via_timeout() {
    let recorder = Arc::new(RunFinishedRecorder::default());
    let registry = ToolRegistry::new();
    let mut config = RunConfig::default();
    config.timeout = Some(Duration::from_millis(20));
    config.middlewares = vec![recorder.clone()];

    let stream = run_chat(&BlockingModel, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let _: Vec<_> = stream.collect().await;

    assert_eq!(
        recorder.finished.load(Ordering::SeqCst),
        1,
        "on_run_finished must fire exactly once on timeout abort"
    );
    match recorder.last_reason.lock().unwrap().as_ref() {
        Some(FinishReason::Aborted(msg)) => assert!(
            msg.starts_with("timeout exceeded"),
            "unexpected reason: {msg}"
        ),
        other => panic!("expected Aborted reason, got {other:?}"),
    }
}

/// `on_before_tool_call` is the canonical "slow" middleware hook
/// (approval-style flows often block on a network call). A timeout
/// firing while the engine is awaiting a middleware decision must
/// abort the run rather than wait for the middleware to return.
#[tokio::test]
async fn timeout_aborts_run_inside_slow_middleware_hook() {
    struct SlowApproval;

    #[async_trait]
    impl ChatMiddleware for SlowApproval {
        async fn on_before_tool_call(
            &self,
            _run_id: &RunId,
            _step_id: &StepId,
            _name: &str,
            _args: &Value,
        ) -> ToolDecision {
            std::future::pending::<()>().await;
            unreachable!("timeout should drop this future before it returns")
        }
    }

    struct UnusedTool;
    #[async_trait]
    impl ToolDyn for UnusedTool {
        fn name(&self) -> String {
            "noop".into()
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "noop",
                "stub",
                json!({"type":"object","properties":{},"required":[]}),
                vec![],
            )
        }
        async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
            ToolResultContent::text("never")
        }
    }

    let turn = vec![
        StreamChunk::ToolCallStarted {
            id: "toolu_x".into(),
            name: "noop".into(),
        },
        StreamChunk::ToolCallFinished {
            id: "toolu_x".into(),
            name: "noop".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let model = ScriptedModel::new([turn]);

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(UnusedTool)).unwrap();

    let mw: Arc<dyn ChatMiddleware> = Arc::new(SlowApproval);
    let mut config = RunConfig::default();
    config.timeout = Some(Duration::from_millis(30));
    config.middlewares = vec![mw];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream.collect().await;

    match run_finished(&chunks) {
        StreamChunk::RunFinished { reason, .. } => match reason {
            FinishReason::Aborted(msg) => assert!(
                msg.starts_with("timeout exceeded"),
                "expected timeout reason, got {msg}"
            ),
            other => panic!("expected Aborted reason, got {other:?}"),
        },
        other => panic!("expected RunFinished, got {other:?}"),
    }
}

/// Wire-up test for `ToolContext::cancellation()`: a tool that clones
/// the token into a side task must see the same cancellation event
/// the caller triggers via `RunConfig.cancellation`. The engine's
/// `race_abort` drops the tool's own future when the token fires, but
/// the spawned watcher lives independently and survives long enough
/// to flip the observed flag.
#[tokio::test]
async fn tool_context_cancellation_mirrors_run_config() {
    struct ObserverTool {
        observed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl ToolDyn for ObserverTool {
        fn name(&self) -> String {
            "observer".into()
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "observer",
                "test tool",
                json!({"type":"object","properties":{},"required":[]}),
                vec![],
            )
        }
        async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
            let token = ctx.cancellation().clone();
            let flag = self.observed.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                flag.store(true, Ordering::SeqCst);
            });
            std::future::pending::<()>().await;
            unreachable!("cancellation should drop this future before it returns")
        }
    }

    let turn = vec![
        StreamChunk::ToolCallStarted {
            id: "toolu_obs".into(),
            name: "observer".into(),
        },
        StreamChunk::ToolCallFinished {
            id: "toolu_obs".into(),
            name: "observer".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let model = ScriptedModel::new([turn]);

    let observed = Arc::new(AtomicBool::new(false));
    let mut registry = ToolRegistry::new();
    registry
        .register(Arc::new(ObserverTool {
            observed: observed.clone(),
        }))
        .unwrap();

    let token = CancellationToken::new();
    let token_for_task = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        token_for_task.cancel();
    });

    let mut config = RunConfig::default();
    config.cancellation = Some(token);

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let _: Vec<_> = stream.collect().await;

    // The tool future is dropped by `race_abort` the moment the token
    // fires; the spawned watcher lives in the runtime independently
    // and needs a moment to set the flag.
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(
        observed.load(Ordering::SeqCst),
        "tool's side task must observe cancellation through ctx.cancellation()"
    );
}

/// Pre-existing `HookAction::Terminate` path was already covered by the
/// engine unit tests; this test pins it through the public reliability
/// surface so we catch a regression where the abort-hook helper
/// accidentally changes ordering or skips `on_run_finished`.
#[tokio::test]
async fn hook_action_terminate_still_fires_on_run_finished() {
    struct AbortingMw;

    #[async_trait]
    impl ChatMiddleware for AbortingMw {
        async fn on_run_started(
            &self,
            _run_id: &RunId,
            _messages: &[Message],
            _config: &RunConfig,
        ) -> HookAction {
            HookAction::Terminate {
                reason: "no go".into(),
            }
        }
    }

    let recorder = Arc::new(RunFinishedRecorder::default());
    let registry = ToolRegistry::new();
    let mws: Vec<Arc<dyn ChatMiddleware>> = vec![Arc::new(AbortingMw), recorder.clone()];
    let mut config = RunConfig::default();
    config.middlewares = mws;

    let model = ScriptedModel::new(Vec::<Vec<StreamChunk>>::new());
    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let _: Vec<_> = stream.collect().await;

    assert_eq!(recorder.finished.load(Ordering::SeqCst), 1);
    match recorder.last_reason.lock().unwrap().as_ref() {
        Some(FinishReason::Aborted(r)) => assert_eq!(r, "no go"),
        other => panic!("expected Aborted reason, got {other:?}"),
    }
}
