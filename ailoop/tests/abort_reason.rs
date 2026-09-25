//! Each [`AbortReason`] variant is produced by the scenario it names,
//! observed through the public `Conversation` surface. Aborts are `Ok`,
//! so every test unwraps the outcome and matches the typed reason —
//! never the rendered text.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use ailoop::{
    AbortReason, AssistantBlock, CancellationToken, ChatMiddleware, ChatRequest, CompletionModel,
    Conversation, FinishReason, HookAction, Message, RunErrorInfo, RunFinishedInfo, RunOptions,
    RunStartInfo, StreamChunk, ToolCallInfo, ToolDecision, ToolDefinition, ToolResultContent,
    Usage, UserBlock,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel};
use ailoop_tools::{ToolContext, ToolDyn};
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::{Value, json};

/// `chat_stream` never resolves, so the abort future always wins.
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

struct GetWeather;

#[async_trait]
impl ToolDyn for GetWeather {
    fn name(&self) -> &str {
        "get_weather"
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

fn tool_turn(id: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCallStarted {
            call_id: id.into(),
            name: "get_weather".into(),
        },
        StreamChunk::ToolCallFinished {
            call_id: id.into(),
            name: "get_weather".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ]
}

#[tokio::test]
async fn timeout_produces_timeout_with_configured_duration() {
    let timeout = Duration::from_millis(20);
    let mut chat = Conversation::builder(BlockingModel).build().expect("build");

    let outcome = chat
        .run_with_options("hi", RunOptions::new().timeout(timeout))
        .await
        .expect("aborts are Ok");

    match outcome.finish_reason {
        FinishReason::Aborted(AbortReason::Timeout(d)) => assert_eq!(d, timeout),
        other => panic!("expected Aborted(Timeout), got {other:?}"),
    }
}

#[tokio::test]
async fn cancellation_produces_cancelled() {
    let token = CancellationToken::new();
    let mut chat = Conversation::builder(BlockingModel).build().expect("build");

    let canceller = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        canceller.cancel();
    });

    let outcome = chat
        .run_with_options("hi", RunOptions::new().cancellation(token))
        .await
        .expect("aborts are Ok");

    match outcome.finish_reason {
        FinishReason::Aborted(AbortReason::Cancelled) => {}
        other => panic!("expected Aborted(Cancelled), got {other:?}"),
    }
}

#[tokio::test]
async fn hook_terminate_produces_terminated() {
    struct Deny;

    #[async_trait]
    impl ChatMiddleware for Deny {
        async fn on_run_started(&self, _run: &RunStartInfo<'_>) -> HookAction {
            HookAction::Terminate {
                reason: "quota exhausted".into(),
            }
        }
    }

    let model = ScriptedModel::new(Vec::<Vec<StreamChunk>>::new());
    let mut chat = Conversation::builder(model)
        .middleware(Arc::new(Deny))
        .build()
        .expect("build");

    let outcome = chat.run("hi").await.expect("aborts are Ok");

    match outcome.finish_reason {
        FinishReason::Aborted(ref r @ AbortReason::Terminated { ref reason, .. }) => {
            assert_eq!(reason, "quota exhausted");
            assert_eq!(r.to_string(), "quota exhausted");
        }
        other => panic!("expected Aborted(Terminated), got {other:?}"),
    }
}

#[tokio::test]
async fn tool_terminate_produces_tool_terminated_with_tool_name() {
    struct DenyTools;

    #[async_trait]
    impl ChatMiddleware for DenyTools {
        async fn on_before_tool_call(&self, _call: &ToolCallInfo, _args: &Value) -> ToolDecision {
            ToolDecision::Terminate {
                reason: "tool not allowed".into(),
            }
        }
    }

    let model = ScriptedModel::new(vec![tool_turn("toolu_1")]);
    let mut chat = Conversation::builder(model)
        .tool(GetWeather)
        .middleware(Arc::new(DenyTools))
        .build()
        .expect("build");

    let outcome = chat.run("hi").await.expect("aborts are Ok");

    match outcome.finish_reason {
        FinishReason::Aborted(
            ref r @ AbortReason::ToolTerminated {
                ref tool_name,
                ref reason,
                ..
            },
        ) => {
            assert_eq!(tool_name, "get_weather");
            assert_eq!(reason, "tool not allowed");
            assert_eq!(r.to_string(), "tool not allowed");
        }
        other => panic!("expected Aborted(ToolTerminated), got {other:?}"),
    }
}

#[derive(Default)]
struct LifecycleCounter {
    finished: AtomicUsize,
    errored: AtomicUsize,
}

#[async_trait]
impl ChatMiddleware for LifecycleCounter {
    async fn on_run_finished(&self, _run: &RunFinishedInfo<'_>) {
        self.finished.fetch_add(1, Ordering::SeqCst);
    }

    async fn on_run_error(&self, _run: &RunErrorInfo<'_>) {
        self.errored.fetch_add(1, Ordering::SeqCst);
    }
}

/// Hitting `max_iterations` is an abort, not an error: the run returns
/// `Ok`, keeps every completed tool_use/tool_result pair, persists them
/// to history, and fires `on_run_finished` (not `on_run_error`) once.
#[tokio::test]
async fn max_iterations_produces_max_iterations_and_keeps_partial_work() {
    let model = ScriptedModel::new((0..5).map(|i| tool_turn(&format!("toolu_{i}"))));
    let counter = Arc::new(LifecycleCounter::default());
    let mut chat = Conversation::builder(model)
        .tool(GetWeather)
        .middleware(counter.clone())
        .build()
        .expect("build");

    let outcome = chat
        .run_with_options("hi", RunOptions::new().max_iterations(2))
        .await
        .expect("max_iterations is an abort, not an Err");

    match outcome.finish_reason {
        FinishReason::Aborted(AbortReason::MaxIterations(n)) => assert_eq!(n, 2),
        other => panic!("expected Aborted(MaxIterations), got {other:?}"),
    }

    let tool_calls = outcome
        .new_messages
        .iter()
        .filter_map(|m| match m {
            Message::Assistant { blocks } => Some(blocks),
            _ => None,
        })
        .flatten()
        .filter(|b| matches!(b, AssistantBlock::ToolCall { .. }))
        .count();
    let tool_results = outcome
        .new_messages
        .iter()
        .filter_map(|m| match m {
            Message::User { blocks } => Some(blocks),
            _ => None,
        })
        .flatten()
        .filter(|b| matches!(b, UserBlock::ToolResult { .. }))
        .count();
    assert_eq!(tool_calls, 2, "one tool call per completed iteration");
    assert_eq!(tool_results, 2, "every tool_use keeps its tool_result");

    // Kickoff + the run's new messages land in history.
    assert_eq!(
        chat.messages().len(),
        1 + outcome.new_messages.len(),
        "partial work must be persisted to history"
    );

    assert_eq!(counter.finished.load(Ordering::SeqCst), 1);
    assert_eq!(counter.errored.load(Ordering::SeqCst), 0);
}
