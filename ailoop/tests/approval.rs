use ailoop::{ApprovalMiddleware, Conversation, ToolDecision, ailoop_tool};
use ailoop_core::{ChatMiddleware, ChatRequest, CompletionModel, StreamChunk, ToolCallInfo};
use futures::stream::BoxStream;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct MockModel;

#[async_trait::async_trait]
impl CompletionModel for MockModel {
    type Error = std::convert::Infallible;

    fn name(&self) -> &str {
        "mock"
    }
    fn model(&self) -> &str {
        "mock"
    }

    async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[ailoop_tool(description = "list", tags(ReadOnly))]
async fn list_dir(_path: String) -> i32 {
    0
}

#[ailoop_tool(description = "delete", tags(Destructive, WritesFiles))]
async fn delete_file(_path: String) -> i32 {
    0
}

#[tokio::test]
async fn approve_all_fires_for_every_tool() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_cb = counter.clone();
    let mw = ApprovalMiddleware::approve_all(move |_req| {
        let c = counter_cb.clone();
        async move {
            c.fetch_add(1, Ordering::SeqCst);
            ToolDecision::Continue
        }
    });

    let run_id = ailoop_core::RunId::new();
    let step_id = ailoop_core::StepId::new();
    let _ = mw
        .on_before_tool_call(
            &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "anything"),
            &json!({}),
        )
        .await;
    let _ = mw
        .on_before_tool_call(
            &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "else"),
            &json!({}),
        )
        .await;

    assert_eq!(counter.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn for_named_only_fires_for_listed_tools() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_cb = counter.clone();
    let mw = ApprovalMiddleware::for_named(["delete_file"], move |_req| {
        let c = counter_cb.clone();
        async move {
            c.fetch_add(1, Ordering::SeqCst);
            ToolDecision::Continue
        }
    });

    let run_id = ailoop_core::RunId::new();
    let step_id = ailoop_core::StepId::new();
    let _ = mw
        .on_before_tool_call(
            &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "list_dir"),
            &json!({}),
        )
        .await;
    let _ = mw
        .on_before_tool_call(
            &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "delete_file"),
            &json!({}),
        )
        .await;
    let _ = mw
        .on_before_tool_call(
            &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "other"),
            &json!({}),
        )
        .await;

    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn for_named_returns_continue_for_non_gated() {
    let mw = ApprovalMiddleware::for_named(["delete_file"], |_req| async move {
        ToolDecision::Skip {
            reason: "should-not-fire".into(),
        }
    });

    let run_id = ailoop_core::RunId::new();
    let step_id = ailoop_core::StepId::new();
    let decision = mw
        .on_before_tool_call(
            &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "list_dir"),
            &json!({}),
        )
        .await;
    assert!(matches!(decision, ToolDecision::Continue));
}

#[tokio::test]
async fn approval_callback_decision_is_returned() {
    let mw = ApprovalMiddleware::approve_all(|_req| async move {
        ToolDecision::Skip {
            reason: "denied".into(),
        }
    });

    let run_id = ailoop_core::RunId::new();
    let step_id = ailoop_core::StepId::new();
    let decision = mw
        .on_before_tool_call(
            &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "anything"),
            &json!({}),
        )
        .await;
    match decision {
        ToolDecision::Skip { reason } => assert_eq!(reason, "denied"),
        _ => panic!("expected Skip"),
    }
}

#[test]
fn builder_with_approval_compiles_and_builds() {
    // Smoke test: the builder methods compose with tool registration and
    // produce a working Conversation. Wiring correctness is verified in
    // the conversation.rs unit tests with private-field access.
    let _chat = Conversation::builder(MockModel)
        .tool(ListDir)
        .tool(DeleteFile)
        .with_approval(|_req| async move { ToolDecision::Continue })
        .build()
        .unwrap();
}
