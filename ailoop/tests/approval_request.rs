//! The approval callback receives an [`ApprovalRequest`] carrying the
//! call, the tool's tags and the context the model saw on that step,
//! keyed per run so concurrent runs sharing one middleware never see
//! each other's messages.

use std::sync::{Arc, Mutex};

use ailoop::{
    ApprovalMiddleware, ApprovalRequest, ChatRequest, CompletionModel, Conversation, FinishReason,
    Message, RunConfig, RunId, StreamChunk, ToolDecision, ToolRegistry, ToolTag, Usage,
    advanced::run_chat, ailoop_tool,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel};
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Value, json};
use tokio::sync::Barrier;

#[ailoop_tool(description = "delete a path", tags(Destructive, WritesFiles))]
async fn delete_file(_path: String) -> i32 {
    0
}

fn tool_turn(id: &str, args: Value) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCallStarted {
            call_id: id.into(),
            name: "delete_file".into(),
        },
        StreamChunk::ToolCallFinished {
            call_id: id.into(),
            name: "delete_file".into(),
            args,
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ]
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

fn delete_then_finish(path: &str) -> ScriptedModel {
    ScriptedModel::new([
        tool_turn("toolu_1", json!({ "path": path })),
        text_turn("done"),
    ])
}

#[tokio::test]
async fn approval_request_carries_call_tags_and_user_message() {
    let seen: Arc<Mutex<Vec<ApprovalRequest>>> = Arc::default();
    let seen_cb = seen.clone();

    let mut chat = Conversation::builder(delete_then_finish("build/"))
        .tool(DeleteFile)
        .approval(move |req| {
            seen_cb.lock().unwrap().push(req);
            async { ToolDecision::Continue }
        })
        .build()
        .expect("build");

    let outcome = chat.run("clean the build").await.expect("run");

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1, "one gated call");
    let req = &seen[0];
    assert_eq!(req.name, "delete_file");
    assert_eq!(req.args, json!({ "path": "build/" }));
    assert_eq!(&*req.tags, &[ToolTag::Destructive, ToolTag::WritesFiles]);
    assert_eq!(req.run_id, outcome.run_id);
    assert!(
        req.messages.contains(&Message::user("clean the build")),
        "messages must include the run's user message, got {:?}",
        req.messages
    );
}

/// `ScriptedModel` whose first `chat_stream` waits on a barrier shared
/// with another run. Both runs have therefore recorded their step
/// context (`on_chat_request` precedes `chat_stream`) before either one
/// reaches its gated tool call.
struct BarrierModel {
    inner: ScriptedModel,
    barrier: Arc<Barrier>,
    waited: Mutex<bool>,
}

#[async_trait::async_trait]
impl CompletionModel for BarrierModel {
    type Error = ScriptedError;

    fn name(&self) -> &str {
        self.inner.name()
    }
    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        let first = !std::mem::replace(&mut *self.waited.lock().unwrap(), true);
        if first {
            self.barrier.wait().await;
        }
        self.inner.chat_stream(req).await
    }
}

async fn run_to_finish(
    model: &BarrierModel,
    registry: &ToolRegistry,
    gate: Arc<ApprovalMiddleware>,
    prompt: &str,
) -> RunId {
    let mut config = RunConfig::default();
    config.middlewares = vec![gate];
    let stream = run_chat(model, vec![Message::user(prompt)], registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream.collect().await;
    chunks
        .into_iter()
        .find_map(|c| match c {
            Ok(StreamChunk::RunFinished { run_id, .. }) => Some(run_id),
            _ => None,
        })
        .expect("run should finish")
}

#[tokio::test]
async fn concurrent_runs_sharing_a_gate_do_not_mix_context() {
    let seen: Arc<Mutex<Vec<ApprovalRequest>>> = Arc::default();
    let seen_cb = seen.clone();
    let gate = Arc::new(ApprovalMiddleware::for_named(["delete_file"], move |req| {
        seen_cb.lock().unwrap().push(req);
        async { ToolDecision::Continue }
    }));

    let barrier = Arc::new(Barrier::new(2));
    let model_a = BarrierModel {
        inner: delete_then_finish("a/"),
        barrier: barrier.clone(),
        waited: Mutex::new(false),
    };
    let model_b = BarrierModel {
        inner: delete_then_finish("b/"),
        barrier,
        waited: Mutex::new(false),
    };
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(DeleteFile)).unwrap();

    let (run_a, run_b) = tokio::join!(
        run_to_finish(&model_a, &registry, gate.clone(), "clean a"),
        run_to_finish(&model_b, &registry, gate.clone(), "clean b"),
    );
    assert_ne!(run_a, run_b);

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "one gated call per run");
    for req in seen.iter() {
        let (own, other) = if req.run_id == run_a {
            (("a/", "clean a"), "clean b")
        } else {
            assert_eq!(req.run_id, run_b);
            (("b/", "clean b"), "clean a")
        };
        assert_eq!(req.args, json!({ "path": own.0 }));
        assert!(req.messages.contains(&Message::user(own.1)));
        assert!(
            !req.messages.contains(&Message::user(other)),
            "run {:?} saw the other run's context: {:?}",
            req.run_id,
            req.messages
        );
        assert!(req.tags.is_empty(), "for_named does not know tool tags");
    }
}
