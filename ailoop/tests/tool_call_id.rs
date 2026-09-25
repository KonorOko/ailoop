//! Every tool hook receives the provider's call id through
//! [`ToolCallInfo`], and the tool sees the same id on
//! [`ToolContext::call_id`]. The ids must match the ones on the
//! `ToolCallFinished` / `ToolResult` chunks, so a middleware can pair the
//! hooks of one call even when two calls in a step are otherwise
//! identical.

use std::sync::{Arc, Mutex};

use ailoop::{
    ApprovalMiddleware, ChatMiddleware, Conversation, FinishReason, StreamChunk, ToolCallInfo,
    ToolContext, ToolDecision, ToolDefinition, ToolDyn, ToolResultContent, Usage,
};
use ailoop_core::testing::ScriptedModel;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Records `(hook, call_id, name)` for every tool hook and the ids on
/// the tool chunks of the stream.
#[derive(Default)]
struct CallLog {
    hooks: Mutex<Vec<(&'static str, String, String)>>,
    finished: Mutex<Vec<String>>,
    results: Mutex<Vec<String>>,
}

impl CallLog {
    fn push(&self, hook: &'static str, call: &ToolCallInfo) {
        self.hooks
            .lock()
            .unwrap()
            .push((hook, call.call_id.clone(), call.name.clone()));
    }
}

#[async_trait]
impl ChatMiddleware for CallLog {
    async fn on_chunk(&self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::ToolCallFinished { call_id: id, .. } => {
                self.finished.lock().unwrap().push(id.clone())
            }
            StreamChunk::ToolResult { call_id, .. } => {
                self.results.lock().unwrap().push(call_id.clone())
            }
            _ => {}
        }
    }
    async fn on_before_tool_call_mut(&self, call: &ToolCallInfo, _args: &mut Value) {
        self.push("before_mut", call);
    }
    async fn on_before_tool_call(&self, call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        self.push("before", call);
        ToolDecision::Continue
    }
    async fn on_after_tool_call_mut(
        &self,
        call: &ToolCallInfo,
        _args: &Value,
        _result: &mut ToolResultContent,
    ) {
        self.push("after_mut", call);
    }
    async fn on_after_tool_call(
        &self,
        call: &ToolCallInfo,
        _args: &Value,
        _result: &ToolResultContent,
    ) {
        self.push("after", call);
    }
}

/// Records the `ToolContext::call_id` of every dispatch.
struct Echo {
    seen: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl ToolDyn for Echo {
    fn name(&self) -> String {
        "echo".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "echo",
            "stub",
            json!({"type":"object","properties":{},"required":[]}),
            vec![],
        )
    }
    async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
        self.seen.lock().unwrap().push(ctx.call_id().to_string());
        ToolResultContent::text(ctx.call_id())
    }
}

/// One step with two calls to the same tool with the same args, then a
/// closing text turn.
fn model() -> ScriptedModel {
    let mut turn = Vec::new();
    for id in ["toolu_a", "toolu_b"] {
        turn.push(StreamChunk::ToolCallStarted {
            call_id: id.into(),
            name: "echo".into(),
        });
        turn.push(StreamChunk::ToolCallFinished {
            call_id: id.into(),
            name: "echo".into(),
            args: json!({"x": 1}),
        });
    }
    turn.push(StreamChunk::TurnFinished {
        reason: FinishReason::ToolUse,
        usage: Usage::default(),
        service_tier: None,
    });
    let done = vec![
        StreamChunk::TextDelta { delta: "ok".into() },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    ScriptedModel::new([turn, done])
}

#[tokio::test]
async fn every_tool_hook_receives_its_call_id() {
    let log = Arc::new(CallLog::default());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut chat = Conversation::builder(model())
        .tool_dyn(Arc::new(Echo { seen: seen.clone() }))
        .middleware(log.clone())
        .build()
        .expect("build");

    let outcome = chat.run("go").await.expect("run");
    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));

    let ids = ["toolu_a", "toolu_b"];
    assert_eq!(*log.finished.lock().unwrap(), ids);
    assert_eq!(*log.results.lock().unwrap(), ids);
    assert_eq!(*seen.lock().unwrap(), ids, "ToolContext::call_id");

    let hooks = log.hooks.lock().unwrap();
    for id in ids {
        let per_call: Vec<&str> = hooks
            .iter()
            .filter(|(_, call_id, _)| call_id == id)
            .map(|(hook, _, name)| {
                assert_eq!(name, "echo");
                *hook
            })
            .collect();
        assert_eq!(
            per_call,
            ["before_mut", "before", "after_mut", "after"],
            "hooks for {id}"
        );
    }
    assert_eq!(
        hooks.len(),
        8,
        "no hook fired without a known id: {hooks:?}"
    );
}

#[tokio::test]
async fn approval_request_carries_the_call_id() {
    let requested = Arc::new(Mutex::new(Vec::new()));
    let requested_cb = requested.clone();
    let mut chat = Conversation::builder(model())
        .tool_dyn(Arc::new(Echo {
            seen: Arc::default(),
        }))
        .middleware(Arc::new(ApprovalMiddleware::approve_all(move |req| {
            requested_cb.lock().unwrap().push(req.call_id.clone());
            async { ToolDecision::Continue }
        })))
        .build()
        .expect("build");

    chat.run("go").await.expect("run");

    assert_eq!(*requested.lock().unwrap(), ["toolu_a", "toolu_b"]);
}
