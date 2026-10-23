//! `StepInfo` tells `on_chat_request` which iteration it is in and the
//! run's cap.

use std::sync::{Arc, Mutex};

use ailoop::{
    ChatMiddleware, ChatRequest, FinishReason, Message, RunConfig, StepInfo, StreamChunk,
    ToolDefinition, ToolResultContent, Usage, advanced::run_chat,
};
use ailoop_core::testing::ScriptedModel;
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
use futures::StreamExt;
use serde_json::{Value, json};

#[derive(Default)]
struct Steps(Mutex<Vec<(usize, usize)>>);

#[async_trait::async_trait]
impl ChatMiddleware for Steps {
    async fn on_chat_request(&self, step: &StepInfo, _req: &mut ChatRequest) {
        self.0
            .lock()
            .unwrap()
            .push((step.iteration, step.max_iterations));
    }
}

struct Noop;

#[async_trait::async_trait]
impl ToolDyn for Noop {
    fn name(&self) -> &str {
        "noop"
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new("noop", "does nothing", json!({"type": "object"}), vec![])
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        ToolResultContent::text("ok")
    }
}

#[tokio::test]
async fn on_chat_request_sees_iteration_and_cap() {
    let tool_turn = vec![
        StreamChunk::ToolCallStarted {
            call_id: "t1".into(),
            name: "noop".into(),
        },
        StreamChunk::ToolCallFinished {
            call_id: "t1".into(),
            name: "noop".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let end_turn = vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }];
    let model = ScriptedModel::new([tool_turn, end_turn]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Noop)).unwrap();
    let steps = Arc::new(Steps::default());
    let mut config = RunConfig::default();
    config.max_iterations = 7;
    config.middlewares = vec![steps.clone()];

    let _: Vec<_> = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run starts")
        .collect()
        .await;

    assert_eq!(*steps.0.lock().unwrap(), [(0, 7), (1, 7)]);
}

#[test]
fn step_info_defaults_and_builder() {
    let step = StepInfo::new(ailoop::RunId::new(), ailoop::StepId::new());
    assert_eq!(
        (step.iteration, step.max_iterations),
        (0, ailoop::DEFAULT_MAX_ITERATIONS)
    );
    let step = step.with_iteration(3, 10);
    assert_eq!((step.iteration, step.max_iterations), (3, 10));
}
