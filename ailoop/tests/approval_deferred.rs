//! The capability-tag approval gate must cover tools that start
//! deferred (registered but not in the initial active set) and are
//! activated mid-run via `ctx.tools().activate(...)` — the
//! `search_tools` pattern.

use std::sync::{Arc, Mutex};

use ailoop::{
    Conversation, FinishReason, StreamChunk, ToolContext, ToolDecision, ToolDefinition, ToolDyn,
    ToolResultContent, ToolTag, Usage,
};
use ailoop_core::testing::ScriptedModel;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Untagged meta-tool: activates the tool named in `args.name`.
struct EnableTool;

#[async_trait]
impl ToolDyn for EnableTool {
    fn name(&self) -> String {
        "enable_tool".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "enable_tool",
            "activate a deferred tool",
            json!({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}),
            vec![],
        )
    }
    async fn call(&self, args: Value, ctx: &ToolContext) -> ToolResultContent {
        let name = args["name"].as_str().unwrap_or_default();
        match ctx.tools().activate(name) {
            Ok(()) => ToolResultContent::text(format!("{name} enabled")),
            Err(e) => ToolResultContent::error(e.to_string()),
        }
    }
}

/// Tagged tool that records every execution.
struct Recording {
    name: &'static str,
    tags: Vec<ToolTag>,
    executed: Arc<Mutex<usize>>,
}

#[async_trait]
impl ToolDyn for Recording {
    fn name(&self) -> String {
        self.name.into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name,
            "stub",
            json!({"type":"object","properties":{},"required":[]}),
            self.tags.clone(),
        )
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        *self.executed.lock().unwrap() += 1;
        ToolResultContent::text("done")
    }
}

fn tool_turn(id: &str, name: &str, args: Value) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCallStarted {
            call_id: id.into(),
            name: name.into(),
        },
        StreamChunk::ToolCallFinished {
            call_id: id.into(),
            name: name.into(),
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

/// Script: activate `target`, call it, then finish with text.
fn activate_then_call(target: &str) -> ScriptedModel {
    ScriptedModel::new([
        tool_turn("toolu_1", "enable_tool", json!({"name": target})),
        tool_turn("toolu_2", target, json!({})),
        text_turn("ok"),
    ])
}

/// Runs the scripted activate-then-call flow under `approval`
/// and returns (callback invocations, target executions).
async fn run_gated(target: &'static str, tags: Vec<ToolTag>) -> (Vec<String>, usize) {
    let executed = Arc::new(Mutex::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_cb = seen.clone();

    let mut chat = Conversation::builder(activate_then_call(target))
        .tool_dyn(Arc::new(EnableTool))
        .tool_dyn(Arc::new(Recording {
            name: target,
            tags,
            executed: executed.clone(),
        }))
        .initial_active_tools(["enable_tool"])
        .approval(move |req| {
            seen_cb.lock().unwrap().push(req.name);
            async move {
                ToolDecision::Skip {
                    reason: "denied".into(),
                }
            }
        })
        .build()
        .expect("build");

    assert_eq!(chat.active_tool_names(), vec!["enable_tool".to_string()]);

    let outcome = chat.run("go").await.expect("run");
    assert!(
        matches!(outcome.finish_reason, FinishReason::EndTurn),
        "unexpected finish: {:?}",
        outcome.finish_reason
    );

    let seen = seen.lock().unwrap().clone();
    let executed = *executed.lock().unwrap();
    (seen, executed)
}

#[tokio::test]
async fn deferred_destructive_tool_activated_at_runtime_is_gated() {
    let (seen, executed) = run_gated("delete_file", vec![ToolTag::Destructive]).await;

    assert_eq!(seen, vec!["delete_file".to_string()]);
    assert_eq!(executed, 0, "Skip must prevent execution");
}

#[tokio::test]
async fn deferred_readonly_tool_activated_at_runtime_is_not_gated() {
    let (seen, executed) = run_gated("read_file", vec![ToolTag::ReadOnly]).await;

    assert!(seen.is_empty(), "ReadOnly tool must not reach the callback");
    assert_eq!(executed, 1);
}
