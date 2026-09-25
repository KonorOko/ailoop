//! The engine only runs tools in the run's active set. A deferred tool
//! the model names without activating it, or a tool removed by
//! `with_capabilities`, gets an in-band "not found" error result and
//! the run goes on.

use std::sync::{Arc, Mutex};

use ailoop::{
    ChatMiddleware, Conversation, FinishReason, Message, StreamChunk, ToolCallInfo, ToolContext,
    ToolDecision, ToolDefinition, ToolDyn, ToolResultContent, ToolTag, Usage, UserBlock,
};
use ailoop_core::testing::ScriptedModel;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Meta-tool: activates the tool named in `args.name`.
struct EnableTool {
    tags: Vec<ToolTag>,
}

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
            self.tags.clone(),
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

/// Records the name passed to every `on_before_tool_call`.
#[derive(Default)]
struct HookLog(Mutex<Vec<String>>);

#[async_trait]
impl ChatMiddleware for HookLog {
    async fn on_before_tool_call(&self, call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        self.0.lock().unwrap().push(call.name.clone());
        ToolDecision::Continue
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

/// Every tool result in history, by call id.
fn tool_results(messages: &[Message]) -> Vec<(String, ToolResultContent)> {
    messages
        .iter()
        .filter_map(|m| match m {
            Message::User { blocks } => Some(blocks),
            _ => None,
        })
        .flatten()
        .filter_map(|b| match b {
            UserBlock::ToolResult {
                call_id, content, ..
            } => Some((call_id.clone(), content.clone())),
            _ => None,
        })
        .collect()
}

fn result_for(messages: &[Message], call_id: &str) -> ToolResultContent {
    tool_results(messages)
        .into_iter()
        .find(|(id, _)| id == call_id)
        .map(|(_, content)| content)
        .unwrap_or_else(|| panic!("no tool_result for {call_id}"))
}

fn assert_not_found(content: &ToolResultContent, name: &str) {
    assert!(content.is_error, "expected an error result: {content:?}");
    let text = content.collect_text();
    assert!(
        text.contains(&format!("Tool '{name}' not found")),
        "unexpected error text: {text}"
    );
}

#[tokio::test]
async fn deferred_tool_called_without_activation_does_not_run() {
    let executed = Arc::new(Mutex::new(0));
    let approvals = Arc::new(Mutex::new(0));
    let approvals_cb = approvals.clone();
    let hooks = Arc::new(HookLog::default());

    let model = ScriptedModel::new([
        tool_turn("toolu_1", "delete_file", json!({})),
        text_turn("ok"),
    ]);
    let mut chat = Conversation::builder(model)
        .tool_dyn(Arc::new(EnableTool { tags: vec![] }))
        .tool_dyn(Arc::new(Recording {
            name: "delete_file",
            tags: vec![ToolTag::Destructive],
            executed: executed.clone(),
        }))
        .initial_active_tools(["enable_tool"])
        .middleware(hooks.clone())
        .with_approval(move |_req| {
            *approvals_cb.lock().unwrap() += 1;
            async { ToolDecision::Continue }
        })
        .build()
        .expect("build");

    let outcome = chat.run("go").await.expect("run");

    assert!(
        matches!(outcome.finish_reason, FinishReason::EndTurn),
        "the run must go on: {:?}",
        outcome.finish_reason
    );
    assert_eq!(*executed.lock().unwrap(), 0, "inactive tool must not run");
    assert_eq!(*approvals.lock().unwrap(), 0, "rejected before the gate");
    assert!(hooks.0.lock().unwrap().is_empty(), "no tool hook fires");

    let content = result_for(chat.history_messages(), "toolu_1");
    assert_not_found(&content, "delete_file");
    assert!(
        content
            .collect_text()
            .contains("Available tools: [enable_tool]"),
        "must list only the active tools: {}",
        content.collect_text()
    );
}

#[tokio::test]
async fn tool_activated_at_runtime_runs() {
    let executed = Arc::new(Mutex::new(0));
    let model = ScriptedModel::new([
        tool_turn("toolu_1", "enable_tool", json!({"name": "read_file"})),
        tool_turn("toolu_2", "read_file", json!({})),
        text_turn("ok"),
    ]);
    let mut chat = Conversation::builder(model)
        .tool_dyn(Arc::new(EnableTool { tags: vec![] }))
        .tool_dyn(Arc::new(Recording {
            name: "read_file",
            tags: vec![ToolTag::ReadOnly],
            executed: executed.clone(),
        }))
        .initial_active_tools(["enable_tool"])
        .build()
        .expect("build");

    let outcome = chat.run("go").await.expect("run");

    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(*executed.lock().unwrap(), 1);
    let content = result_for(chat.history_messages(), "toolu_2");
    assert!(!content.is_error);
    assert_eq!(content.collect_text(), "done");
}

#[tokio::test]
async fn capability_filtered_tool_never_runs() {
    let executed = Arc::new(Mutex::new(0));
    let model = ScriptedModel::new([
        // Called by name, then activated, then called again.
        tool_turn("toolu_1", "delete_file", json!({})),
        tool_turn("toolu_2", "enable_tool", json!({"name": "delete_file"})),
        tool_turn("toolu_3", "delete_file", json!({})),
        text_turn("ok"),
    ]);
    let mut chat = Conversation::builder(model)
        .tool_dyn(Arc::new(EnableTool {
            tags: vec![ToolTag::ReadOnly],
        }))
        .tool_dyn(Arc::new(Recording {
            name: "delete_file",
            tags: vec![ToolTag::Destructive],
            executed: executed.clone(),
        }))
        .with_capabilities(&[ToolTag::ReadOnly])
        .build()
        .expect("build");

    let outcome = chat.run("go").await.expect("run");

    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(*executed.lock().unwrap(), 0, "filtered tool must never run");

    let messages = chat.history_messages();
    assert_not_found(&result_for(messages, "toolu_1"), "delete_file");
    let activation = result_for(messages, "toolu_2");
    assert!(activation.is_error, "activate must fail: {activation:?}");
    assert!(activation.collect_text().contains("not registered"));
    assert_not_found(&result_for(messages, "toolu_3"), "delete_file");
}
