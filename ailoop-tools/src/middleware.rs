use ailoop_core::{ChatMiddleware, ToolDecision};
use serde_json::Value;

pub struct ApprovaleGate {}

#[async_trait::async_trait]
impl ChatMiddleware for ApprovaleGate {
    async fn on_before_tool_call(&self, name: &str, args: &Value) -> ToolDecision {
        ToolDecision::Continue
    }
}
