use serde::Serialize;
use serde_json::Value;

use crate::Message;

#[derive(Clone)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop_sequences: Vec<String>,
    pub max_tokens: u32,
    /// How the model should pick among the available tools. `None`
    /// leaves the provider default (typically `Auto`). Mapped to each
    /// provider's wire format by the adapter.
    pub tool_choice: Option<ToolChoice>,
    /// Forbid the model from emitting more than one `tool_use` block
    /// per turn. `None` leaves the provider default (parallel allowed).
    /// Adapters lower this to the field their API expects:
    /// `tool_choice.disable_parallel_tool_use` for Anthropic,
    /// `parallel_tool_calls` (negated) for Chat Completions.
    pub disable_parallel_tool_use: Option<bool>,

    pub additional_params: Option<Value>,
}

/// Constraint placed on the model's tool selection for a single
/// request. Variant naming follows Anthropic's wire vocabulary; the
/// Chat Completions adapter translates `Any` → `"required"` and
/// `None_` → `"none"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// Model decides whether to call a tool. Provider default.
    Auto,
    /// Model must call **some** tool, but may pick which one.
    /// Translates to `"required"` on Chat Completions.
    Any,
    /// Model must call this specific tool.
    Tool { name: String },
    /// Model is forbidden from calling any tool. Trailing underscore
    /// avoids collision with the keyword `None`.
    None_,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub tags: Vec<ToolTag>,
}

impl ToolDefinition {
    pub fn new(
        name: &str,
        description: &str,
        input_schema: serde_json::Value,
        tags: Vec<ToolTag>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            tags,
        }
    }
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub enum ToolTag {
    Destructive,
    ReadOnly,
    Network,
    WritesFiles,
    Custom(String),
}
