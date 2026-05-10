use serde::Serialize;
use serde_json::Value;

use crate::{CacheControl, Message, SystemPrompt};

#[derive(Clone)]
#[non_exhaustive]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub system_prompt: Option<SystemPrompt>,
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

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: None,
            tools: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            max_tokens: 4096,
            tool_choice: None,
            disable_parallel_tool_use: None,
            additional_params: None,
        }
    }
}

impl ChatRequest {
    pub fn new(messages: Vec<Message>, max_tokens: u32) -> Self {
        Self {
            messages,
            max_tokens,
            ..Default::default()
        }
    }
}

/// Constraint placed on the model's tool selection for a single
/// request. Variant naming follows Anthropic's wire vocabulary; the
/// Chat Completions adapter translates `Any` → `"required"` and
/// `None_` → `"none"`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub tags: Vec<ToolTag>,
    /// Cache breakpoint for this tool entry on providers that support
    /// per-tool prompt caching (Anthropic). Adapters without per-tool
    /// caching ignore the field.
    #[serde(skip)]
    pub cache_control: Option<CacheControl>,
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
            cache_control: None,
        }
    }

    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolTag {
    Destructive,
    ReadOnly,
    Network,
    WritesFiles,
    Custom(String),
}
