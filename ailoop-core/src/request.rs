use serde::Serialize;
use serde_json::Value;

use crate::Message;

pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub system_prompt: Option<String>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub temperature: Option<f32>,
    pub max_tokens: u32,

    pub aditional_params: Option<Value>,
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

#[derive(Debug, Serialize, Clone)]
pub enum ToolTag {
    Destructive,
    ReadOnly,
    Network,
    WritesFiles,
    Custom(String),
}
