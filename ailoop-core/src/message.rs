use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub enum Message {
    User { blocks: Vec<UserBlock> },
    Assistant { blocks: Vec<AssistantBlock> },
}

impl Message {
    pub fn user(text: impl Into<String>) -> Message {
        Message::User {
            blocks: vec![UserBlock::Text(text.into())],
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Message {
        Message::Assistant {
            blocks: vec![AssistantBlock::Text(text.into())],
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum UserBlock {
    Text(String),
    ToolResult {
        call_id: String,
        content: ToolResultContent,
    },
}

#[derive(Debug, Clone, Serialize)]
pub enum AssistantBlock {
    Text(String),
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    /// Visible reasoning emitted by the model. `signature` is provider-issued
    /// material that must be replayed verbatim on subsequent turns when tools
    /// are involved (Anthropic extended thinking). Providers without a
    /// signature concept (e.g. OpenAI reasoning) leave it `None`.
    Reasoning {
        text: String,
        signature: Option<String>,
    },
    /// Opaque reasoning block whose content the provider chose to hide.
    /// `data` is verbatim provider material — store it untouched and replay
    /// it back when the next request continues a tool-use chain.
    RedactedReasoning {
        data: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub enum ToolResultContent {
    Text(String),
    Error(String),
}

impl From<String> for ToolResultContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ToolResultContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}
