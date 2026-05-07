use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicEvent {
    MessageStart {
        message: MessageStartPayload,
    },
    ContentBlockStart {
        index: u32,
        content_block: AnthropicBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: AnthropicDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaPayload,
        usage: UsageDelta,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicErrorPayload,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    // Thinking {
    //     thinking: String,
    // },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    // SignatureDelta { signature: String },
}

#[derive(Deserialize)]
pub(crate) struct MessageStartPayload {
    pub usage: MessageStartUsage,
}

#[derive(Deserialize)]
pub(crate) struct MessageStartUsage {
    pub input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
}

#[derive(Deserialize)]
pub(crate) struct MessageDeltaPayload {
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct UsageDelta {
    pub output_tokens: u32,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicErrorPayload {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}
