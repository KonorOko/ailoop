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
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    /// Catches block types we don't know about yet (e.g. server_tool_use,
    /// web_search_tool_result) so deserialization doesn't panic. The stream
    /// handler ignores unknown blocks; if you depend on a beta block, model
    /// it explicitly.
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    /// Forward-compat for future delta types (e.g. citations_delta).
    #[serde(other)]
    Unknown,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_thinking_block_start() {
        let json = r#"{"type":"thinking","thinking":"","signature":""}"#;
        let block: AnthropicBlock = serde_json::from_str(json).unwrap();
        assert!(matches!(
            block,
            AnthropicBlock::Thinking { thinking, signature }
                if thinking.is_empty() && signature.is_empty()
        ));
    }

    #[test]
    fn parses_redacted_thinking_block() {
        let json = r#"{"type":"redacted_thinking","data":"opaque-payload"}"#;
        let block: AnthropicBlock = serde_json::from_str(json).unwrap();
        match block {
            AnthropicBlock::RedactedThinking { data } => assert_eq!(data, "opaque-payload"),
            _ => panic!("expected RedactedThinking"),
        }
    }

    #[test]
    fn parses_signature_delta() {
        let json = r#"{"type":"signature_delta","signature":"sig-abc"}"#;
        let delta: AnthropicDelta = serde_json::from_str(json).unwrap();
        match delta {
            AnthropicDelta::SignatureDelta { signature } => assert_eq!(signature, "sig-abc"),
            _ => panic!("expected SignatureDelta"),
        }
    }

    #[test]
    fn unknown_block_type_does_not_panic() {
        // server_tool_use is a real block type we don't model. Should fall
        // through to Unknown rather than fail deserialization.
        let json = r#"{"type":"server_tool_use","id":"x","name":"y","input":{}}"#;
        let block: AnthropicBlock = serde_json::from_str(json).unwrap();
        assert!(matches!(block, AnthropicBlock::Unknown));
    }

    #[test]
    fn unknown_delta_type_does_not_panic() {
        let json = r#"{"type":"citations_delta","citation":{}}"#;
        let delta: AnthropicDelta = serde_json::from_str(json).unwrap();
        assert!(matches!(delta, AnthropicDelta::Unknown));
    }
}
