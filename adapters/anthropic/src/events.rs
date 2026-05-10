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

#[derive(Default, Deserialize)]
pub(crate) struct MessageStartUsage {
    pub input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Legacy flat counter. The API still emits it alongside the newer
    /// `cache_creation` object so callers that only read this field keep
    /// working; when the breakdown is present, this equals
    /// `ephemeral_5m_input_tokens + ephemeral_1h_input_tokens`.
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// New TTL-broken-down counter. `None` on older API versions.
    #[serde(default)]
    pub cache_creation: Option<CacheCreationBreakdown>,
    /// Provider service tier reported on this turn (`"standard"` /
    /// `"priority"` / `"batch"`). Useful for correlating latency with
    /// the tier billed.
    #[serde(default)]
    pub service_tier: Option<String>,
}

/// Cache creation tokens broken down by TTL bucket. Anthropic added
/// this object in late 2024 alongside the existing flat
/// `cache_creation_input_tokens` integer so consumers can attribute
/// cache writes to the 5m vs 1h ephemeral pool independently.
#[derive(Default, Deserialize)]
pub(crate) struct CacheCreationBreakdown {
    #[serde(default)]
    pub ephemeral_5m_input_tokens: u32,
    #[serde(default)]
    pub ephemeral_1h_input_tokens: u32,
}

#[derive(Deserialize)]
pub(crate) struct MessageDeltaPayload {
    pub stop_reason: Option<String>,
}

#[derive(Default, Deserialize)]
pub(crate) struct UsageDelta {
    pub output_tokens: u32,
    /// Anthropic may include cache fields in `message_delta.usage` on
    /// some plans / beta paths; keep them defaulted so older fixtures
    /// without these fields keep deserializing.
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_creation: Option<CacheCreationBreakdown>,
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

    /// Older fixtures (pre-`cache_creation` object, pre-`service_tier`)
    /// must still deserialize cleanly thanks to `#[serde(default)]` on
    /// the new fields.
    #[test]
    fn parses_legacy_message_start_usage_without_new_fields() {
        let json = r#"{"input_tokens": 100, "cache_read_input_tokens": 25, "cache_creation_input_tokens": 50}"#;
        let usage: MessageStartUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.cache_read_input_tokens, 25);
        assert_eq!(usage.cache_creation_input_tokens, 50);
        assert!(usage.cache_creation.is_none());
        assert!(usage.service_tier.is_none());
    }

    #[test]
    fn parses_message_start_usage_with_breakdown_and_tier() {
        let json = r#"{
            "input_tokens": 100,
            "cache_read_input_tokens": 25,
            "cache_creation_input_tokens": 150,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 100,
                "ephemeral_1h_input_tokens": 50
            },
            "service_tier": "priority"
        }"#;
        let usage: MessageStartUsage = serde_json::from_str(json).unwrap();
        let breakdown = usage.cache_creation.expect("breakdown should parse");
        assert_eq!(breakdown.ephemeral_5m_input_tokens, 100);
        assert_eq!(breakdown.ephemeral_1h_input_tokens, 50);
        assert_eq!(usage.service_tier.as_deref(), Some("priority"));
    }

    #[test]
    fn parses_usage_delta_with_cache_fields() {
        let json = r#"{
            "output_tokens": 42,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 3,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 3,
                "ephemeral_1h_input_tokens": 0
            }
        }"#;
        let delta: UsageDelta = serde_json::from_str(json).unwrap();
        assert_eq!(delta.output_tokens, 42);
        assert_eq!(delta.cache_read_input_tokens, 7);
        assert_eq!(delta.cache_creation_input_tokens, 3);
        let breakdown = delta.cache_creation.expect("breakdown should parse");
        assert_eq!(breakdown.ephemeral_5m_input_tokens, 3);
    }
}
