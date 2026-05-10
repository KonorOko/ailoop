use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Cache breakpoint placed on a content block, system prompt block, or
/// tool definition. Providers that support prompt caching (Anthropic
/// today) read these to decide which prefix is cacheable and at what
/// TTL; providers without explicit caching ignore the field. The
/// presence of `cache_control` only declares intent — the actual cache
/// hit/miss is reported via [`crate::Usage::cached_input_tokens`] and
/// the cache-creation counters.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheControl {
    /// Ephemeral cache with the provider's default TTL (5 minutes on
    /// Anthropic). Equivalent to omitting `ttl` on the wire.
    Ephemeral,
    /// Ephemeral cache with an explicit TTL. Anthropic accepts only
    /// `5m` and `1h`; the adapter rounds to the nearest supported value
    /// and warns if neither fits cleanly.
    EphemeralWithTtl(Duration),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Message {
    User { blocks: Vec<UserBlock> },
    Assistant { blocks: Vec<AssistantBlock> },
}

impl Message {
    pub fn user(text: impl Into<String>) -> Message {
        Message::User {
            blocks: vec![UserBlock::text(text)],
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Message {
        Message::Assistant {
            blocks: vec![AssistantBlock::text(text)],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UserBlock {
    Text {
        text: String,
        /// Per-request hint; not part of persisted conversation state.
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
    ToolResult {
        call_id: String,
        content: ToolResultContent,
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
}

impl UserBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<ToolResultContent>) -> Self {
        Self::ToolResult {
            call_id: call_id.into(),
            content: content.into(),
            cache_control: None,
        }
    }

    /// Builder-style helper: set or replace the cache breakpoint on this
    /// block. Use `None` to clear.
    pub fn with_cache_control(mut self, cache_control: Option<CacheControl>) -> Self {
        match &mut self {
            Self::Text {
                cache_control: cc, ..
            } => *cc = cache_control,
            Self::ToolResult {
                cache_control: cc, ..
            } => *cc = cache_control,
        }
        self
    }

    pub fn cache_control(&self) -> Option<&CacheControl> {
        match self {
            Self::Text { cache_control, .. } | Self::ToolResult { cache_control, .. } => {
                cache_control.as_ref()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AssistantBlock {
    Text {
        text: String,
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
    /// Visible reasoning emitted by the model. `signature` is provider-issued
    /// material that must be replayed verbatim on subsequent turns when tools
    /// are involved (Anthropic extended thinking). Providers without a
    /// signature concept (e.g. OpenAI reasoning) leave it `None`.
    ///
    /// Reasoning blocks intentionally have no `cache_control` slot:
    /// Anthropic does not accept the field on `thinking` /
    /// `redacted_thinking` blocks. Place breakpoints on adjacent text or
    /// tool blocks instead.
    Reasoning {
        text: String,
        signature: Option<String>,
    },
    /// Opaque reasoning block whose content the provider chose to hide.
    /// `data` is verbatim provider material — store it untouched and replay
    /// it back when the next request continues a tool-use chain.
    RedactedReasoning { data: String },
}

impl AssistantBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        args: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            args,
            cache_control: None,
        }
    }

    /// Builder-style helper: set or replace the cache breakpoint on this
    /// block. No-op for reasoning variants (they do not carry cache
    /// breakpoints on the wire).
    pub fn with_cache_control(mut self, cache_control: Option<CacheControl>) -> Self {
        match &mut self {
            Self::Text {
                cache_control: cc, ..
            } => *cc = cache_control,
            Self::ToolCall {
                cache_control: cc, ..
            } => *cc = cache_control,
            Self::Reasoning { .. } | Self::RedactedReasoning { .. } => {}
        }
        self
    }

    pub fn cache_control(&self) -> Option<&CacheControl> {
        match self {
            Self::Text { cache_control, .. } | Self::ToolCall { cache_control, .. } => {
                cache_control.as_ref()
            }
            Self::Reasoning { .. } | Self::RedactedReasoning { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
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

/// System prompt passed to the provider. `Plain(String)` matches the
/// pre-caching API and is what `From<String>` / `From<&str>` produce —
/// callers that don't care about prompt caching keep using strings.
/// `Blocks(...)` opts in to per-block cache breakpoints (Anthropic emits
/// the `system` field as an array; other providers concatenate the
/// block texts).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SystemPrompt {
    Plain(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SystemBlock {
    pub text: String,
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache_control: None,
        }
    }

    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

impl From<String> for SystemPrompt {
    fn from(value: String) -> Self {
        Self::Plain(value)
    }
}

impl From<&str> for SystemPrompt {
    fn from(value: &str) -> Self {
        Self::Plain(value.to_string())
    }
}

impl SystemPrompt {
    /// Concatenate all blocks into a single string. Used by adapters
    /// without per-block caching (Chat Completions) and as a debug aid.
    pub fn as_text(&self) -> String {
        match self {
            Self::Plain(s) => s.clone(),
            Self::Blocks(bs) => bs
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip(msg: &Message) -> Message {
        let json = serde_json::to_string(msg).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    fn assert_user_text_eq(msg: &Message, expected: &str) {
        match msg {
            Message::User { blocks } => match &blocks[0] {
                UserBlock::Text {
                    text,
                    cache_control,
                } => {
                    assert_eq!(text, expected);
                    assert!(cache_control.is_none(), "cache_control must not round-trip");
                }
                other => panic!("expected UserBlock::Text, got {other:?}"),
            },
            other => panic!("expected Message::User, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_user_text_drops_cache_control() {
        let msg = Message::User {
            blocks: vec![UserBlock::text("hello").with_cache_control(Some(CacheControl::Ephemeral))],
        };
        let restored = round_trip(&msg);
        assert_user_text_eq(&restored, "hello");
    }

    #[test]
    fn round_trip_user_tool_result() {
        let msg = Message::User {
            blocks: vec![UserBlock::tool_result(
                "call-1",
                ToolResultContent::Text("ok".into()),
            )],
        };
        let restored = round_trip(&msg);
        match &restored {
            Message::User { blocks } => match &blocks[0] {
                UserBlock::ToolResult {
                    call_id,
                    content,
                    cache_control,
                } => {
                    assert_eq!(call_id, "call-1");
                    assert!(matches!(content, ToolResultContent::Text(t) if t == "ok"));
                    assert!(cache_control.is_none());
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_assistant_text_and_tool_call() {
        let msg = Message::Assistant {
            blocks: vec![
                AssistantBlock::text("thinking out loud"),
                AssistantBlock::tool_call("c1", "fetch", json!({"q": "x"})),
            ],
        };
        let restored = round_trip(&msg);
        match &restored {
            Message::Assistant { blocks } => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    AssistantBlock::Text { text, .. } => assert_eq!(text, "thinking out loud"),
                    other => panic!("expected Text, got {other:?}"),
                }
                match &blocks[1] {
                    AssistantBlock::ToolCall { id, name, args, .. } => {
                        assert_eq!(id, "c1");
                        assert_eq!(name, "fetch");
                        assert_eq!(args, &json!({"q": "x"}));
                    }
                    other => panic!("expected ToolCall, got {other:?}"),
                }
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_assistant_reasoning_variants() {
        let msg = Message::Assistant {
            blocks: vec![
                AssistantBlock::Reasoning {
                    text: "consider X".into(),
                    signature: Some("sig-1".into()),
                },
                AssistantBlock::RedactedReasoning {
                    data: "opaque".into(),
                },
            ],
        };
        let restored = round_trip(&msg);
        match &restored {
            Message::Assistant { blocks } => match (&blocks[0], &blocks[1]) {
                (
                    AssistantBlock::Reasoning { text, signature },
                    AssistantBlock::RedactedReasoning { data },
                ) => {
                    assert_eq!(text, "consider X");
                    assert_eq!(signature.as_deref(), Some("sig-1"));
                    assert_eq!(data, "opaque");
                }
                other => panic!("unexpected blocks: {other:?}"),
            },
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_tool_result_error_variant() {
        let content = ToolResultContent::Error("boom".into());
        let json = serde_json::to_string(&content).unwrap();
        let back: ToolResultContent = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, ToolResultContent::Error(s) if s == "boom"));
    }
}
