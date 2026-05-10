use ailoop_core::Message;
use serde::{Deserialize, Serialize};

use crate::errors::FromMessagesError;

/// Persistable image of a conversation's logical state.
///
/// Carries only the data needed to rebuild a `Conversation`'s history:
/// the message vector and the parallel `pinned` mask. Runtime state
/// (model, tools, middlewares, per-turn `Usage` / `RunId`) is
/// deliberately excluded — the model is re-supplied at resume time and
/// telemetry is the caller's concern.
///
/// Versioned via `version` so the on-disk shape can evolve without
/// breaking older payloads. Today only `version == 1` is accepted;
/// deserializing any other value fails with an explicit error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(try_from = "RawConversationSnapshot")]
pub struct ConversationSnapshot {
    pub version: u32,
    pub messages: Vec<Message>,
    pub pinned: Vec<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawConversationSnapshot {
    version: u32,
    messages: Vec<Message>,
    pinned: Vec<bool>,
}

impl ConversationSnapshot {
    pub const VERSION: u32 = 1;

    pub fn new(messages: Vec<Message>, pinned: Vec<bool>) -> Result<Self, FromMessagesError> {
        if messages.len() != pinned.len() {
            return Err(FromMessagesError::LengthMismatch {
                messages: messages.len(),
                pinned: pinned.len(),
            });
        }
        Ok(Self {
            version: Self::VERSION,
            messages,
            pinned,
        })
    }
}

impl TryFrom<RawConversationSnapshot> for ConversationSnapshot {
    type Error = String;

    fn try_from(raw: RawConversationSnapshot) -> Result<Self, Self::Error> {
        if raw.version != Self::VERSION {
            return Err(format!(
                "unsupported snapshot version {} (expected {})",
                raw.version,
                Self::VERSION
            ));
        }
        if raw.messages.len() != raw.pinned.len() {
            return Err(format!(
                "messages/pinned length mismatch: messages={}, pinned={}",
                raw.messages.len(),
                raw.pinned.len()
            ));
        }
        Ok(Self {
            version: raw.version,
            messages: raw.messages,
            pinned: raw.pinned,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{AssistantBlock, Message};
    use serde_json::json;

    #[test]
    fn round_trip_through_json() {
        let snap = ConversationSnapshot::new(
            vec![
                Message::user("hi"),
                Message::Assistant {
                    blocks: vec![AssistantBlock::tool_call("c1", "fetch", json!({"x": 1}))],
                },
            ],
            vec![true, false],
        )
        .expect("valid lengths");

        let s = serde_json::to_string(&snap).unwrap();
        let back: ConversationSnapshot = serde_json::from_str(&s).unwrap();
        assert_eq!(back, snap);
    }

    #[test]
    fn deserialize_rejects_unsupported_version() {
        let bad = json!({
            "version": 999,
            "messages": [],
            "pinned": []
        })
        .to_string();
        let err = serde_json::from_str::<ConversationSnapshot>(&bad)
            .expect_err("expected version mismatch error");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported snapshot version 999"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn deserialize_rejects_length_mismatch() {
        let bad = json!({
            "version": 1,
            "messages": [{ "User": { "blocks": [{ "Text": { "text": "hi" } }] } }],
            "pinned": []
        })
        .to_string();
        let err = serde_json::from_str::<ConversationSnapshot>(&bad)
            .expect_err("expected length mismatch error");
        assert!(
            err.to_string().contains("length mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn new_rejects_length_mismatch() {
        let err = ConversationSnapshot::new(vec![Message::user("hi")], vec![]).unwrap_err();
        assert!(matches!(
            err,
            FromMessagesError::LengthMismatch {
                messages: 1,
                pinned: 0
            }
        ));
    }
}
