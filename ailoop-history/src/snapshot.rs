//! [`ConversationSnapshot`] — versioned wire format used by
//! [`HistoryStore`](crate::HistoryStore) backends.

use ailoop_core::{Message, Suspension};
use serde::{Deserialize, Serialize, Serializer};

use crate::errors::FromMessagesError;

/// Persistable image of a conversation's logical state.
///
/// Carries only the data needed to rebuild a `Conversation`'s history:
/// the message vector, the parallel `pinned` mask and the suspended
/// run, if any. Runtime state (model, tools, middlewares, per-turn
/// `Usage` / `RunId`) is deliberately excluded — the model is
/// re-supplied at resume time and telemetry is the caller's concern.
///
/// # Versions
///
/// Every payload carries a `version`. A snapshot is written with the
/// lowest version that can hold it, so payloads that older readers
/// understand stay readable by them:
///
/// - `1`: no suspension.
/// - `2`: carries a [`Suspension`]. A reader that ignored the field
///   would load a history ending on tool calls with no results, which
///   the next provider request rejects, so readers that predate it must
///   refuse the payload instead.
///
/// Reading accepts every version from [`MIN_VERSION`](Self::MIN_VERSION)
/// to [`VERSION`](Self::VERSION) (a version-1 payload with a suspension,
/// as written before version 2 existed, included) and rejects any other
/// at parse time, so callers fail loudly on a forward-incompatible file.
///
/// The version goes up when a reader that ignores a new field would
/// misread the payload; a field that can be ignored safely does not
/// raise it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(try_from = "RawConversationSnapshot")]
#[non_exhaustive]
pub struct ConversationSnapshot {
    /// Wire-format version: the one the payload was read with, or, for
    /// a snapshot built with [`ConversationSnapshot::new`], the one it
    /// will be written with. Serializing ignores this field and writes
    /// the lowest version that holds the contents (see "Versions"
    /// above), so a snapshot whose `suspension` was set directly is
    /// still written as version 2.
    pub version: u32,
    /// History messages, in chronological order.
    pub messages: Vec<Message>,
    /// Parallel pin mask — `pinned[i]` is the survivor flag for
    /// `messages[i]`. Same length as `messages`; the
    /// `#[serde(try_from)]` arm rejects payloads where the two
    /// vectors disagree, so post-deserialize the invariant holds.
    pub pinned: Vec<bool>,
    /// The conversation's suspended run, if it has one: the history
    /// then ends on the suspended step's tool calls, and this holds
    /// what resuming needs. Omitted from the serialized form when
    /// `None` (a version-1 payload); its presence makes the payload
    /// version 2.
    pub suspension: Option<Suspension>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawConversationSnapshot {
    version: u32,
    messages: Vec<Message>,
    pinned: Vec<bool>,
    #[serde(default)]
    suspension: Option<Suspension>,
}

/// The serialized form, borrowed from a [`ConversationSnapshot`] so
/// writing one does not copy the history.
#[derive(Serialize)]
struct RawConversationSnapshotRef<'a> {
    version: u32,
    messages: &'a [Message],
    pinned: &'a [bool],
    #[serde(skip_serializing_if = "Option::is_none")]
    suspension: Option<&'a Suspension>,
}

impl ConversationSnapshot {
    /// Newest wire-format version this build reads and writes. See
    /// "Versions" on [`ConversationSnapshot`].
    pub const VERSION: u32 = 2;

    /// Oldest wire-format version this build reads.
    pub const MIN_VERSION: u32 = 1;

    /// Bundle `messages` and `pinned` into a snapshot without a
    /// suspension, tagged version 1. Returns
    /// [`FromMessagesError::LengthMismatch`] when the two vectors
    /// disagree — same invariant the deserialize path enforces, so
    /// snapshots produced via this constructor always round-trip.
    pub fn new(messages: Vec<Message>, pinned: Vec<bool>) -> Result<Self, FromMessagesError> {
        if messages.len() != pinned.len() {
            return Err(FromMessagesError::LengthMismatch {
                messages: messages.len(),
                pinned: pinned.len(),
            });
        }
        Ok(Self {
            version: Self::MIN_VERSION,
            messages,
            pinned,
            suspension: None,
        })
    }

    /// Attach the conversation's suspended run, setting
    /// [`version`](Self::version) to match. See [`Self::suspension`].
    pub fn with_suspension(mut self, suspension: Option<Suspension>) -> Self {
        self.suspension = suspension;
        self.version = Self::version_for(self.suspension.as_ref());
        self
    }

    /// The lowest version that holds a snapshot with `suspension`.
    fn version_for(suspension: Option<&Suspension>) -> u32 {
        match suspension {
            Some(_) => 2,
            None => 1,
        }
    }
}

impl Serialize for ConversationSnapshot {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        RawConversationSnapshotRef {
            version: Self::version_for(self.suspension.as_ref()),
            messages: &self.messages,
            pinned: &self.pinned,
            suspension: self.suspension.as_ref(),
        }
        .serialize(serializer)
    }
}

impl TryFrom<RawConversationSnapshot> for ConversationSnapshot {
    type Error = String;

    fn try_from(raw: RawConversationSnapshot) -> Result<Self, Self::Error> {
        if !(Self::MIN_VERSION..=Self::VERSION).contains(&raw.version) {
            return Err(format!(
                "unsupported snapshot version {} (expected {} to {})",
                raw.version,
                Self::MIN_VERSION,
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
            suspension: raw.suspension,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{AssistantBlock, CallState, Message, RunId, StepId, SuspendedCall};
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

    fn suspension() -> Suspension {
        Suspension::new(
            RunId::new(),
            StepId::new(),
            vec![SuspendedCall::new(
                "c1",
                "fetch",
                json!({"x": 1}),
                CallState::Deferred {
                    reason: "needs approval".into(),
                },
            )],
        )
    }

    fn suspended() -> ConversationSnapshot {
        ConversationSnapshot::new(
            vec![
                Message::user("hi"),
                Message::Assistant {
                    blocks: vec![AssistantBlock::tool_call("c1", "fetch", json!({"x": 1}))],
                },
            ],
            vec![false, false],
        )
        .unwrap()
        .with_suspension(Some(suspension()))
    }

    /// A snapshot without a suspension serializes as version 1 without
    /// the field, so payloads from before it existed and after it are
    /// the same shape.
    #[test]
    fn no_suspension_keeps_the_v1_shape() {
        let snap = ConversationSnapshot::new(vec![Message::user("hi")], vec![false]).unwrap();
        assert_eq!(snap.version, 1);
        let value = serde_json::to_value(&snap).unwrap();
        assert_eq!(
            value,
            json!({"version": 1, "messages": value["messages"], "pinned": [false]})
        );
        let back: ConversationSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(back, snap);
        assert_eq!(back.suspension, None);
    }

    /// A reader that ignored the suspension would load a history ending
    /// on unanswered tool calls, so a snapshot with one is version 2.
    #[test]
    fn suspension_writes_version_2_and_round_trips() {
        let snap = suspended();
        assert_eq!(snap.version, 2);
        let value = serde_json::to_value(&snap).unwrap();
        assert_eq!(value["version"], json!(2));
        assert!(value.get("suspension").is_some());
        let back: ConversationSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(back, snap);
    }

    /// The written version follows the contents, not the public field.
    #[test]
    fn suspension_set_directly_still_writes_version_2() {
        let mut snap = ConversationSnapshot::new(vec![Message::user("hi")], vec![false]).unwrap();
        snap.suspension = Some(suspension());
        let value = serde_json::to_value(&snap).unwrap();
        assert_eq!(value["version"], json!(2));

        let cleared = suspended().with_suspension(None);
        assert_eq!(cleared.version, 1);
        assert_eq!(serde_json::to_value(&cleared).unwrap()["version"], json!(1));
    }

    /// Payloads written before version 2 existed carried the suspension
    /// under version 1; they stay readable.
    #[test]
    fn v1_with_suspension_is_accepted() {
        let snap = suspended();
        let mut value = serde_json::to_value(&snap).unwrap();
        value["version"] = json!(1);
        let back: ConversationSnapshot = serde_json::from_value(value).unwrap();
        assert_eq!(back.version, 1);
        assert_eq!(back.suspension, snap.suspension);
        assert_eq!(serde_json::to_value(&back).unwrap()["version"], json!(2));
    }

    #[test]
    fn deserialize_rejects_unsupported_version() {
        for version in [0, 3, 999] {
            let bad = json!({
                "version": version,
                "messages": [],
                "pinned": []
            })
            .to_string();
            let err = serde_json::from_str::<ConversationSnapshot>(&bad)
                .expect_err("expected version mismatch error");
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("unsupported snapshot version {version}")),
                "unexpected error message: {msg}"
            );
        }
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
