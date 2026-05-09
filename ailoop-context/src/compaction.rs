use ailoop_core::{Message, UserBlock};
use async_trait::async_trait;

use crate::errors::CompactionError;

/// Result of a successful [`CompactionStrategy::compact`] call.
///
/// `messages` and `pinned` are parallel: `pinned[i]` describes the
/// pin state of `messages[i]` in the post-compaction history. The
/// strategy is responsible for forwarding the pin state of every
/// message it preserves so the [`crate::ContextManager`] can keep its
/// internal mask consistent across compactions.
#[derive(Debug, Clone)]
pub struct CompactionOutput {
    pub messages: Vec<Message>,
    pub pinned: Vec<bool>,
}

#[async_trait]
pub trait CompactionStrategy: Send + Sync {
    /// Stable, machine-readable name of the strategy. Used by
    /// `HistoryCompacted` events so callers can attribute compaction
    /// to a specific algorithm in logs/metrics.
    fn name(&self) -> &'static str;

    /// Compact `messages` into a smaller history.
    ///
    /// `pinned` is a parallel slice of the same length as `messages`:
    /// `pinned[i] == true` marks `messages[i]` as "must survive". A
    /// strategy must include every pinned message in its output (in
    /// the original relative order) and forward its `true` pin state
    /// in the returned [`CompactionOutput::pinned`].
    ///
    /// `preserve_n_last` is a hint: at minimum the last N messages
    /// (after walking back to a safe boundary that doesn't strand a
    /// `ToolResult` from its `ToolCall`) should be kept verbatim.
    async fn compact(
        &self,
        messages: &[Message],
        pinned: &[bool],
        preserve_n_last: usize,
    ) -> Result<CompactionOutput, CompactionError>;
}

pub struct TruncateStrategy;

#[async_trait]
impl CompactionStrategy for TruncateStrategy {
    fn name(&self) -> &'static str {
        "truncate"
    }

    async fn compact(
        &self,
        messages: &[Message],
        pinned: &[bool],
        preserve_n_last: usize,
    ) -> Result<CompactionOutput, CompactionError> {
        if messages.len() <= preserve_n_last {
            return Err(CompactionError::NotEnoughHistory);
        }

        let mut start = messages.len() - preserve_n_last;

        // Walk the cut backwards until messages[start] is a safe boundary:
        // a User message whose blocks contain no ToolResult. Otherwise we'd
        // strand a ToolResult from its corresponding ToolCall in the
        // Assistant message we're about to drop, which the provider rejects.
        while start > 0 && !is_safe_start(&messages[start]) {
            start -= 1;
        }

        let mut out_messages = Vec::with_capacity(messages.len());
        let mut out_pinned = Vec::with_capacity(messages.len());

        // Pinned messages from the dropped prefix survive at their
        // original relative position. The caller is responsible for
        // pinning ToolCall/ToolResult pairs together — see
        // `ContextManager::pin_with_tool_result`.
        for (i, msg) in messages.iter().enumerate().take(start) {
            if pinned[i] {
                out_messages.push(msg.clone());
                out_pinned.push(true);
            }
        }

        for (i, msg) in messages.iter().enumerate().skip(start) {
            out_messages.push(msg.clone());
            out_pinned.push(pinned[i]);
        }

        Ok(CompactionOutput {
            messages: out_messages,
            pinned: out_pinned,
        })
    }
}

fn is_safe_start(msg: &Message) -> bool {
    match msg {
        Message::User { blocks } => !blocks
            .iter()
            .any(|b| matches!(b, UserBlock::ToolResult { .. })),
        Message::Assistant { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{AssistantBlock, ToolResultContent};
    use serde_json::json;

    fn tool_call(id: &str) -> Message {
        Message::Assistant {
            blocks: vec![AssistantBlock::tool_call(id, "t", json!({}))],
        }
    }

    fn tool_result(call_id: &str) -> Message {
        Message::User {
            blocks: vec![UserBlock::tool_result(
                call_id,
                ToolResultContent::Text("ok".into()),
            )],
        }
    }

    fn unpinned(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    #[tokio::test]
    async fn keeps_normal_history_intact_when_no_pairs() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant_text("hello"),
            Message::user("again"),
            Message::assistant_text("yes"),
        ];

        let out = TruncateStrategy
            .compact(&messages, &unpinned(messages.len()), 2)
            .await
            .unwrap();
        assert_eq!(out.messages.len(), 2);
        assert!(matches!(out.messages[0], Message::User { .. }));
        assert_eq!(out.pinned, vec![false, false]);
    }

    #[tokio::test]
    async fn walks_back_when_cut_lands_on_tool_result() {
        let messages = vec![
            Message::user("solve this"),
            tool_call("c1"),
            tool_result("c1"),
            Message::assistant_text("done"),
        ];

        let out = TruncateStrategy
            .compact(&messages, &unpinned(messages.len()), 2)
            .await
            .unwrap();
        assert_eq!(out.messages.len(), 4);
    }

    #[tokio::test]
    async fn walks_back_when_cut_lands_on_assistant() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant_text("hey"),
            Message::user("more"),
            Message::assistant_text("done"),
        ];

        let out = TruncateStrategy
            .compact(&messages, &unpinned(messages.len()), 1)
            .await
            .unwrap();
        assert_eq!(out.messages.len(), 2);
        assert!(matches!(out.messages[0], Message::User { .. }));
    }

    #[tokio::test]
    async fn pinned_prefix_message_survives_truncation() {
        let messages = vec![
            Message::user("system-ish pinned"),
            Message::user("turn 1 q"),
            Message::assistant_text("turn 1 a"),
            Message::user("turn 2 q"),
            Message::assistant_text("turn 2 a"),
        ];
        let mut pinned = unpinned(messages.len());
        pinned[0] = true;

        let out = TruncateStrategy
            .compact(&messages, &pinned, 2)
            .await
            .unwrap();

        assert_eq!(out.messages.len(), 3, "pinned prefix + tail of 2");
        assert!(matches!(&out.messages[0], Message::User { blocks }
            if matches!(&blocks[0], UserBlock::Text { text, .. } if text == "system-ish pinned")));
        assert_eq!(out.pinned, vec![true, false, false]);
    }
}
