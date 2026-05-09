use ailoop_core::{Message, UserBlock};

use crate::errors::CompactionError;

pub trait CompactionStrategy: Send + Sync {
    /// Stable, machine-readable name of the strategy. Used by
    /// `HistoryCompacted` events so callers can attribute compaction
    /// to a specific algorithm in logs/metrics.
    fn name(&self) -> &'static str;

    fn compact(
        &self,
        messages: &[Message],
        preserve_n_last: usize,
    ) -> Result<Vec<Message>, CompactionError>;
}

pub struct TruncateStrategy;

impl CompactionStrategy for TruncateStrategy {
    fn name(&self) -> &'static str {
        "truncate"
    }

    fn compact(
        &self,
        messages: &[Message],
        preserve_n_last: usize,
    ) -> Result<Vec<Message>, CompactionError> {
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

        Ok(messages[start..].to_vec())
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

    #[test]
    fn keeps_normal_history_intact_when_no_pairs() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant_text("hello"),
            Message::user("again"),
            Message::assistant_text("yes"),
        ];

        let out = TruncateStrategy.compact(&messages, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], Message::User { .. }));
    }

    #[test]
    fn walks_back_when_cut_lands_on_tool_result() {
        let messages = vec![
            Message::user("solve this"),
            tool_call("c1"),
            tool_result("c1"),
            Message::assistant_text("done"),
        ];

        let out = TruncateStrategy.compact(&messages, 2).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn walks_back_when_cut_lands_on_assistant() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant_text("hey"),
            Message::user("more"),
            Message::assistant_text("done"),
        ];

        let out = TruncateStrategy.compact(&messages, 1).unwrap();
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], Message::User { .. }));
    }
}
