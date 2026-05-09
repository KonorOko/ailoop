use ailoop_core::{AssistantBlock, Message, UserBlock};

use crate::{
    compaction::{CompactionStrategy, TruncateStrategy},
    errors::CompactionError,
    tokens::{CharEstimator, TokenEstimator},
};

/// Reports what `ContextManager::compact_if_needed` did when it ran.
/// Returned wrapped in `Option`: `None` means compaction was not needed
/// (history fits within `max_tokens`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionReport {
    pub before: usize,
    pub after: usize,
    pub strategy: &'static str,
}

pub struct ContextManager {
    messages: Vec<Message>,
    /// Parallel to `messages`: `pinned[i] == true` marks `messages[i]`
    /// as "must survive compaction". The two vectors are kept the same
    /// length by every internal mutation; new messages default to
    /// `false` and only the explicit pin API flips the flag.
    pinned: Vec<bool>,
    max_tokens: usize,
    preserve_n_last: usize,
    strategy: Box<dyn CompactionStrategy>,
    estimator: Box<dyn TokenEstimator>,
}

impl ContextManager {
    pub fn builder(max_tokens: usize) -> ContextManagerBuilder {
        ContextManagerBuilder::new(max_tokens)
    }
}

impl ContextManager {
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.pinned.push(false);
    }

    pub fn estimated_tokens(&self) -> usize {
        self.estimator.estimate_context(&self.messages)
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn extend(&mut self, new_messages: Vec<Message>) {
        let added = new_messages.len();
        self.messages.extend(new_messages);
        self.pinned.extend(std::iter::repeat_n(false, added));
    }

    /// Pinned-state slice, parallel to [`Self::messages`]. Useful for
    /// tests; production callers normally interact via the `pin_*`
    /// helpers and inspect [`Self::is_pinned`].
    pub fn pinned(&self) -> &[bool] {
        &self.pinned
    }

    pub fn is_pinned(&self, idx: usize) -> bool {
        self.pinned.get(idx).copied().unwrap_or(false)
    }

    /// Pin the most recently added message so it survives every future
    /// compaction. No-op when the history is empty.
    ///
    /// Indices are not stable across compactions: a `pin_last()` made
    /// before compaction stays pinned (the strategy keeps it), but its
    /// numeric index in the new history may shift. Re-derive indices
    /// (or rely on `pin_last`) after compaction runs.
    pub fn pin_last(&mut self) {
        if let Some(last) = self.pinned.last_mut() {
            *last = true;
        }
    }

    /// Pin the message at `idx`. Panics on out-of-bounds, matching the
    /// convention of `Vec::index`.
    pub fn pin_at(&mut self, idx: usize) {
        assert!(idx < self.messages.len(), "pin_at: index {idx} out of bounds");
        self.pinned[idx] = true;
    }

    /// Clear the pin on the message at `idx`. Panics on out-of-bounds.
    pub fn unpin_at(&mut self, idx: usize) {
        assert!(idx < self.messages.len(), "unpin_at: index {idx} out of bounds");
        self.pinned[idx] = false;
    }

    /// Pin `idx` together with every message that pairs with it via a
    /// `tool_call` ↔ `tool_result` link. Without this, pinning a lone
    /// `Assistant` `ToolCall` (or a lone `User` `ToolResult`) and then
    /// letting compaction run would strand the partner — most providers
    /// reject that as a malformed history.
    ///
    /// Resolution rules:
    /// - If `messages[idx]` is an `Assistant` with `ToolCall` blocks,
    ///   any `User` message containing a `ToolResult` whose `call_id`
    ///   matches one of those calls is also pinned.
    /// - If `messages[idx]` is a `User` with `ToolResult` blocks, any
    ///   `Assistant` message containing a `ToolCall` whose `id` matches
    ///   is also pinned.
    /// - Messages without tool blocks are pinned alone (effectively a
    ///   `pin_at`).
    ///
    /// Panics on out-of-bounds. The lookup scans the full history but
    /// each message is examined at most once.
    pub fn pin_with_tool_result(&mut self, idx: usize) {
        assert!(
            idx < self.messages.len(),
            "pin_with_tool_result: index {idx} out of bounds"
        );

        self.pinned[idx] = true;

        let target_ids: Vec<String> = match &self.messages[idx] {
            Message::Assistant { blocks } => blocks
                .iter()
                .filter_map(|b| match b {
                    AssistantBlock::ToolCall { id, .. } => Some(id.clone()),
                    _ => None,
                })
                .collect(),
            Message::User { blocks } => blocks
                .iter()
                .filter_map(|b| match b {
                    UserBlock::ToolResult { call_id, .. } => Some(call_id.clone()),
                    _ => None,
                })
                .collect(),
        };

        if target_ids.is_empty() {
            return;
        }

        let is_assistant_target = matches!(self.messages[idx], Message::Assistant { .. });

        for (i, msg) in self.messages.iter().enumerate() {
            if i == idx || self.pinned[i] {
                continue;
            }
            let matches = match (is_assistant_target, msg) {
                (true, Message::User { blocks }) => blocks.iter().any(|b| matches!(b,
                    UserBlock::ToolResult { call_id, .. } if target_ids.iter().any(|t| t == call_id))),
                (false, Message::Assistant { blocks }) => blocks.iter().any(|b| matches!(b,
                    AssistantBlock::ToolCall { id, .. } if target_ids.iter().any(|t| t == id))),
                _ => false,
            };
            if matches {
                self.pinned[i] = true;
            }
        }
    }

    pub async fn compact_if_needed(
        &mut self,
    ) -> Result<Option<CompactionReport>, CompactionError> {
        if self.estimated_tokens() < self.max_tokens {
            return Ok(None);
        }

        let before = self.messages.len();
        let output = self
            .strategy
            .compact(&self.messages, &self.pinned, self.preserve_n_last)
            .await?;
        debug_assert_eq!(
            output.messages.len(),
            output.pinned.len(),
            "strategy must return a pinned mask matching the message vector",
        );
        let after = output.messages.len();
        let strategy = self.strategy.name();
        self.messages = output.messages;
        self.pinned = output.pinned;
        Ok(Some(CompactionReport {
            before,
            after,
            strategy,
        }))
    }
}

pub struct ContextManagerBuilder {
    max_tokens: usize,
    preserve_n_last: usize,
    estimator: Box<dyn TokenEstimator>,
    strategy: Box<dyn CompactionStrategy>,
}

impl ContextManagerBuilder {
    fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            preserve_n_last: 4,
            estimator: Box::new(CharEstimator),
            strategy: Box::new(TruncateStrategy),
        }
    }
}

impl ContextManagerBuilder {
    pub fn preserve_n_last(mut self, n: usize) -> Self {
        self.preserve_n_last = n;
        self
    }

    pub fn estimator(self, estimator: Box<dyn TokenEstimator>) -> ContextManagerBuilder {
        ContextManagerBuilder {
            max_tokens: self.max_tokens,
            preserve_n_last: self.preserve_n_last,
            estimator,
            strategy: self.strategy,
        }
    }

    pub fn strategy(self, strategy: Box<dyn CompactionStrategy>) -> ContextManagerBuilder {
        ContextManagerBuilder {
            max_tokens: self.max_tokens,
            preserve_n_last: self.preserve_n_last,
            estimator: self.estimator,
            strategy,
        }
    }

    pub fn build(self) -> ContextManager {
        ContextManager {
            messages: Vec::new(),
            pinned: Vec::new(),
            max_tokens: self.max_tokens,
            preserve_n_last: self.preserve_n_last,
            strategy: self.strategy,
            estimator: self.estimator,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{AssistantBlock, ToolResultContent, UserBlock};
    use serde_json::json;

    fn tool_call_msg(id: &str) -> Message {
        Message::Assistant {
            blocks: vec![AssistantBlock::tool_call(id, "t", json!({}))],
        }
    }

    fn tool_result_msg(call_id: &str) -> Message {
        Message::User {
            blocks: vec![UserBlock::tool_result(
                call_id,
                ToolResultContent::Text("ok".into()),
            )],
        }
    }

    #[tokio::test]
    async fn compact_if_needed_returns_none_when_under_budget() {
        let mut mgr = ContextManager::builder(10_000).build();
        mgr.add_message(Message::user("hi"));
        mgr.add_message(Message::assistant_text("hello"));

        let report = mgr
            .compact_if_needed()
            .await
            .expect("compaction should succeed");
        assert!(report.is_none(), "no compaction expected when under budget");
    }

    #[tokio::test]
    async fn compact_if_needed_returns_report_when_over_budget() {
        // CharEstimator is len()/4. Use a tiny budget so a couple of
        // small messages already trip the limit.
        let mut mgr = ContextManager::builder(10).preserve_n_last(2).build();
        mgr.add_message(Message::user("first turn"));
        mgr.add_message(Message::assistant_text("first reply"));
        mgr.add_message(Message::user("second turn"));
        mgr.add_message(Message::assistant_text("second reply"));
        mgr.add_message(Message::user("third turn"));

        let report = mgr
            .compact_if_needed()
            .await
            .expect("compaction should succeed")
            .expect("expected compaction to run");

        assert_eq!(report.strategy, "truncate");
        assert!(
            report.after < report.before,
            "compaction must drop messages"
        );
    }

    #[tokio::test]
    async fn pin_last_survives_compaction() {
        let mut mgr = ContextManager::builder(10).preserve_n_last(2).build();
        mgr.add_message(Message::user("pinned anchor"));
        mgr.pin_last();
        for i in 0..5 {
            mgr.add_message(Message::user(format!("turn {i} q")));
            mgr.add_message(Message::assistant_text(format!("turn {i} a")));
        }

        let report = mgr
            .compact_if_needed()
            .await
            .expect("compaction should succeed")
            .expect("expected compaction to run");
        assert!(report.after < report.before);

        // The pinned anchor must still be the first message and still pinned.
        let first = mgr.messages().first().expect("history should be non-empty");
        match first {
            Message::User { blocks } => match &blocks[0] {
                UserBlock::Text { text, .. } => assert_eq!(text, "pinned anchor"),
                other => panic!("expected pinned text block, got {other:?}"),
            },
            other => panic!("expected pinned user message, got {other:?}"),
        }
        assert!(mgr.is_pinned(0), "pinned mask must be preserved across compaction");
    }

    #[tokio::test]
    async fn pin_with_tool_result_keeps_pair_intact() {
        let mut mgr = ContextManager::builder(10).preserve_n_last(2).build();
        mgr.add_message(Message::user("task"));
        mgr.add_message(tool_call_msg("c1"));
        mgr.add_message(tool_result_msg("c1"));
        mgr.add_message(Message::assistant_text("result"));
        // Pin the tool_call (idx 1). The helper should also pin the
        // tool_result (idx 2) so the pair survives compaction together.
        mgr.pin_with_tool_result(1);
        assert!(mgr.is_pinned(1));
        assert!(mgr.is_pinned(2), "partner tool_result must be pinned too");

        // Add filler so we overflow the budget.
        for i in 0..6 {
            mgr.add_message(Message::user(format!("filler {i}")));
            mgr.add_message(Message::assistant_text(format!("ack {i}")));
        }

        mgr.compact_if_needed()
            .await
            .expect("compaction should succeed")
            .expect("expected compaction to run");

        // The pinned pair should still be present in the same relative order.
        let mut saw_call = false;
        let mut saw_result = false;
        for msg in mgr.messages() {
            match msg {
                Message::Assistant { blocks } => {
                    if blocks
                        .iter()
                        .any(|b| matches!(b, AssistantBlock::ToolCall { id, .. } if id == "c1"))
                    {
                        saw_call = true;
                    }
                }
                Message::User { blocks } => {
                    if blocks.iter().any(
                        |b| matches!(b, UserBlock::ToolResult { call_id, .. } if call_id == "c1"),
                    ) {
                        assert!(saw_call, "tool_result must follow its tool_call");
                        saw_result = true;
                    }
                }
            }
        }
        assert!(saw_call && saw_result, "pinned pair must survive compaction");
    }

    #[tokio::test]
    async fn pin_with_tool_result_on_result_pins_the_call() {
        let mut mgr = ContextManager::builder(10).preserve_n_last(1).build();
        mgr.add_message(tool_call_msg("c1"));
        mgr.add_message(tool_result_msg("c1"));
        mgr.add_message(Message::user("later"));

        mgr.pin_with_tool_result(1);
        assert!(mgr.is_pinned(0), "tool_call partner must be pinned");
        assert!(mgr.is_pinned(1));
    }
}
