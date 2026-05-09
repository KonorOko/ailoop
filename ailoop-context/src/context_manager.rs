use ailoop_core::Message;

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
    }

    pub fn estimated_tokens(&self) -> usize {
        self.estimator.estimate_context(&self.messages)
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn extend(&mut self, new_messages: Vec<Message>) {
        self.messages.extend(new_messages);
    }

    pub fn compact_if_needed(&mut self) -> Result<Option<CompactionReport>, CompactionError> {
        if self.estimated_tokens() < self.max_tokens {
            return Ok(None);
        }

        let before = self.messages.len();
        let compact_messages = self
            .strategy
            .compact(&self.messages, self.preserve_n_last)?;
        let after = compact_messages.len();
        let strategy = self.strategy.name();
        self.messages = compact_messages;
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

    pub fn strategy<C2>(self, strategy: Box<dyn CompactionStrategy>) -> ContextManagerBuilder {
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

    #[test]
    fn compact_if_needed_returns_none_when_under_budget() {
        let mut mgr = ContextManager::builder(10_000).build();
        mgr.add_message(Message::user("hi"));
        mgr.add_message(Message::assistant_text("hello"));

        let report = mgr.compact_if_needed().expect("compaction should succeed");
        assert!(report.is_none(), "no compaction expected when under budget");
    }

    #[test]
    fn compact_if_needed_returns_report_when_over_budget() {
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
            .expect("compaction should succeed")
            .expect("expected compaction to run");

        assert_eq!(report.strategy, "truncate");
        assert!(
            report.after < report.before,
            "compaction must drop messages"
        );
    }
}
