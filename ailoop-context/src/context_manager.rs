use ailoop_core::Message;

use crate::{
    compaction::{CompactionStrategy, TruncateStrategy},
    errors::CompactionError,
    tokens::{CharEstimator, TokenEstimator},
};

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

    pub fn compact_if_needed(&mut self) -> Result<(), CompactionError> {
        if self.estimated_tokens() < self.max_tokens {
            return Ok(());
        }

        let compact_messages = self
            .strategy
            .compact(&self.messages, self.preserve_n_last)?;
        self.messages = compact_messages;
        Ok(())
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
