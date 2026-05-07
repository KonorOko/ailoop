use ailoop_core::Message;

use crate::errors::CompactionError;

pub trait CompactionStrategy {
    fn compact(
        &self,
        messages: &[Message],
        preserve_n_last: usize,
    ) -> Result<Vec<Message>, CompactionError>;
}

pub struct TruncateStrategy;

impl CompactionStrategy for TruncateStrategy {
    fn compact(
        &self,
        messages: &[Message],
        preserve_n_last: usize,
    ) -> Result<Vec<Message>, CompactionError> {
        if messages.len() <= preserve_n_last {
            return Err(CompactionError::NotEnoughHistory);
        }

        Ok(messages[messages.len() - preserve_n_last..].to_vec())
    }
}
