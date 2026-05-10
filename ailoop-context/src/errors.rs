use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CompactionError {
    #[error("Not enough history to compact")]
    NotEnoughHistory,

    #[error("Budget exceeded")]
    BudgetExceeded,

    /// A strategy that calls a [`ailoop_core::CompletionModel`]
    /// (notably [`crate::compaction::SummarizeStrategy`]) failed to
    /// produce a summary. The original error is rendered as a string
    /// to keep [`CompactionError`] a non-generic, object-safe enum
    /// usable behind `Box<dyn CompactionStrategy>`.
    #[error("Summarization failed: {0}")]
    SummarizationFailed(String),
}

#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FromMessagesError {
    #[error(
        "messages and pinned must have the same length (messages: {messages}, pinned: {pinned})"
    )]
    LengthMismatch { messages: usize, pinned: usize },
}
