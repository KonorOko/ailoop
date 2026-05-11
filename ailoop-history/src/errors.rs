//! Failure surfaces of compaction and snapshot validation.

use thiserror::Error;

/// Failure surface of [`CompactionStrategy::compact`] (and therefore of
/// [`History::compact_if_needed`]).
///
/// The façade [`Conversation::run`](https://docs.rs/ailoop) /
/// [`Conversation::stream`](https://docs.rs/ailoop) wraps this in
/// [`EngineError::Context`](https://docs.rs/ailoop) when compaction
/// fails mid-run.
///
/// [`CompactionStrategy::compact`]: crate::CompactionStrategy::compact
/// [`History::compact_if_needed`]: crate::History::compact_if_needed
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CompactionError {
    /// History has fewer messages than `preserve_n_last`, so there is
    /// nothing the strategy can drop. Indicates a configuration bug
    /// rather than a transient failure: raise the budget or lower
    /// [`HistoryBuilder::preserve_n_last`].
    ///
    /// [`HistoryBuilder::preserve_n_last`]: crate::HistoryBuilder::preserve_n_last
    #[error("Not enough history to compact")]
    NotEnoughHistory,

    /// A strategy that calls a [`ailoop_core::CompletionModel`]
    /// (notably [`SummarizeStrategy`](crate::SummarizeStrategy))
    /// failed to produce a summary. The original error is rendered
    /// as a string to keep [`CompactionError`] a non-generic,
    /// object-safe enum usable behind `Box<dyn CompactionStrategy>`.
    #[error("Summarization failed: {0}")]
    SummarizationFailed(String),
}

/// Returned by [`ConversationSnapshot::new`] and
/// [`History::from_messages`] when the supplied parallel
/// vectors are inconsistent. Surfaces at restore time so a malformed
/// snapshot fails loudly instead of corrupting state silently.
///
/// [`ConversationSnapshot::new`]: crate::ConversationSnapshot::new
/// [`History::from_messages`]: crate::History::from_messages
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum FromMessagesError {
    /// `messages` and `pinned` were not the same length. Both fields
    /// carry the observed lengths so callers can report the exact
    /// mismatch.
    #[error(
        "messages and pinned must have the same length (messages: {messages}, pinned: {pinned})"
    )]
    LengthMismatch {
        /// Observed length of the messages vector.
        messages: usize,
        /// Observed length of the pin mask.
        pinned: usize,
    },
}
