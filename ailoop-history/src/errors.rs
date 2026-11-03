//! Failure surfaces of compaction and snapshot validation.

use std::error::Error as StdError;

use ailoop_core::{ProviderError, Usage};
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
    /// failed to produce a summary.
    ///
    /// [`CompactionError`] stays non-generic so it works behind
    /// `Box<dyn CompactionStrategy>`, so the model's error is boxed.
    /// Read it through [`std::error::Error::source`] and
    /// `downcast_ref`. Use [`CompactionError::is_context_overflow`] to
    /// tell whether the summary request itself was too large, since
    /// [`ProviderError`] cannot be queried through a boxed error.
    ///
    /// Build it with [`CompactionError::summarization_failed_from`] or
    /// [`CompactionError::summarization_failed`].
    #[error("Summarization failed: {message}")]
    #[non_exhaustive]
    SummarizationFailed {
        /// Description of the failure. For a model error, its
        /// `Display` text.
        message: String,
        /// `true` when the model rejected the summary request because
        /// it did not fit the context window.
        context_overflow: bool,
        /// The original error, when there is one. `None` for a failure
        /// that no error caused, such as a summary with no text.
        source: Option<Box<dyn StdError + Send + Sync>>,
        /// What the strategy's model requests spent before the failure.
        /// Zero when none reported usage. Read it with
        /// [`CompactionError::usage`] and set it with
        /// [`CompactionError::with_usage`].
        usage: Usage,
    },
}

impl CompactionError {
    /// Build a [`CompactionError::SummarizationFailed`] from each of
    /// its parts. For a strategy whose failure is not a model error, or
    /// whose error does not implement [`ProviderError`]; otherwise
    /// prefer [`CompactionError::summarization_failed_from`]. Its usage
    /// is zero until [`CompactionError::with_usage`] sets it.
    pub fn summarization_failed(
        message: impl Into<String>,
        context_overflow: bool,
        source: Option<Box<dyn StdError + Send + Sync>>,
    ) -> Self {
        Self::SummarizationFailed {
            message: message.into(),
            context_overflow,
            source,
            usage: Usage::default(),
        }
    }

    /// Build a [`CompactionError::SummarizationFailed`] from a model
    /// error: its `Display` text becomes the message, its
    /// [`ProviderError::is_context_overflow`] the overflow flag, and
    /// the error itself the source. Its usage is zero until
    /// [`CompactionError::with_usage`] sets it.
    pub fn summarization_failed_from<E>(error: E) -> Self
    where
        E: StdError + ProviderError + Send + Sync + 'static,
    {
        Self::SummarizationFailed {
            message: error.to_string(),
            context_overflow: error.is_context_overflow(),
            source: Some(Box::new(error)),
            usage: Usage::default(),
        }
    }

    /// Set what a [`CompactionError::SummarizationFailed`] spent before
    /// failing, replacing any earlier value. Other variants spend
    /// nothing and are returned unchanged.
    pub fn with_usage(mut self, spent: Usage) -> Self {
        if let Self::SummarizationFailed { usage, .. } = &mut self {
            *usage = spent;
        }
        self
    }

    /// What the compaction spent before failing, which the provider may
    /// bill even though no summary came out. Zero for every variant
    /// other than [`CompactionError::SummarizationFailed`].
    pub fn usage(&self) -> Usage {
        match self {
            Self::SummarizationFailed { usage, .. } => *usage,
            _ => Usage::default(),
        }
    }

    /// `true` when a [`CompactionError::SummarizationFailed`] was caused
    /// by the summary request not fitting the context window. Resending
    /// the same request fails the same way; summarizing less at a time
    /// may succeed. `false` for every other error.
    pub fn is_context_overflow(&self) -> bool {
        matches!(
            self,
            Self::SummarizationFailed {
                context_overflow: true,
                ..
            }
        )
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    /// Classified by its type alone: its text never says "overflow".
    #[derive(Debug)]
    struct RejectedError {
        overflow: bool,
    }

    impl fmt::Display for RejectedError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("request rejected")
        }
    }

    impl StdError for RejectedError {}

    impl ProviderError for RejectedError {
        fn is_context_overflow(&self) -> bool {
            self.overflow
        }
    }

    #[test]
    fn summarization_failed_from_takes_the_flag_from_the_provider_error() {
        for overflow in [true, false] {
            let err = CompactionError::summarization_failed_from(RejectedError { overflow });

            assert_eq!(err.is_context_overflow(), overflow);
            assert_eq!(err.to_string(), "Summarization failed: request rejected");
            let source = err
                .source()
                .expect("source")
                .downcast_ref::<RejectedError>();
            assert_eq!(source.expect("original error").overflow, overflow);
        }
    }

    #[test]
    fn summarization_failed_keeps_each_part() {
        let err = CompactionError::summarization_failed("no budget", true, None);
        assert!(err.is_context_overflow());
        assert_eq!(err.to_string(), "Summarization failed: no budget");
        assert!(err.source().is_none());

        let err = CompactionError::summarization_failed(
            "wrapped",
            false,
            Some(Box::new(RejectedError { overflow: true })),
        );
        assert!(
            !err.is_context_overflow(),
            "the explicit flag wins over the source"
        );
        assert!(err.source().unwrap().is::<RejectedError>());
    }

    #[test]
    fn not_enough_history_is_not_an_overflow() {
        assert!(!CompactionError::NotEnoughHistory.is_context_overflow());
    }

    fn spent() -> Usage {
        let mut usage = Usage::default();
        usage.input_tokens = 7;
        usage
    }

    #[test]
    fn summarization_failed_constructors_start_at_zero_usage() {
        for err in [
            CompactionError::summarization_failed("no budget", false, None),
            CompactionError::summarization_failed_from(RejectedError { overflow: false }),
        ] {
            assert_eq!(err.usage(), Usage::default());
            assert_eq!(err.with_usage(spent()).usage(), spent());
        }
    }

    #[test]
    fn not_enough_history_never_reports_usage() {
        let err = CompactionError::NotEnoughHistory.with_usage(spent());
        assert_eq!(err.usage(), Usage::default());
    }
}
