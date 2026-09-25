//! Provider-agnostic questions about a model error.
//!
//! [`Retryable`](crate::Retryable) answers "should the call be sent
//! again as-is?". [`ProviderError`] answers "what went wrong, in terms
//! the layers above the model can act on?" — without those layers
//! knowing the adapter's concrete error type.

/// Provider-agnostic classification of a [`CompletionModel`](crate::CompletionModel)
/// error.
///
/// Implemented by each adapter's error type (`AnthropicError`,
/// `AzureOpenAIError`). Every method has a conservative default, so an
/// adapter that cannot detect a condition implements the trait with an
/// empty body and callers see `false`.
///
/// Kept separate from [`Retryable`](crate::Retryable) on purpose: a
/// context overflow is [`Permanent`](crate::RetryClassification::Permanent)
/// for [`RetryingModel`](crate::RetryingModel) — resending the same
/// prompt fails the same way — yet it is recoverable by a caller that
/// owns the history and can shrink it before trying again.
pub trait ProviderError {
    /// `true` when the provider rejected the request because the prompt
    /// does not fit the model's context window (for example Anthropic's
    /// `"prompt is too long"` `invalid_request_error`, or Azure OpenAI's
    /// `context_length_exceeded`).
    ///
    /// A caller that sees `true` can compact the history and reissue
    /// the request; retrying unchanged will not help.
    fn is_context_overflow(&self) -> bool {
        false
    }
}
