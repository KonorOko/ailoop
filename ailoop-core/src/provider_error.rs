//! Provider-agnostic questions about a model error.
//!
//! [`Retryable`](crate::Retryable) answers "should the call be sent
//! again as-is?". [`ProviderError`] answers "what went wrong, in terms
//! the layers above the model can act on?" — without those layers
//! knowing the adapter's concrete error type.

/// What went wrong with a [`CompletionModel`](crate::CompletionModel)
/// call, in provider-agnostic terms. Returned by
/// [`ProviderError::kind`].
///
/// The kind carries no data on purpose. How long to wait before trying
/// again lives only in
/// [`Retryable::retry_classification`](crate::Retryable::retry_classification),
/// so the two answers cannot disagree: combine `kind()` with
/// `retry_classification()` of the same error.
///
/// A service that cannot be reached is [`Network`](Self::Network)
/// whether it is remote or running on the same machine; adapters that
/// talk to local servers say which one it was through their own API.
///
/// `#[non_exhaustive]`: match it with a `_` arm, which should treat an
/// unknown kind like [`Other`](Self::Other).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ProviderErrorKind {
    /// The request did not reach the provider, or the connection broke
    /// before a response arrived: DNS, TLS, connection refused or
    /// reset, a transport failure while reading the stream.
    Network,
    /// The credentials are missing, invalid or expired.
    Auth,
    /// The credentials are valid but lack access to the requested
    /// model or resource.
    Permission,
    /// The caller sent too many requests or tokens in a time window.
    /// Waiting helps.
    RateLimited,
    /// The provider is overloaded or failing on its side (5xx,
    /// overloaded). Waiting helps.
    Unavailable,
    /// The account ran out of credit or of a quota that waiting does
    /// not refill. Needs action from the account owner.
    QuotaExhausted,
    /// The prompt does not fit the model's context window. Compacting
    /// the history and trying again can help; resending unchanged
    /// cannot.
    ContextOverflow,
    /// The request body is larger than the provider accepts (for
    /// example large images), independent of the context window.
    RequestTooLarge,
    /// A content-safety filter blocked the request or the response.
    ContentFilter,
    /// The model, or the deployment that serves it, does not exist.
    ModelNotFound,
    /// The model or the adapter cannot handle a kind of content or
    /// feature the request uses.
    Unsupported(Capability),
    /// The response stream ended before the provider said it was done,
    /// so the response may be incomplete.
    Interrupted,
    /// The provider rejected the request as malformed or invalid for a
    /// reason none of the other kinds describe.
    InvalidRequest,
    /// Anything else, including errors the adapter cannot classify.
    Other,
}

/// A feature a model or adapter may not support. Carried by
/// [`ProviderErrorKind::Unsupported`].
///
/// `#[non_exhaustive]`: match it with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Capability {
    /// Image input.
    Images,
    /// Tool definitions and tool calls.
    Tools,
}

/// Provider-agnostic classification of a [`CompletionModel`](crate::CompletionModel)
/// error.
///
/// Implemented by each adapter's error type (`AnthropicError`,
/// `AzureOpenAIError`). [`kind`](Self::kind) is the primitive and the
/// other methods derive from it, so an adapter only implements `kind`.
/// Every method has a conservative default, so an adapter that cannot
/// classify anything implements the trait with an empty body and
/// callers see [`ProviderErrorKind::Other`].
///
/// Kept separate from [`Retryable`](crate::Retryable) on purpose: a
/// context overflow is [`Permanent`](crate::RetryClassification::Permanent)
/// for [`RetryingModel`](crate::RetryingModel) — resending the same
/// prompt fails the same way — yet it is recoverable by a caller that
/// owns the history and can shrink it before trying again.
pub trait ProviderError {
    /// What went wrong. Defaults to [`ProviderErrorKind::Other`].
    ///
    /// Implementations must not call
    /// [`is_context_overflow`](Self::is_context_overflow), whose
    /// default calls this method.
    fn kind(&self) -> ProviderErrorKind {
        ProviderErrorKind::Other
    }

    /// `true` when the provider rejected the request because the prompt
    /// does not fit the model's context window (for example Anthropic's
    /// `"prompt is too long"` `invalid_request_error`, or Azure OpenAI's
    /// `context_length_exceeded`).
    ///
    /// A caller that sees `true` can compact the history and reissue
    /// the request; retrying unchanged will not help.
    ///
    /// Defaults to `kind() == ProviderErrorKind::ContextOverflow`; there
    /// is no need to override it.
    fn is_context_overflow(&self) -> bool {
        matches!(self.kind(), ProviderErrorKind::ContextOverflow)
    }
}

/// A model that cannot fail never overflows. Lets mocks and
/// infallible models (`type Error = Infallible`) satisfy a
/// `ProviderError` bound without boilerplate.
impl ProviderError for std::convert::Infallible {}

#[cfg(test)]
mod tests {
    use super::*;

    struct Classified(ProviderErrorKind);

    impl ProviderError for Classified {
        fn kind(&self) -> ProviderErrorKind {
            self.0
        }
    }

    struct Unclassified;

    impl ProviderError for Unclassified {}

    /// An implementation written before `kind()` existed.
    struct OverflowOnly;

    impl ProviderError for OverflowOnly {
        fn is_context_overflow(&self) -> bool {
            true
        }
    }

    #[test]
    fn defaults_are_other_and_no_overflow() {
        assert_eq!(Unclassified.kind(), ProviderErrorKind::Other);
        assert!(!Unclassified.is_context_overflow());
    }

    #[test]
    fn is_context_overflow_derives_from_kind() {
        assert!(Classified(ProviderErrorKind::ContextOverflow).is_context_overflow());
        assert!(!Classified(ProviderErrorKind::RequestTooLarge).is_context_overflow());
    }

    #[test]
    fn overriding_only_the_helper_does_not_recurse() {
        assert!(OverflowOnly.is_context_overflow());
        assert_eq!(OverflowOnly.kind(), ProviderErrorKind::Other);
    }
}
