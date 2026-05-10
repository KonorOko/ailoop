//! Token-counting hooks for [`crate::ContextManager`].
//!
//! As of Train B (Real token budget), the canonical trait lives in
//! `ailoop-prompts` so every crate that needs to size text or messages
//! measures against the same contract. This module re-exports
//! [`Tokenizer`] for callers that already depend on `ailoop-context`,
//! and keeps the legacy [`CharEstimator`] name working — marked
//! `#[deprecated]` — so existing call sites compile without immediate
//! churn.
//!
//! New code should reference `ailoop_prompts::Tokenizer` (or the
//! re-export here) and pass implementations into
//! [`crate::ContextManagerBuilder::tokenizer`]. The `CharTokenizer`
//! fallback (`text.len() / 4`) is documented as a coarse approximation,
//! fine for tests and bring-up but explicitly not the recommended
//! production default — wire up `ailoop_anthropic::OnlineCalibratedTokenizer`
//! (or your provider's equivalent) when budgets matter.
//!
//! The pre-Train-B `TokenEstimator` trait has been subsumed by
//! [`Tokenizer`]; it lived only inside `ailoop-context`, so removing
//! it does not affect downstream crates. Use `Tokenizer` directly and
//! call its `count_*` methods (see the trait in `ailoop-prompts`).

pub use ailoop_prompts::{CharTokenizer, Tokenizer};

/// Deprecated alias for [`CharTokenizer`]. The fallback
/// `text.len() / 4` implementation moved to `ailoop-prompts` so every
/// crate can reach it without depending on `ailoop-context`. Re-exported
/// here as a type alias for back-compat with code that referenced
/// `ailoop_context::tokens::CharEstimator`.
#[deprecated(
    since = "0.1.1",
    note = "renamed to `CharTokenizer`; moved to `ailoop-prompts` and re-exported here"
)]
pub type CharEstimator = CharTokenizer;
