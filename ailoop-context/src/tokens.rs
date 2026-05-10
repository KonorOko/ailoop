//! Token-counting hooks for [`crate::ContextManager`].
//!
//! Re-exports of [`Tokenizer`] and [`CharTokenizer`] from `ailoop-core`
//! so callers that already depend on `ailoop-context` can reach the
//! canonical trait without an extra dep. Pass implementations into
//! [`crate::ContextManagerBuilder::tokenizer`]; the `CharTokenizer`
//! fallback (`text.len() / 4`) is a coarse approximation suitable for
//! tests and bring-up, not for production budgeting — wire up
//! `ailoop_anthropic::OnlineCalibratedTokenizer` (or your provider's
//! equivalent) when budgets matter.

pub use ailoop_core::{CharTokenizer, Tokenizer};
