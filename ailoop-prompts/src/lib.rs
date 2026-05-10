mod errors;
mod prompt;

pub use errors::PromptError;
pub use prompt::{Prompt, PromptSection};

// Re-export the canonical `Tokenizer` trait so callers that already
// pull in `ailoop-prompts` (most do, indirectly via the top-level
// `ailoop` crate) can reach it without an extra dep on `ailoop-core`.
// The trait itself lives in `ailoop-core` because `Message` lives
// there, and forcing every tokenizer consumer through the prompts
// crate would couple `ailoop-context` (history management) to prompt
// assembly, which has nothing to do with token counting.
pub use ailoop_core::{CharTokenizer, Tokenizer};
