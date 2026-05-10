mod errors;
mod prompt;

pub use errors::PromptError;
pub use prompt::{Prompt, PromptSection};

// Re-export `Tokenizer` from `ailoop-core` so callers that already
// depend on `ailoop-prompts` can reach it without pulling in
// `ailoop-core` directly. The trait lives in `ailoop-core` next to
// `Message` so `ailoop-context` (history compaction) and this crate
// (prompt assembly) share a counter without depending on each other.
pub use ailoop_core::{CharTokenizer, Tokenizer};
