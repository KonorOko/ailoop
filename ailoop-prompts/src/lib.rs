//! Composable system-prompt assembly for ailoop.
//!
//! [`Prompt`] is an ordered list of [`PromptSection`]s. Each section
//! optionally carries a name; `Prompt::render` emits unnamed sections
//! verbatim and prefixes named sections with a `## {name}\n\n` header.
//! That render output is what
//! [`ConversationBuilder::system_prompt`](https://docs.rs/ailoop)
//! sees when a [`Prompt`] is fed to it via `Display` / `Into<String>`.
//!
//! Use this crate when a single hard-coded `system_prompt("…")` is
//! not enough — typically when the prompt is composed from multiple
//! sources (a tone preamble, a per-task playbook, a few-shot block,
//! a tool-usage guide). For prompts loaded from disk see
//! [`PromptSection::from_file`].
//!
//! ## Mini-index
//!
//! - [`Prompt`] / [`PromptBuilder`] — the container and its fluent
//!   constructor.
//! - [`PromptSection`] — one named or unnamed block.
//! - [`PromptError`] — failure surface of [`PromptSection::from_file`].
//! - [`Tokenizer`] / [`CharTokenizer`] — re-exported from
//!   [`ailoop_core`] for [`Prompt::token_count`] /
//!   [`PromptSection::token_count`] callers.

#![deny(missing_docs)]

mod errors;
mod prompt;

pub use errors::PromptError;
pub use prompt::{Prompt, PromptBuilder, PromptSection};

// Re-export `Tokenizer` from `ailoop-core` so callers that already
// depend on `ailoop-prompts` can reach it without pulling in
// `ailoop-core` directly. The trait lives in `ailoop-core` next to
// `Message` so `ailoop-context` (history compaction) and this crate
// (prompt assembly) share a counter without depending on each other.
pub use ailoop_core::{CharTokenizer, Tokenizer};
