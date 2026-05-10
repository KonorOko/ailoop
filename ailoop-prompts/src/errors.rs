//! Failure surface of [`PromptSection::from_file`](crate::PromptSection::from_file).

use std::path::PathBuf;
use thiserror::Error;

/// Errors surfaced when loading prompt content from disk.
///
/// The façade [`ConversationBuilder::system_prompt_file`](https://docs.rs/ailoop)
/// wraps this in [`BuildError::Prompt`](https://docs.rs/ailoop) so a
/// missing prompt file fails the builder rather than the run.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PromptError {
    /// `std::fs::read_to_string(path)` failed (file not found,
    /// permission denied, invalid UTF-8). The wrapped path is the one
    /// supplied to [`PromptSection::from_file`](crate::PromptSection::from_file)
    /// so callers can pinpoint the misconfigured entry.
    #[error("failed to read prompt file {path}: {source}")]
    LoadFile {
        /// Path the read attempt targeted.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
}
