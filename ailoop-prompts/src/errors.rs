use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PromptError {
    #[error("failed to read prompt file {path}: {source}")]
    LoadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
