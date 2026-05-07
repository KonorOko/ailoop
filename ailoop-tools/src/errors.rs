use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolRegistryError {
    #[error("Tool '{0}' not found")]
    NotFound(String),

    #[error("Tool {0} is core")]
    NotDeferred(String),

    #[error("Tool {0} already registered")]
    AlreadyRegistered(String),
}
