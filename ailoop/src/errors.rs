use ailoop_prompts::PromptError;
use ailoop_tools::errors::ToolRegistryError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError<E: std::error::Error> {
    #[error("model error: {0}")]
    Model(E),

    #[error("tool error: {0}")]
    Tool(#[from] ailoop_tools::errors::ToolRegistryError),

    #[error("context error: {0}")]
    Context(#[from] ailoop_context::errors::CompactionError),

    #[error("agent loop exceeded max iterations ({0})")]
    MaxIterationsExceeded(usize),
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BuildError {
    #[error("tool registration failed: {0}")]
    ToolRegistry(#[from] ToolRegistryError),

    #[error("prompt error: {0}")]
    Prompt(#[from] PromptError),
}
