use ailoop_prompts::PromptError;
use ailoop_tools::errors::ToolRegistryError;

/// Failure surface of [`Conversation::run`] /
/// [`Conversation::stream`] and the underlying [`run_chat`] engine.
///
/// **Aborts are not in here.** Cancellation via
/// [`RunConfig::cancellation`], timeout via [`RunConfig::timeout`],
/// and middleware/tool returning `Terminate` all surface as
/// `Ok(_)` carrying [`FinishReason::Aborted`] — see
/// [`Conversation::run`] for the full contract. `EngineError` is
/// reserved for transport / setup-time failures: a model-side HTTP
/// error, a tool registry error not tied to a single tool call, a
/// context-manager compaction failure, or the engine breaking out
/// because [`RunConfig::max_iterations`] was hit.
///
/// [`Conversation::run`]: crate::Conversation::run
/// [`Conversation::stream`]: crate::Conversation::stream
/// [`run_chat`]: crate::advanced::run_chat
/// [`RunConfig::cancellation`]: ailoop_core::RunConfig::cancellation
/// [`RunConfig::timeout`]: ailoop_core::RunConfig::timeout
/// [`RunConfig::max_iterations`]: ailoop_core::RunConfig::max_iterations
/// [`FinishReason::Aborted`]: ailoop_core::FinishReason::Aborted
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError<E: std::error::Error> {
    /// The provider adapter (the [`CompletionModel`]) returned an
    /// error from `chat_stream`. The wrapped value is the adapter's
    /// own error type — typically an HTTP status, a transport error,
    /// or a parsing failure.
    ///
    /// [`CompletionModel`]: ailoop_core::CompletionModel
    #[error("model error: {0}")]
    Model(E),

    /// The tool registry returned an error during a tool call other
    /// than `NotFound` (which the engine handles in-band by feeding an
    /// `Error` tool result back to the model rather than aborting the
    /// run).
    #[error("tool error: {0}")]
    Tool(#[from] ailoop_tools::errors::ToolRegistryError),

    /// History compaction failed. Typically this means a configured
    /// [`CompactionStrategy`] could not satisfy the token budget — see
    /// [`CompactionError`] for the concrete cases.
    ///
    /// [`CompactionStrategy`]: ailoop_context::CompactionStrategy
    /// [`CompactionError`]: ailoop_context::CompactionError
    #[error("context error: {0}")]
    Context(#[from] ailoop_context::CompactionError),

    /// The agent loop broke out because the iteration counter reached
    /// [`RunConfig::max_iterations`]. The wrapped value is the
    /// configured cap. Raise the cap or attach an
    /// [`AntiLoop`](crate::AntiLoop) middleware to stop earlier.
    ///
    /// [`RunConfig::max_iterations`]: ailoop_core::RunConfig::max_iterations
    #[error("agent loop exceeded max iterations ({0})")]
    MaxIterationsExceeded(usize),
}

/// Errors accumulated by [`ConversationBuilder`] and surfaced when
/// [`build`](crate::ConversationBuilder::build) is called. Distinct
/// from [`EngineError`]: these fire during *setup*, not during a run.
///
/// [`ConversationBuilder`]: crate::ConversationBuilder
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BuildError {
    /// A tool failed to register — typically a duplicate tool name
    /// across two `tool(...)` / `tool_dyn(...)` calls.
    #[error("tool registration failed: {0}")]
    ToolRegistry(#[from] ToolRegistryError),

    /// A prompt file (passed to
    /// [`tool_with_prompt_file`](crate::ConversationBuilder::tool_with_prompt_file)
    /// or [`system_prompt_file`](crate::ConversationBuilder::system_prompt_file))
    /// could not be read or parsed at builder time.
    #[error("prompt error: {0}")]
    Prompt(#[from] PromptError),
}
