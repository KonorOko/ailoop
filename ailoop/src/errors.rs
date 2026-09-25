use ailoop_prompts::PromptError;
use ailoop_tools::errors::ToolRegistryError;

/// Failure surface of [`Conversation::run`] /
/// [`Conversation::stream`] and the underlying [`run_chat`] engine.
///
/// **Aborts are not in here.** Cancellation via
/// [`RunConfig::cancellation`], timeout via [`RunConfig::timeout`],
/// hitting [`RunConfig::max_iterations`], and middleware/tool
/// returning `Terminate` all surface as `Ok(_)` carrying
/// [`FinishReason::Aborted`] — see [`Conversation::run`] for the full
/// contract. `EngineError` is reserved for transport / setup-time
/// failures: a model-side HTTP error, a tool registry error not tied
/// to a single tool call, or a context-manager compaction failure.
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
    /// [`CompactionStrategy`]: ailoop_history::CompactionStrategy
    /// [`CompactionError`]: ailoop_history::CompactionError
    #[error("context error: {0}")]
    Context(#[from] ailoop_history::CompactionError),

    /// The provider rejected the request because the prompt does not
    /// fit the model's context window
    /// ([`ProviderError::is_context_overflow`]), and compacting the
    /// history could not fix it. Only [`Conversation`] returns this,
    /// and only with
    /// [`recover_from_context_overflow`](crate::ConversationBuilder::recover_from_context_overflow)
    /// on (the default). It is returned when any of these happens:
    ///
    /// - the request still overflowed after one forced compaction and
    ///   retry;
    /// - the forced compaction had nothing to drop
    ///   ([`CompactionError::NotEnoughHistory`]);
    /// - the forced compaction did not reduce the estimated token count.
    ///
    /// The typical cause is a turn whose own content (for example a
    /// huge tool result) is larger than the window: the built-in
    /// strategies never cut into the run in progress.
    ///
    /// The wrapped value is the provider's last error. The
    /// conversation history is rolled back to its state when the run
    /// started, so no half-finished turn is left behind. With recovery
    /// turned off, an overflow surfaces as [`EngineError::Model`]
    /// instead.
    ///
    /// [`ProviderError::is_context_overflow`]: ailoop_core::ProviderError::is_context_overflow
    /// [`Conversation`]: crate::Conversation
    /// [`CompactionError::NotEnoughHistory`]: ailoop_history::CompactionError::NotEnoughHistory
    #[error("prompt does not fit the context window even after compaction: {0}")]
    ContextOverflow(E),
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

    /// [`tools_with_prompt_file`](crate::ConversationBuilder::tools_with_prompt_file)
    /// was called with an empty tool-name iterator — a group with no
    /// tools could never fire, so the builder refuses it rather than
    /// silently dropping the prompt section.
    #[error("tools_with_prompt_file called with an empty tool-name list")]
    EmptyToolGroup,
}
