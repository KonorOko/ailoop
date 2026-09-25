use ailoop_core::{Message, Usage};
use ailoop_prompts::PromptError;
use ailoop_tools::ToolRegistryError;

/// What went wrong in a failed [`Conversation::run`] /
/// [`Conversation::stream`] or [`run_chat`] call. Those entry points
/// return it wrapped in a [`RunError`], which also carries the steps
/// the run completed before failing.
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
    Tool(#[from] ailoop_tools::ToolRegistryError),

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

/// A run that ended in `Err`: the cause ([`EngineError`]), what the
/// run spent, and the messages of every step the run completed before
/// it failed.
///
/// Returned by [`Conversation::run`] / [`Conversation::stream`] (and
/// their `_with_options` variants), yielded as the error item of
/// [`RunStream`], and returned by [`run_chat`].
///
/// # Partial messages
///
/// [`partial_messages`](Self::partial_messages) holds the messages of
/// the steps that finished before the failure: the same list as the
/// `new_messages_so_far` of the last [`StreamChunk::StepFinished`] the
/// run emitted, or empty when the first step failed. Every `tool_use`
/// in it is followed by its `tool_result`. The step that failed is left
/// out entirely, including any assistant text streamed before the
/// error.
///
/// A model error in the middle of a response arrives before that
/// step's tools run, so the partial list records every tool the run
/// executed.
///
/// # History
///
/// [`Conversation`] still rolls its history back on `Err`: it ends
/// with the kickoff message, as if the run never happened. The partial
/// messages are a copy. To keep the record of what already ran (for
/// example tools with side effects), append them yourself:
///
/// ```no_run
/// # async fn demo<M>(chat: &mut ailoop::Conversation<M>)
/// # where M: ailoop::CompletionModel, M::Error: ailoop::ProviderError {
/// if let Err(err) = chat.run("deploy the service").await {
///     eprintln!("run failed: {err}");
///     let parts = err.into_parts();
///     chat.history_extend(parts.partial_messages);
/// }
/// # }
/// ```
///
/// # Usage
///
/// [`usage`](Self::usage) is what the run spent before failing,
/// counted like [`StreamChunk::RunFinished::usage`] on a run that ends
/// in `Ok`: every provider turn that finished plus the usage tools
/// reported through [`ToolContext::report_usage`], sub-agents included.
/// The turn that failed is not counted: a response cut off by an error
/// never reports its usage, although the provider may still bill it.
///
/// The usage can cover more than the partial messages. When a turn
/// finishes and the step then fails (for example a tool registry
/// error), the turn and anything its tools reported are counted, but
/// the step is left out of the partial messages.
///
/// An error raised before the run starts (history compaction before the
/// first step in [`Conversation`]) has zero usage.
///
/// `RunError` converts into [`EngineError`] with `?`, dropping the
/// partial messages and the usage, so code that propagates
/// `EngineError` keeps compiling.
///
/// [`Conversation`]: crate::Conversation
/// [`Conversation::run`]: crate::Conversation::run
/// [`Conversation::stream`]: crate::Conversation::stream
/// [`RunStream`]: crate::RunStream
/// [`run_chat`]: crate::advanced::run_chat
/// [`StreamChunk::StepFinished`]: ailoop_core::StreamChunk::StepFinished
/// [`StreamChunk::RunFinished::usage`]: ailoop_core::StreamChunk::RunFinished::usage
/// [`ToolContext::report_usage`]: ailoop_tools::ToolContext::report_usage
#[derive(Debug)]
#[non_exhaustive]
pub struct RunError<E: std::error::Error> {
    kind: EngineError<E>,
    usage: Usage,
    partial_messages: Vec<Message>,
}

impl<E: std::error::Error> RunError<E> {
    pub(crate) fn new(kind: EngineError<E>, usage: Usage, partial_messages: Vec<Message>) -> Self {
        Self {
            kind,
            usage,
            partial_messages,
        }
    }

    /// What made the run fail.
    pub fn kind(&self) -> &EngineError<E> {
        &self.kind
    }

    /// Consumes the error and returns its cause, dropping the partial
    /// messages.
    pub fn into_kind(self) -> EngineError<E> {
        self.kind
    }

    /// What the run spent before failing: finished turns plus usage
    /// reported by tools. The failed turn is not counted. See the
    /// [type docs](Self#usage).
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Messages of the steps completed before the failure. See the
    /// [type docs](Self#partial-messages).
    pub fn partial_messages(&self) -> &[Message] {
        &self.partial_messages
    }

    /// Splits the error into owned parts: its cause, its usage and its
    /// partial messages.
    pub fn into_parts(self) -> RunErrorParts<E> {
        RunErrorParts {
            kind: self.kind,
            usage: self.usage,
            partial_messages: self.partial_messages,
        }
    }
}

/// The owned parts of a [`RunError`], returned by
/// [`RunError::into_parts`].
///
/// `#[non_exhaustive]` so a later release can add parts; destructure
/// it with `..`:
///
/// ```no_run
/// # fn demo<E: std::error::Error>(err: ailoop::RunError<E>) {
/// let ailoop::RunErrorParts { kind, partial_messages, .. } = err.into_parts();
/// # }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct RunErrorParts<E: std::error::Error> {
    /// What made the run fail.
    pub kind: EngineError<E>,
    /// What the run spent before failing. See the
    /// [`RunError` docs](RunError#usage).
    pub usage: Usage,
    /// Messages of the steps completed before the failure. See the
    /// [`RunError` docs](RunError#partial-messages).
    pub partial_messages: Vec<Message>,
}

impl<E: std::error::Error> std::fmt::Display for RunError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.kind, f)
    }
}

// `Display` already prints the cause, so `source` skips it and goes
// one level down; error reporters would print it twice otherwise.
impl<E: std::error::Error> std::error::Error for RunError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.kind.source()
    }
}

/// Wraps an error raised before any step ran: the usage is zero and
/// the partial list is empty.
impl<E: std::error::Error> From<EngineError<E>> for RunError<E> {
    fn from(kind: EngineError<E>) -> Self {
        Self::new(kind, Usage::default(), Vec::new())
    }
}

/// Keeps `?` working in functions that return [`EngineError`]. The
/// partial messages are dropped.
impl<E: std::error::Error> From<RunError<E>> for EngineError<E> {
    fn from(err: RunError<E>) -> Self {
        err.kind
    }
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
