//! Engine event vocabulary: [`StreamChunk`], [`FinishReason`], and
//! [`Usage`].

use std::{ops::Add, sync::Arc};

use crate::{Message, RunId, StepId, ToolResultContent};

/// Event the engine emits as a run progresses.
///
/// Variants split into two families:
///
/// 1. **Provider stream events**, surfaced once per turn —
///    `TextDelta`, `ToolCall*`, `Reasoning*`, `RedactedReasoningBlock`,
///    `TurnFinished`. Adapters lower wire deltas into these. Started/
///    Finished pairs always nest cleanly: a `ToolCallFinished` arrives
///    before any other tool call's `Started`.
/// 2. **Engine lifecycle events**, synthesized by the engine itself
///    around the provider stream — `RunStarted`, `StepStarted`,
///    `StepFinished`, `ToolResult`, `RunFinished`, `HistoryCompacted`.
///
/// Every chunk reaches every [`crate::ChatMiddleware::on_chunk`]; the
/// engine also drives its own assistant-history reconstruction off
/// these events, so middlewares that override
/// [`crate::ChatMiddleware::on_chunk_mut`] can rewrite them in flight
/// to influence what gets persisted.
#[derive(Debug)]
#[non_exhaustive]
pub enum StreamChunk {
    /// Incremental visible text from the model. Concatenate deltas in
    /// arrival order to reconstruct the assistant text block.
    TextDelta {
        /// New text appended this delta. Empty deltas are legal.
        delta: String,
    },
    /// A new tool call has begun. The model has emitted the tool name
    /// but no arguments yet. Pair with the matching
    /// [`Self::ToolCallFinished`] (same `id`) once the call is fully
    /// assembled.
    ToolCallStarted {
        /// Provider-assigned id; mirrors back as `call_id` on the
        /// [`Self::ToolResult`] the engine emits after execution.
        id: String,
        /// Tool name as registered in the request's `tools` list.
        name: String,
    },
    /// Incremental tool-call argument JSON. Concatenate deltas in
    /// arrival order to rebuild the `args` JSON for live UIs;
    /// engines that only need the final structure can ignore these
    /// and read [`Self::ToolCallFinished::args`] instead.
    ToolCallArgsDelta {
        /// Tool call id; matches the originating
        /// [`Self::ToolCallStarted::id`].
        id: String,
        /// JSON fragment appended this delta.
        delta: String,
    },
    /// A tool call is fully assembled and ready to execute. The
    /// engine invokes the tool after this chunk and emits a
    /// [`Self::ToolResult`] when execution completes.
    ToolCallFinished {
        /// Tool call id; matches the originating
        /// [`Self::ToolCallStarted::id`].
        id: String,
        /// Tool name, repeated for convenience so consumers do not
        /// have to track the originating `Started` chunk.
        name: String,
        /// Final, parsed JSON arguments.
        args: serde_json::Value,
    },
    /// Incremental reasoning text. Same accumulation contract as
    /// [`Self::TextDelta`], but feeds an
    /// [`crate::AssistantBlock::Reasoning`] block instead of
    /// [`crate::AssistantBlock::Text`]. Some providers (Anthropic
    /// extended thinking) require the assembled reasoning to be
    /// replayed verbatim on subsequent turns when tools are involved.
    ReasoningDelta {
        /// New reasoning text appended this delta.
        delta: String,
    },
    /// End of a visible reasoning block. Carries the provider signature when
    /// applicable (Anthropic extended thinking); other providers may emit
    /// `None`. Engines should pair this with the accumulated reasoning text
    /// to materialize an `AssistantBlock::Reasoning`.
    ReasoningFinished {
        /// Provider signature for the reasoning block (Anthropic
        /// extended thinking). Persist alongside the reasoning text
        /// in [`crate::AssistantBlock::Reasoning`]; replay verbatim on
        /// subsequent turns when tools are involved.
        signature: Option<String>,
    },
    /// A complete redacted reasoning block delivered atomically. `data` is
    /// opaque provider material that must be replayed verbatim on the next
    /// request. Engines should materialize `AssistantBlock::RedactedReasoning`
    /// directly from this chunk; no deltas are emitted around it.
    RedactedReasoningBlock {
        /// Verbatim provider payload; treat as opaque bytes.
        data: String,
    },
    /// End of a single provider turn. Equivalent to a Chat
    /// Completions `finish_reason` plus the final `usage`. Multiple
    /// turns can fire per run when the model is in a tool-use loop.
    TurnFinished {
        /// Why the model stopped this turn.
        reason: FinishReason,
        /// Token counters reported by the provider for this turn.
        usage: Usage,
        /// Provider-reported service tier for the turn (Anthropic:
        /// `"standard"` / `"priority"` / `"batch"`). `None` when the
        /// provider does not surface one. Per-turn rather than
        /// aggregated because it is a categorical label, not a counter.
        service_tier: Option<String>,
    },

    // Extend
    /// Engine has accepted the run; emitted exactly once per run
    /// before the first provider call.
    RunStarted {
        /// Identifier shared by every chunk this run produces.
        run_id: RunId,
    },
    /// Engine is starting a step (one provider turn plus the tool
    /// calls it triggers).
    StepStarted {
        /// Run this step belongs to.
        run_id: RunId,
        /// Identifier shared by every chunk this step produces.
        step_id: StepId,
        /// 0-based iteration number; bounded by
        /// [`crate::RunConfig::max_iterations`].
        iteration: usize,
    },
    /// Engine has finished a step. Includes the cumulative messages
    /// added to history so far, so observers can snapshot
    /// mid-conversation without waiting for [`Self::RunFinished`].
    StepFinished {
        /// Run this step belongs to.
        run_id: RunId,
        /// Step that just finished.
        step_id: StepId,
        /// 0-based iteration number, matching [`Self::StepStarted::iteration`].
        iteration: usize,
        /// All messages this run has appended to history so far,
        /// shared so observers can read without cloning the vector.
        new_messages_so_far: Arc<Vec<Message>>,
    },
    /// A tool finished executing and produced a reply. Emitted
    /// **after** [`Self::ToolCallFinished`] and before the next
    /// provider turn picks up the result.
    ToolResult {
        /// Run that owns the tool call.
        run_id: RunId,
        /// Step that owns the tool call.
        step_id: StepId,
        /// Matches [`Self::ToolCallFinished::id`].
        call_id: String,
        /// Tool reply, with the [`ToolResultContent::Error`] flag
        /// preserved for the next provider turn.
        content: ToolResultContent,
    },
    /// Engine has finished the run. Emitted exactly once per run,
    /// even on aborts and middleware terminations.
    RunFinished {
        /// Run that just finished.
        run_id: RunId,
        /// Why the run ended.
        reason: FinishReason,
        /// Token totals across every turn in the run.
        usage: Usage,
        /// All messages this run added to history. On abort, partial
        /// tool results are preserved so the next run sees a
        /// consistent shape.
        new_messages: Vec<Message>,
    },
    /// Emitted by `Conversation::stream` (not the engine) when history
    /// compaction ran before the request was sent. Carries message
    /// counts from before/after compaction and the strategy's name so
    /// observability middlewares can report what was dropped.
    HistoryCompacted {
        /// Run for which compaction ran. Shared with the
        /// engine-emitted chunks of the same run.
        run_id: RunId,
        /// Number of messages in history before compaction.
        before_count: usize,
        /// Number of messages in history after compaction.
        after_count: usize,
        /// Name reported by the strategy
        /// (`CompactionStrategy::name()`), e.g. `"truncate"` or
        /// `"summarize"`.
        strategy: &'static str,
    },
}

/// Reason a provider turn (or an entire run) ended.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FinishReason {
    /// Model produced a complete reply with no tool call. The natural
    /// terminator of a run.
    EndTurn,
    /// Model emitted at least one tool call. The engine continues the
    /// run by executing tools and issuing the next turn.
    ToolUse,
    /// Model stopped because [`crate::ChatRequest::max_tokens`] was
    /// reached. The reply is partial.
    MaxTokens,
    /// Model emitted one of the configured
    /// [`crate::ChatRequest::stop_sequences`].
    StopSequence,
    /// Run was terminated outside the model: cancellation token,
    /// timeout, [`crate::HookAction::Terminate`], or
    /// [`crate::ToolDecision::Terminate`]. The string carries a
    /// human-readable reason (`"cancelled by caller"`,
    /// `"timeout: ..."`, the middleware-supplied `reason`, etc.).
    /// The engine guarantees this is the *only* finish reason ever
    /// surfaced for caller-initiated stops — `Err` results are
    /// reserved for transport errors.
    Aborted(String),
    /// Provider reported a finish reason the adapter did not map to
    /// one of the typed variants. Treat as terminal.
    Other(String),
}

/// Token counters reported by the provider for a turn.
///
/// Aggregated to the run level by the engine and surfaced on
/// [`StreamChunk::RunFinished`] / [`StreamChunk::TurnFinished`]. Fields
/// not surfaced by a given provider stay at `0`.
#[derive(Debug, Default, Clone, Copy)]
#[non_exhaustive]
pub struct Usage {
    /// Total prompt tokens charged this turn (cached + uncached).
    pub input_tokens: u32,
    /// Tokens generated by the model this turn.
    pub output_tokens: u32,
    /// Subset of `input_tokens` that were served from prompt cache
    /// rather than recomputed. Zero when the provider does not
    /// support prompt caching or when this turn missed the cache.
    pub cached_input_tokens: u32,
    /// Total tokens written to a cache during this turn. When the
    /// provider reports a TTL breakdown (Anthropic), this equals the sum
    /// of [`Self::cache_creation_5m_tokens`] + [`Self::cache_creation_1h_tokens`].
    /// When only the legacy flat field is reported, the breakdown stays
    /// at zero and only this total is populated.
    pub cache_creation_input_tokens: u32,
    /// Cache writes with a 5-minute TTL (Anthropic ephemeral default).
    /// Zero when the provider does not surface a TTL breakdown.
    pub cache_creation_5m_tokens: u32,
    /// Cache writes with a 1-hour TTL (Anthropic explicit ttl="1h").
    /// Zero when the provider does not surface a TTL breakdown.
    pub cache_creation_1h_tokens: u32,
}

impl Add for Usage {
    type Output = Usage;

    fn add(self, other: Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            cached_input_tokens: self.cached_input_tokens + other.cached_input_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens
                + other.cache_creation_input_tokens,
            cache_creation_5m_tokens: self.cache_creation_5m_tokens
                + other.cache_creation_5m_tokens,
            cache_creation_1h_tokens: self.cache_creation_1h_tokens
                + other.cache_creation_1h_tokens,
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.cache_creation_5m_tokens += other.cache_creation_5m_tokens;
        self.cache_creation_1h_tokens += other.cache_creation_1h_tokens;
    }
}
