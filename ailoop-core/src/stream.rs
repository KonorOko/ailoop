//! Engine event vocabulary: [`StreamChunk`], [`FinishReason`],
//! [`AbortReason`], and [`Usage`].

use std::{fmt, ops::Add, sync::Arc, time::Duration};

use crate::{Message, RunId, StepId, ToolResultContent};

/// Event the engine emits as a run progresses.
///
/// Variants split into two families:
///
/// 1. **Provider stream events**, surfaced once per turn —
///    `TextDelta`, `ToolCall*`, `Reasoning*`, `RedactedReasoningBlock`,
///    `TurnFinished`. Adapters lower wire deltas into these. Started/
///    Finished pairs always nest cleanly: a `ToolCallFinished` (or
///    `ToolCallMalformed`) arrives before any other tool call's `Started`.
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
    /// but no arguments yet. Closed by exactly one matching (same `id`)
    /// [`Self::ToolCallFinished`] or [`Self::ToolCallMalformed`] once
    /// the call is fully assembled.
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
    /// A tool call whose accumulated argument text is not a JSON
    /// object — typically because the turn was cut off by
    /// `max_tokens` in the middle of the arguments. Closes the matching
    /// [`Self::ToolCallStarted`] in place of [`Self::ToolCallFinished`].
    ///
    /// The engine never runs the tool. It records the call in history
    /// with an empty-object input (providers require an object there),
    /// answers it with an error [`Self::ToolResult`] carrying the raw
    /// text so the model can retry, and skips every tool hook
    /// (`on_before_tool_call*`, `on_after_tool_call*`): nothing ran, so
    /// approval, call counters and result rewriters are not involved.
    /// The synthesized `ToolResult` still goes through
    /// [`crate::ChatMiddleware::on_chunk`].
    ///
    /// Adapters build this with [`Self::tool_call_from_raw_args`].
    ToolCallMalformed {
        /// Tool call id; matches the originating
        /// [`Self::ToolCallStarted::id`].
        id: String,
        /// Tool name, repeated for convenience.
        name: String,
        /// Argument text exactly as the provider streamed it.
        raw: String,
        /// Why `raw` was rejected (the JSON parser's message, or a note
        /// that the value is not an object).
        error: String,
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
    ///
    /// Visible to [`crate::ChatMiddleware`] implementations only. The
    /// engine reads `reason` and accumulates `usage` into the run
    /// total, then drops the chunk before it reaches the stream
    /// consumer — a caller iterating the stream returned by
    /// `Conversation::stream_with_options` never observes this
    /// variant. The aggregated total is surfaced on
    /// [`Self::RunFinished::usage`]. To observe per-turn data (for
    /// example, the final turn's `input_tokens` for a
    /// "context full" indicator), implement [`crate::ChatMiddleware`]
    /// and read it from [`crate::ChatMiddleware::on_chunk`]:
    /// middleware hooks run before the engine's per-variant
    /// filtering, so they see `TurnFinished` directly.
    TurnFinished {
        /// Why the model stopped this turn.
        reason: FinishReason,
        /// Token counters reported by the provider for this turn.
        /// Only this run's model: usage reported by tools (e.g. a
        /// sub-agent) is not here, it is added to
        /// [`Self::RunFinished::usage`].
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
    #[non_exhaustive]
    RunStarted {
        /// Identifier shared by every chunk this run produces.
        run_id: RunId,
    },
    /// Engine is starting a step (one provider turn plus the tool
    /// calls it triggers).
    #[non_exhaustive]
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
    #[non_exhaustive]
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
    /// provider turn picks up the result. Also emitted, with an error
    /// reply synthesized by the engine, for every
    /// [`Self::ToolCallMalformed`], and for every call a run aborted
    /// before running (`"Tool not run: the run was aborted (<reason>)"`),
    /// so each call of a step gets exactly one `ToolResult`.
    #[non_exhaustive]
    ToolResult {
        /// Run that owns the tool call.
        run_id: RunId,
        /// Step that owns the tool call.
        step_id: StepId,
        /// Matches [`Self::ToolCallFinished::id`].
        call_id: String,
        /// Tool reply, with `is_error` preserved for the next provider
        /// turn.
        content: ToolResultContent,
    },
    /// Engine has finished the run. Emitted exactly once per run,
    /// even on aborts and middleware terminations.
    #[non_exhaustive]
    RunFinished {
        /// Run that just finished.
        run_id: RunId,
        /// Why the run ended.
        reason: FinishReason,
        /// Everything the run spent: the sum of its own turns plus
        /// the usage tools reported through
        /// `ToolContext::report_usage` (a `SubAgentTool`'s child run,
        /// recursively). On aborts, what was spent up to the cutoff.
        /// See [`Usage`] for how to split own vs. delegated spend.
        usage: Usage,
        /// All messages this run added to history. Every `tool_use` in
        /// it has its `tool_result`, aborts included: calls that
        /// completed keep their result, and calls the abort stopped
        /// get an error result, so the history can be sent again.
        new_messages: Vec<Message>,
    },
    /// History compaction ran on behalf of this run. Carries message
    /// counts from before/after compaction and the strategy's name so
    /// observability middlewares can report what was dropped.
    ///
    /// Emitted by the history-backed `Conversation` path (never by
    /// bare `run_chat`) at three points: as the first chunk when the
    /// history was compacted before the run started; between a
    /// `StepFinished` and the next `StepStarted` when compacting
    /// between iterations is enabled; and inside a step after a
    /// context-window overflow, right before the request is retried.
    /// Mid-run counts cover the whole history, including the messages
    /// the run has added so far.
    #[non_exhaustive]
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

impl StreamChunk {
    /// Build a [`StreamChunk::RunStarted`]. The engine-emitted variants
    /// are `#[non_exhaustive]` so they can gain fields; these
    /// constructors are how code outside `ailoop-core` builds them,
    /// e.g. to unit-test a middleware's [`crate::ChatMiddleware::on_chunk`].
    pub fn run_started(run_id: RunId) -> Self {
        Self::RunStarted { run_id }
    }

    /// Build a [`StreamChunk::StepStarted`].
    pub fn step_started(run_id: RunId, step_id: StepId, iteration: usize) -> Self {
        Self::StepStarted {
            run_id,
            step_id,
            iteration,
        }
    }

    /// Build a [`StreamChunk::StepFinished`].
    pub fn step_finished(
        run_id: RunId,
        step_id: StepId,
        iteration: usize,
        new_messages_so_far: Arc<Vec<Message>>,
    ) -> Self {
        Self::StepFinished {
            run_id,
            step_id,
            iteration,
            new_messages_so_far,
        }
    }

    /// Build a [`StreamChunk::ToolResult`].
    pub fn tool_result(
        run_id: RunId,
        step_id: StepId,
        call_id: impl Into<String>,
        content: ToolResultContent,
    ) -> Self {
        Self::ToolResult {
            run_id,
            step_id,
            call_id: call_id.into(),
            content,
        }
    }

    /// Build a [`StreamChunk::RunFinished`].
    pub fn run_finished(
        run_id: RunId,
        reason: FinishReason,
        usage: Usage,
        new_messages: Vec<Message>,
    ) -> Self {
        Self::RunFinished {
            run_id,
            reason,
            usage,
            new_messages,
        }
    }

    /// Build a [`StreamChunk::HistoryCompacted`].
    pub fn history_compacted(
        run_id: RunId,
        before_count: usize,
        after_count: usize,
        strategy: &'static str,
    ) -> Self {
        Self::HistoryCompacted {
            run_id,
            before_count,
            after_count,
            strategy,
        }
    }

    /// Closes a streamed tool call from its accumulated argument text.
    ///
    /// Returns [`StreamChunk::ToolCallFinished`] when `raw` parses to a
    /// JSON object, or when it is empty or whitespace-only (a tool
    /// without parameters, which providers may stream as no argument
    /// fragments at all) — that case yields `{}`. Anything else (invalid
    /// or truncated JSON, or a valid non-object value) yields
    /// [`StreamChunk::ToolCallMalformed`], so the tool never runs with
    /// arguments the model did not send.
    ///
    /// Intended for provider adapters, at the point where a tool call's
    /// argument deltas are complete.
    pub fn tool_call_from_raw_args(
        id: impl Into<String>,
        name: impl Into<String>,
        raw: impl Into<String>,
    ) -> Self {
        let (id, name, raw) = (id.into(), name.into(), raw.into());
        if raw.trim().is_empty() {
            return Self::ToolCallFinished {
                id,
                name,
                args: serde_json::Value::Object(Default::default()),
            };
        }
        let error = match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(args @ serde_json::Value::Object(_)) => {
                return Self::ToolCallFinished { id, name, args };
            }
            Ok(_) => "tool arguments must be a JSON object".to_string(),
            Err(e) => e.to_string(),
        };
        Self::ToolCallMalformed {
            id,
            name,
            raw,
            error,
        }
    }
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
    /// [`crate::ToolDecision::Terminate`]. The [`AbortReason`] says
    /// which one; match on its variants instead of parsing the
    /// [`Display`](fmt::Display) text. The engine guarantees this is
    /// the *only* finish reason ever surfaced for caller-initiated
    /// stops — `Err` results are reserved for transport errors.
    Aborted(AbortReason),
    /// Provider reported a finish reason the adapter did not map to
    /// one of the typed variants. Treat as terminal.
    Other(String),
}

/// Why the engine aborted a run, carried by [`FinishReason::Aborted`].
///
/// The [`Display`](fmt::Display) impl renders a human-readable
/// message (`"cancelled by caller"`, `"timeout exceeded after 5s"`, or
/// the middleware-supplied reason verbatim) suitable for logs or for
/// feeding back to a parent model.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AbortReason {
    /// [`crate::RunConfig::timeout`] elapsed. Carries the configured
    /// timeout.
    Timeout(Duration),
    /// [`crate::RunConfig::cancellation`] fired. Cancellation wins
    /// over a timeout that fires at the same instant.
    Cancelled,
    /// A middleware returned [`crate::HookAction::Terminate`] from
    /// [`crate::ChatMiddleware::on_run_started`].
    #[non_exhaustive]
    Terminated {
        /// The middleware-supplied reason.
        reason: String,
    },
    /// A middleware returned [`crate::ToolDecision::Terminate`] from
    /// [`crate::ChatMiddleware::on_before_tool_call`] (e.g. `AntiLoop`
    /// or `MaxToolCalls`).
    #[non_exhaustive]
    ToolTerminated {
        /// Name of the tool whose call was refused.
        tool_name: String,
        /// Provider-assigned id of the refused call, the same as
        /// `call_id` on [`crate::ToolCallInfo`].
        call_id: String,
        /// The middleware-supplied reason.
        reason: String,
    },
    /// The run reached [`crate::RunConfig::max_iterations`] while the
    /// model was still requesting tools. Carries the configured cap.
    /// Messages produced up to that point (including every tool
    /// result) are kept in `new_messages`.
    MaxIterations(usize),
}

impl AbortReason {
    /// Build an [`AbortReason::Terminated`]. The struct variants are
    /// `#[non_exhaustive]`; these constructors build them outside
    /// `ailoop-core`.
    pub fn terminated(reason: impl Into<String>) -> Self {
        Self::Terminated {
            reason: reason.into(),
        }
    }

    /// Build an [`AbortReason::ToolTerminated`].
    pub fn tool_terminated(
        tool_name: impl Into<String>,
        call_id: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::ToolTerminated {
            tool_name: tool_name.into(),
            call_id: call_id.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for AbortReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbortReason::Timeout(d) => write!(f, "timeout exceeded after {d:?}"),
            AbortReason::Cancelled => f.write_str("cancelled by caller"),
            AbortReason::Terminated { reason } | AbortReason::ToolTerminated { reason, .. } => {
                f.write_str(reason)
            }
            AbortReason::MaxIterations(n) => {
                write!(f, "agent loop exceeded max iterations ({n})")
            }
        }
    }
}

/// Token counters reported by the provider for a turn.
///
/// Aggregated to the run level by the engine. The run-level total on
/// [`StreamChunk::RunFinished::usage`] is **everything the run spent**:
/// the sum of its own turns plus the usage tools reported through
/// `ToolContext::report_usage` — which `SubAgentTool` does for its
/// child run, recursively. Aborted runs report what was spent up to
/// the cutoff, including tokens a tool reported before it was dropped.
///
/// To split own vs. delegated spend, sum the per-turn values from
/// [`StreamChunk::TurnFinished::usage`] in a middleware: that is the
/// run's own model; the difference to `RunFinished.usage` is what
/// tools reported. A stream consumer
/// iterating `Conversation::stream_with_options` only sees the run
/// total on [`StreamChunk::RunFinished::usage`]; per-turn `Usage`
/// rides on [`StreamChunk::TurnFinished::usage`], which is exposed to
/// [`crate::ChatMiddleware::on_chunk`] but filtered out of the public
/// stream (see the variant docs for the rationale). Fields not
/// surfaced by a given provider stay at `0`.
///
/// A run that ends in `Err` has no `RunFinished`; its spend up to the
/// failure travels in `RunError::usage` and in the `usage` argument of
/// [`crate::ChatMiddleware::on_run_error`], counted the same way. The
/// turn that failed is not in it: a response cut off by an error never
/// reports its usage.
///
/// Use [`StreamChunk::RunFinished::usage`] for end-of-run totals;
/// reach for the middleware path when per-turn metrics matter — a
/// context-size indicator built from the final turn's `input_tokens`,
/// per-turn latency or service-tier attribution, online tokenizer
/// calibration, and similar uses.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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
    /// Subset of `output_tokens` consumed by hidden reasoning steps.
    /// Populated by providers that itemise reasoning separately from
    /// visible output (OpenAI o-series / gpt-5 via
    /// `completion_tokens_details.reasoning_tokens`). Zero when the
    /// provider folds reasoning into `output_tokens` without a
    /// breakdown (current Anthropic behaviour for extended thinking).
    pub reasoning_tokens: u32,
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
            reasoning_tokens: self.reasoning_tokens + other.reasoning_tokens,
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
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abort_reason_display_preserves_legacy_texts() {
        assert_eq!(
            AbortReason::Timeout(Duration::from_millis(50)).to_string(),
            "timeout exceeded after 50ms"
        );
        assert_eq!(AbortReason::Cancelled.to_string(), "cancelled by caller");
        assert_eq!(
            AbortReason::Terminated {
                reason: "policy".into()
            }
            .to_string(),
            "policy"
        );
        assert_eq!(
            AbortReason::tool_terminated("search", "toolu_1", "loop detected").to_string(),
            "loop detected"
        );
        assert_eq!(
            AbortReason::MaxIterations(5).to_string(),
            "agent loop exceeded max iterations (5)"
        );
    }

    fn close(raw: &str) -> StreamChunk {
        StreamChunk::tool_call_from_raw_args("call_1", "write_file", raw)
    }

    #[test]
    fn empty_raw_args_are_an_empty_object() {
        for raw in ["", "  \n"] {
            match close(raw) {
                StreamChunk::ToolCallFinished { id, name, args } => {
                    assert_eq!(id, "call_1");
                    assert_eq!(name, "write_file");
                    assert_eq!(args, serde_json::json!({}));
                }
                other => panic!("expected ToolCallFinished for {raw:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn object_raw_args_finish_the_call() {
        match close(r#"{"path":"a.txt"}"#) {
            StreamChunk::ToolCallFinished { args, .. } => {
                assert_eq!(args, serde_json::json!({"path": "a.txt"}));
            }
            other => panic!("expected ToolCallFinished, got {other:?}"),
        }
    }

    #[test]
    fn truncated_raw_args_are_malformed() {
        match close(r#"{"path":"a"#) {
            StreamChunk::ToolCallMalformed {
                id,
                name,
                raw,
                error,
            } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "write_file");
                assert_eq!(raw, r#"{"path":"a"#);
                assert!(error.contains("EOF"), "unexpected error: {error}");
            }
            other => panic!("expected ToolCallMalformed, got {other:?}"),
        }
    }

    #[test]
    fn non_object_raw_args_are_malformed() {
        for raw in ["[1]", "null", r#""x""#] {
            match close(raw) {
                StreamChunk::ToolCallMalformed {
                    raw: got, error, ..
                } => {
                    assert_eq!(got, raw);
                    assert_eq!(error, "tool arguments must be a JSON object");
                }
                other => panic!("expected ToolCallMalformed for {raw:?}, got {other:?}"),
            }
        }
    }
}
