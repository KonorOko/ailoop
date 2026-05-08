use std::{ops::Add, sync::Arc};

use crate::{Message, ToolResultContent};

#[derive(Debug)]
#[non_exhaustive]
pub enum StreamChunk {
    TextDelta {
        delta: String,
    },
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallArgsDelta {
        id: String,
        delta: String,
    },
    ToolCallEnd {
        id: String,
        name: String,
        args: serde_json::Value,
    },
    ReasoningDelta {
        delta: String,
    },
    /// End of a visible reasoning block. Carries the provider signature when
    /// applicable (Anthropic extended thinking); other providers may emit
    /// `None`. Engines should pair this with the accumulated reasoning text
    /// to materialize an `AssistantBlock::Reasoning`.
    ReasoningEnd {
        signature: Option<String>,
    },
    /// A complete redacted reasoning block delivered atomically. `data` is
    /// opaque provider material that must be replayed verbatim on the next
    /// request. Engines should materialize `AssistantBlock::RedactedReasoning`
    /// directly from this chunk; no deltas are emitted around it.
    RedactedReasoningBlock {
        data: String,
    },
    TurnFinished {
        reason: FinishReason,
        usage: Usage,
    },

    // Extend
    RunStarted,
    StepStarted {
        iteration: usize,
    },
    StepFinished {
        iteration: usize,
        new_messages_so_far: Arc<Vec<Message>>,
    },
    ToolResult {
        call_id: String,
        content: ToolResultContent,
    },
    RunFinished {
        reason: FinishReason,
        usage: Usage,
        new_messages: Vec<Message>,
    },
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Aborted(String),
    Other(String),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_input_tokens: u32,
    pub cache_creation_input_tokens: u32,
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
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }
}
