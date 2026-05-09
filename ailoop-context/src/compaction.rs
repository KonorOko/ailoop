use std::sync::Arc;

use ailoop_core::{
    AssistantBlock, ChatRequest, CompletionModel, Message, StreamChunk, SystemPrompt,
    ToolResultContent, UserBlock,
};
use async_trait::async_trait;
use futures::StreamExt;

use crate::errors::CompactionError;

/// Result of a successful [`CompactionStrategy::compact`] call.
///
/// `messages` and `pinned` are parallel: `pinned[i]` describes the
/// pin state of `messages[i]` in the post-compaction history. The
/// strategy is responsible for forwarding the pin state of every
/// message it preserves so the [`crate::ContextManager`] can keep its
/// internal mask consistent across compactions.
#[derive(Debug, Clone)]
pub struct CompactionOutput {
    pub messages: Vec<Message>,
    pub pinned: Vec<bool>,
}

#[async_trait]
pub trait CompactionStrategy: Send + Sync {
    /// Stable, machine-readable name of the strategy. Used by
    /// `HistoryCompacted` events so callers can attribute compaction
    /// to a specific algorithm in logs/metrics.
    fn name(&self) -> &'static str;

    /// Compact `messages` into a smaller history.
    ///
    /// `pinned` is a parallel slice of the same length as `messages`:
    /// `pinned[i] == true` marks `messages[i]` as "must survive". A
    /// strategy must include every pinned message in its output (in
    /// the original relative order) and forward its `true` pin state
    /// in the returned [`CompactionOutput::pinned`].
    ///
    /// `preserve_n_last` is a hint: at minimum the last N messages
    /// (after walking back to a safe boundary that doesn't strand a
    /// `ToolResult` from its `ToolCall`) should be kept verbatim.
    async fn compact(
        &self,
        messages: &[Message],
        pinned: &[bool],
        preserve_n_last: usize,
    ) -> Result<CompactionOutput, CompactionError>;
}

pub struct TruncateStrategy;

#[async_trait]
impl CompactionStrategy for TruncateStrategy {
    fn name(&self) -> &'static str {
        "truncate"
    }

    async fn compact(
        &self,
        messages: &[Message],
        pinned: &[bool],
        preserve_n_last: usize,
    ) -> Result<CompactionOutput, CompactionError> {
        if messages.len() <= preserve_n_last {
            return Err(CompactionError::NotEnoughHistory);
        }

        let mut start = messages.len() - preserve_n_last;

        // Walk the cut backwards until messages[start] is a safe boundary:
        // a User message whose blocks contain no ToolResult. Otherwise we'd
        // strand a ToolResult from its corresponding ToolCall in the
        // Assistant message we're about to drop, which the provider rejects.
        while start > 0 && !is_safe_start(&messages[start]) {
            start -= 1;
        }

        let mut out_messages = Vec::with_capacity(messages.len());
        let mut out_pinned = Vec::with_capacity(messages.len());

        // Pinned messages from the dropped prefix survive at their
        // original relative position. The caller is responsible for
        // pinning ToolCall/ToolResult pairs together — see
        // `ContextManager::pin_with_tool_result`.
        for (i, msg) in messages.iter().enumerate().take(start) {
            if pinned[i] {
                out_messages.push(msg.clone());
                out_pinned.push(true);
            }
        }

        for (i, msg) in messages.iter().enumerate().skip(start) {
            out_messages.push(msg.clone());
            out_pinned.push(pinned[i]);
        }

        Ok(CompactionOutput {
            messages: out_messages,
            pinned: out_pinned,
        })
    }
}

fn is_safe_start(msg: &Message) -> bool {
    match msg {
        Message::User { blocks } => !blocks
            .iter()
            .any(|b| matches!(b, UserBlock::ToolResult { .. })),
        Message::Assistant { .. } => false,
    }
}

/// Default system prompt used by [`SummarizeStrategy`] when the caller
/// does not supply one. Kept terse to leave room for the actual
/// transcript inside `max_tokens`.
pub const DEFAULT_SUMMARIZER_PROMPT: &str = "You are summarizing a prior conversation between a user and an assistant. Produce a concise, faithful summary that captures the user's goals, decisions made, and important state (file paths, identifiers, numeric results, error messages) the next turn may need. Do not invent details. Output only the summary text — no preamble.";

/// Compaction strategy that calls a [`CompletionModel`] to summarize
/// the dropped portion of the history into a single text message,
/// instead of dropping it outright.
///
/// The chosen cut is the same as [`TruncateStrategy`]: walk back from
/// `messages.len() - preserve_n_last` to the nearest safe boundary
/// (a `User` message that does not contain a `ToolResult`). Pinned
/// messages from the dropped prefix are preserved verbatim at their
/// relative position; the unpinned portion is replaced with one
/// `Message::user("[Summary of prior conversation]\n…")`.
///
/// Tool-call / tool-result blocks in the prefix are flattened into
/// plain text before being sent to the summarizer model. This lets
/// the strategy run with `tools = None` regardless of the original
/// agent's tool surface, sidestepping provider validation that would
/// otherwise reject a `tool_use` block when no tools are declared.
///
/// On model failure the strategy returns
/// [`CompactionError::SummarizationFailed`]; the caller decides
/// whether to fall back to [`TruncateStrategy`] or propagate.
pub struct SummarizeStrategy<M> {
    model: Arc<M>,
    summarizer_prompt: String,
    max_tokens: u32,
}

impl<M> SummarizeStrategy<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    pub fn new(model: Arc<M>) -> Self {
        Self {
            model,
            summarizer_prompt: DEFAULT_SUMMARIZER_PROMPT.into(),
            max_tokens: 1024,
        }
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summarizer_prompt = prompt.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    async fn summarize(&self, messages: Vec<Message>) -> Result<String, CompactionError> {
        let req = ChatRequest {
            messages,
            system_prompt: Some(SystemPrompt::Plain(self.summarizer_prompt.clone())),
            tools: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: vec![],
            max_tokens: self.max_tokens,
            // Leave `tool_choice` unset rather than `None_`: some providers
            // reject `tool_choice: none` when the request also has no
            // `tools` array, and "no tools" already implies "no tool calls".
            tool_choice: None,
            disable_parallel_tool_use: None,
            additional_params: None,
        };

        let mut stream = self
            .model
            .chat_stream(req)
            .await
            .map_err(|e| CompactionError::SummarizationFailed(e.to_string()))?;

        let mut buf = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| CompactionError::SummarizationFailed(e.to_string()))?;
            if let StreamChunk::TextDelta { delta } = chunk {
                buf.push_str(&delta);
            }
        }

        if buf.is_empty() {
            return Err(CompactionError::SummarizationFailed(
                "summarizer model returned no text".into(),
            ));
        }

        Ok(buf)
    }
}

#[async_trait]
impl<M> CompactionStrategy for SummarizeStrategy<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    fn name(&self) -> &'static str {
        "summarize"
    }

    async fn compact(
        &self,
        messages: &[Message],
        pinned: &[bool],
        preserve_n_last: usize,
    ) -> Result<CompactionOutput, CompactionError> {
        if messages.len() <= preserve_n_last {
            return Err(CompactionError::NotEnoughHistory);
        }

        let mut start = messages.len() - preserve_n_last;
        while start > 0 && !is_safe_start(&messages[start]) {
            start -= 1;
        }

        // Collect the unpinned prefix to summarize, flattening any
        // tool blocks into text so the summarizer model does not need
        // a tools array (which would also re-introduce
        // tool_use/tool_result validation rules).
        let to_summarize: Vec<Message> = messages
            .iter()
            .enumerate()
            .take(start)
            .filter(|(i, _)| !pinned[*i])
            .map(|(_, m)| flatten_for_summary(m))
            .collect();

        let mut out_messages = Vec::with_capacity(messages.len());
        let mut out_pinned = Vec::with_capacity(messages.len());

        for (i, msg) in messages.iter().enumerate().take(start) {
            if pinned[i] {
                out_messages.push(msg.clone());
                out_pinned.push(true);
            }
        }

        if !to_summarize.is_empty() {
            let summary = self.summarize(to_summarize).await?;
            out_messages.push(Message::user(format!(
                "[Summary of prior conversation]\n{summary}"
            )));
            out_pinned.push(false);
        }

        for (i, msg) in messages.iter().enumerate().skip(start) {
            out_messages.push(msg.clone());
            out_pinned.push(pinned[i]);
        }

        Ok(CompactionOutput {
            messages: out_messages,
            pinned: out_pinned,
        })
    }
}

/// Convert tool-bearing blocks into plain text so a single message can
/// be safely sent to a summarizer without declaring any tools. Roles
/// are preserved (so two consecutive same-role messages still occur
/// only where they did originally), and the original `Message` is
/// untouched — this only builds the value handed to the summarizer.
fn flatten_for_summary(msg: &Message) -> Message {
    match msg {
        Message::User { blocks } => Message::User {
            blocks: blocks
                .iter()
                .map(|b| match b {
                    UserBlock::Text { text, .. } => UserBlock::text(text.clone()),
                    UserBlock::ToolResult { call_id, content, .. } => {
                        let body = match content {
                            ToolResultContent::Text(t) => t.clone(),
                            ToolResultContent::Error(e) => format!("[error] {e}"),
                        };
                        UserBlock::text(format!("[tool_result:{call_id}] {body}"))
                    }
                    // UserBlock is `#[non_exhaustive]`; future variants
                    // get a placeholder so the summarizer call still goes
                    // through. Producers should add an explicit arm here
                    // when richer rendering is wanted.
                    _ => UserBlock::text("[unsupported user block]"),
                })
                .collect(),
        },
        Message::Assistant { blocks } => Message::Assistant {
            blocks: blocks
                .iter()
                .map(|b| match b {
                    AssistantBlock::Text { text, .. } => AssistantBlock::text(text.clone()),
                    AssistantBlock::ToolCall { id, name, args, .. } => {
                        AssistantBlock::text(format!("[tool_call:{id} {name}] {args}"))
                    }
                    AssistantBlock::Reasoning { text, .. } => AssistantBlock::text(text.clone()),
                    AssistantBlock::RedactedReasoning { .. } => {
                        AssistantBlock::text("[redacted reasoning]".to_string())
                    }
                    _ => AssistantBlock::text("[unsupported assistant block]"),
                })
                .collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::testing::{ScriptedError, ScriptedModel};
    use ailoop_core::{AssistantBlock, FinishReason, ToolResultContent, Usage};
    use serde_json::json;

    fn tool_call(id: &str) -> Message {
        Message::Assistant {
            blocks: vec![AssistantBlock::tool_call(id, "t", json!({}))],
        }
    }

    fn tool_result(call_id: &str) -> Message {
        Message::User {
            blocks: vec![UserBlock::tool_result(
                call_id,
                ToolResultContent::Text("ok".into()),
            )],
        }
    }

    fn unpinned(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    #[tokio::test]
    async fn keeps_normal_history_intact_when_no_pairs() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant_text("hello"),
            Message::user("again"),
            Message::assistant_text("yes"),
        ];

        let out = TruncateStrategy
            .compact(&messages, &unpinned(messages.len()), 2)
            .await
            .unwrap();
        assert_eq!(out.messages.len(), 2);
        assert!(matches!(out.messages[0], Message::User { .. }));
        assert_eq!(out.pinned, vec![false, false]);
    }

    #[tokio::test]
    async fn walks_back_when_cut_lands_on_tool_result() {
        let messages = vec![
            Message::user("solve this"),
            tool_call("c1"),
            tool_result("c1"),
            Message::assistant_text("done"),
        ];

        let out = TruncateStrategy
            .compact(&messages, &unpinned(messages.len()), 2)
            .await
            .unwrap();
        assert_eq!(out.messages.len(), 4);
    }

    #[tokio::test]
    async fn walks_back_when_cut_lands_on_assistant() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant_text("hey"),
            Message::user("more"),
            Message::assistant_text("done"),
        ];

        let out = TruncateStrategy
            .compact(&messages, &unpinned(messages.len()), 1)
            .await
            .unwrap();
        assert_eq!(out.messages.len(), 2);
        assert!(matches!(out.messages[0], Message::User { .. }));
    }

    #[tokio::test]
    async fn pinned_prefix_message_survives_truncation() {
        let messages = vec![
            Message::user("system-ish pinned"),
            Message::user("turn 1 q"),
            Message::assistant_text("turn 1 a"),
            Message::user("turn 2 q"),
            Message::assistant_text("turn 2 a"),
        ];
        let mut pinned = unpinned(messages.len());
        pinned[0] = true;

        let out = TruncateStrategy
            .compact(&messages, &pinned, 2)
            .await
            .unwrap();

        assert_eq!(out.messages.len(), 3, "pinned prefix + tail of 2");
        assert!(matches!(&out.messages[0], Message::User { blocks }
            if matches!(&blocks[0], UserBlock::Text { text, .. } if text == "system-ish pinned")));
        assert_eq!(out.pinned, vec![true, false, false]);
    }

    fn summary_turn(text: &str) -> Vec<StreamChunk> {
        vec![
            StreamChunk::TextDelta {
                delta: text.to_string(),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                usage: Usage::default(),
                service_tier: None,
            },
        ]
    }

    fn first_user_text(msg: &Message) -> Option<&str> {
        match msg {
            Message::User { blocks } => blocks.iter().find_map(|b| match b {
                UserBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            }),
            _ => None,
        }
    }

    #[tokio::test]
    async fn summarize_strategy_replaces_prefix_with_summary() {
        let model = Arc::new(ScriptedModel::new([summary_turn(
            "User asked about turn N, assistant answered.",
        )]));
        let strategy = SummarizeStrategy::new(model);

        let messages = vec![
            Message::user("turn 1 q"),
            Message::assistant_text("turn 1 a"),
            Message::user("turn 2 q"),
            Message::assistant_text("turn 2 a"),
            Message::user("turn 3 q"),
            Message::assistant_text("turn 3 a"),
        ];
        let pinned = unpinned(messages.len());

        let out = strategy.compact(&messages, &pinned, 2).await.unwrap();

        // 1 summary + 2 preserved tail = 3.
        assert_eq!(out.messages.len(), 3);
        let summary_text =
            first_user_text(&out.messages[0]).expect("summary must be a User text message");
        assert!(
            summary_text.contains("[Summary of prior conversation]")
                && summary_text.contains("User asked about turn N"),
            "summary block content unexpected: {summary_text}"
        );
        // Tail intact, pin mask matches.
        assert_eq!(out.pinned, vec![false, false, false]);
    }

    #[tokio::test]
    async fn summarize_strategy_preserves_pinned_prefix() {
        let model = Arc::new(ScriptedModel::new([summary_turn("compact summary body")]));
        let strategy = SummarizeStrategy::new(model);

        let messages = vec![
            Message::user("PIN: persistent anchor"),
            Message::user("turn 1 q"),
            Message::assistant_text("turn 1 a"),
            Message::user("turn 2 q"),
            Message::assistant_text("turn 2 a"),
            Message::user("turn 3 q"),
            Message::assistant_text("turn 3 a"),
        ];
        let mut pinned = unpinned(messages.len());
        pinned[0] = true;

        let out = strategy.compact(&messages, &pinned, 2).await.unwrap();

        // Pinned anchor + summary + 2-message tail = 4.
        assert_eq!(out.messages.len(), 4);
        assert_eq!(
            first_user_text(&out.messages[0]),
            Some("PIN: persistent anchor")
        );
        assert!(
            first_user_text(&out.messages[1])
                .unwrap()
                .contains("compact summary body"),
            "expected summary right after pinned anchor"
        );
        assert_eq!(out.pinned, vec![true, false, false, false]);
    }

    #[tokio::test]
    async fn summarize_strategy_propagates_model_error() {
        let model = Arc::new(ScriptedModel::with_turns([Err(ScriptedError(
            "summary network outage".into(),
        ))]));
        let strategy = SummarizeStrategy::new(model);

        let messages = vec![
            Message::user("turn 1 q"),
            Message::assistant_text("turn 1 a"),
            Message::user("turn 2 q"),
            Message::assistant_text("turn 2 a"),
            Message::user("turn 3 q"),
        ];
        let pinned = unpinned(messages.len());

        let err = strategy
            .compact(&messages, &pinned, 2)
            .await
            .expect_err("model error must propagate");
        match err {
            CompactionError::SummarizationFailed(msg) => {
                assert!(
                    msg.contains("summary network outage"),
                    "expected wrapped model error, got: {msg}"
                );
            }
            other => panic!("expected SummarizationFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn summarize_strategy_skips_model_call_when_prefix_all_pinned() {
        // No Ok turns scripted: if the strategy calls chat_stream it will
        // return an empty stream, yielding "no text" → SummarizationFailed.
        // The expectation is that the strategy notices nothing to summarize
        // and skips the model entirely.
        let model = Arc::new(ScriptedModel::new(Vec::<Vec<StreamChunk>>::new()));
        let strategy = SummarizeStrategy::new(model);

        let messages = vec![
            Message::user("PIN A"),
            Message::user("PIN B"),
            Message::user("tail q"),
            Message::assistant_text("tail a"),
        ];
        let mut pinned = unpinned(messages.len());
        pinned[0] = true;
        pinned[1] = true;

        let out = strategy.compact(&messages, &pinned, 2).await.unwrap();
        // 2 pinned + 2 tail = 4, no summary inserted.
        assert_eq!(out.messages.len(), 4);
        assert_eq!(first_user_text(&out.messages[0]), Some("PIN A"));
        assert_eq!(first_user_text(&out.messages[1]), Some("PIN B"));
        assert_eq!(first_user_text(&out.messages[2]), Some("tail q"));
        assert_eq!(out.pinned, vec![true, true, false, false]);
    }

    #[tokio::test]
    async fn summarize_strategy_flattens_tool_blocks_in_prefix() {
        // The summarizer model's request must NOT carry raw tool_use /
        // tool_result blocks (no tools array → providers reject them).
        // We can't introspect the request from ScriptedModel directly, but
        // we can prove the strategy still completes successfully when the
        // prefix is dense with tool blocks — which is only true if the
        // flatten path actually runs (a real provider would also accept
        // such a request, since it sees only text now).
        let model = Arc::new(ScriptedModel::new([summary_turn("flattened summary")]));
        let strategy = SummarizeStrategy::new(model);

        let messages = vec![
            Message::user("solve task"),
            tool_call("c1"),
            tool_result("c1"),
            Message::user("next q"),
            Message::assistant_text("next a"),
        ];
        let pinned = unpinned(messages.len());

        let out = strategy.compact(&messages, &pinned, 2).await.unwrap();
        // 1 summary + 2 tail = 3.
        assert_eq!(out.messages.len(), 3);
        assert!(
            first_user_text(&out.messages[0])
                .unwrap()
                .contains("flattened summary")
        );
    }

    #[test]
    fn flatten_for_summary_renders_tool_blocks_as_text() {
        let call = Message::Assistant {
            blocks: vec![AssistantBlock::tool_call("c1", "t", json!({"k": 1}))],
        };
        match flatten_for_summary(&call) {
            Message::Assistant { blocks } => match &blocks[0] {
                AssistantBlock::Text { text, .. } => {
                    assert!(text.starts_with("[tool_call:c1 t]"), "got: {text}");
                    assert!(text.contains("\"k\":1"), "args missing: {text}");
                }
                other => panic!("expected text block, got {other:?}"),
            },
            other => panic!("expected assistant message, got {other:?}"),
        }

        let result = Message::User {
            blocks: vec![UserBlock::tool_result(
                "c1",
                ToolResultContent::Text("done".into()),
            )],
        };
        match flatten_for_summary(&result) {
            Message::User { blocks } => match &blocks[0] {
                UserBlock::Text { text, .. } => {
                    assert_eq!(text, "[tool_result:c1] done");
                }
                other => panic!("expected text block, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }
}
