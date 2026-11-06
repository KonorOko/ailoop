use std::collections::HashMap;

use ailoop_core::{FinishReason, StreamChunk, Usage};
use async_stream::try_stream;
use eventsource_stream::{EventStreamError, Eventsource};
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use reqwest::Response;
use serde_json::{Value, json};

use crate::error_body::{RequestFeatures, classify, parse_error_fields};
use crate::errors::OpenAIError;
use crate::events::ChatCompletionsChunk;

/// One parsed SSE event.
#[derive(Debug)]
pub(crate) enum WireEvent {
    Chunk(ChatCompletionsChunk),
    /// The `data: [DONE]` sentinel.
    Done,
}

pub(crate) fn process_response(
    response: Response,
    features: RequestFeatures,
) -> BoxStream<'static, Result<StreamChunk, OpenAIError>> {
    let events = try_stream! {
        let mut sse = response.bytes_stream().eventsource();
        while let Some(event) = sse.next().await {
            let event = event.map_err(sse_error)?;
            if event.data == "[DONE]" {
                yield WireEvent::Done;
                break;
            }
            if event.data.trim().is_empty() {
                continue;
            }
            let parsed: ChatCompletionsChunk = serde_json::from_str(&event.data)?;
            yield WireEvent::Chunk(parsed);
        }
    };
    process_events(Box::pin(events), features)
}

fn sse_error(error: EventStreamError<reqwest::Error>) -> OpenAIError {
    match error {
        EventStreamError::Transport(source) => OpenAIError::Interrupted {
            message: format!("the connection failed while reading the response: {source}"),
            source: Some(source),
        },
        other => OpenAIError::Sse(other.to_string()),
    }
}

fn stream_error(error: Value, features: RequestFeatures) -> OpenAIError {
    let fields = parse_error_fields(&json!({ "error": error }));
    OpenAIError::Provider {
        kind: classify(None, &fields, features),
        message: fields
            .message
            .unwrap_or_else(|| "the server reported an error while streaming".into()),
    }
}

struct ToolCallState {
    id: String,
    name: String,
    args_buf: String,
}

/// Tool-call bookkeeping. Calls are keyed by the wire `index`; a server
/// that omits it sends every call whole, so a new `id` opens a new slot
/// and a fragment without one continues the last slot.
#[derive(Default)]
struct ToolCalls {
    states: HashMap<u32, ToolCallState>,
    order: Vec<u32>,
    last_slot: Option<u32>,
    next_free: u32,
    closed_this_turn: usize,
}

impl ToolCalls {
    fn slot_for(&mut self, index: Option<u32>, id: Option<&str>) -> u32 {
        let slot = match index {
            Some(i) => i,
            None => match id {
                Some(id) => self
                    .states
                    .iter()
                    .find(|(_, s)| s.id == id)
                    .map(|(slot, _)| *slot)
                    .unwrap_or(self.next_free),
                None => self.last_slot.unwrap_or(self.next_free),
            },
        };
        self.next_free = self.next_free.max(slot + 1);
        self.last_slot = Some(slot);
        slot
    }

    fn close_all(&mut self) -> Vec<StreamChunk> {
        let order = std::mem::take(&mut self.order);
        let mut out = Vec::new();
        for slot in order {
            if let Some(state) = self.states.remove(&slot) {
                self.closed_this_turn += 1;
                // Invalid JSON (for example cut off by `length`) closes
                // as ToolCallMalformed, never as `{}`.
                out.push(StreamChunk::tool_call_from_raw_args(
                    state.id,
                    state.name,
                    state.args_buf,
                ));
            }
        }
        out
    }
}

/// State machine over parsed events, split from [`process_response`] so
/// tests can feed it synthetic events.
pub(crate) fn process_events<S>(
    events: S,
    features: RequestFeatures,
) -> BoxStream<'static, Result<StreamChunk, OpenAIError>>
where
    S: Stream<Item = Result<WireEvent, OpenAIError>> + Send + Unpin + 'static,
{
    let stream = try_stream! {
        let mut events = events;
        let mut calls = ToolCalls::default();
        let mut reasoning_open = false;
        let mut final_finish: Option<String> = None;
        let mut saw_done = false;
        let mut usage = Usage::default();

        while let Some(event) = events.next().await {
            let chunk = match event? {
                WireEvent::Done => {
                    saw_done = true;
                    break;
                }
                WireEvent::Chunk(chunk) => chunk,
            };

            if let Some(error) = chunk.error {
                Err(stream_error(error, features))?;
            }

            if let Some(u) = chunk.usage {
                usage.input_tokens = u.prompt_tokens;
                usage.output_tokens = u.completion_tokens;
                if let Some(details) = u.prompt_tokens_details {
                    usage.cached_input_tokens = details.cached_tokens;
                }
                if let Some(details) = u.completion_tokens_details {
                    usage.reasoning_tokens = details.reasoning_tokens;
                }
            }

            // Only the first choice is read: ailoop never asks for `n > 1`.
            for choice in chunk.choices.into_iter().filter(|c| c.index == 0) {
                let delta = choice.delta;

                let reasoning = delta.reasoning_content.or(delta.reasoning);
                if let Some(text) = reasoning
                    && !text.is_empty()
                {
                    reasoning_open = true;
                    yield StreamChunk::ReasoningDelta { delta: text };
                }

                if let Some(text) = delta.content
                    && !text.is_empty()
                {
                    if reasoning_open {
                        reasoning_open = false;
                        yield StreamChunk::ReasoningFinished { signature: None };
                    }
                    yield StreamChunk::TextDelta { delta: text };
                }

                for tc in delta.tool_calls.unwrap_or_default() {
                    let slot = calls.slot_for(tc.index, tc.id.as_deref());
                    let function = tc.function;

                    // The first fragment of a call carries its name (and
                    // usually its id); later ones only argument text.
                    let opening = if calls.states.contains_key(&slot) {
                        None
                    } else {
                        function
                            .as_ref()
                            .and_then(|f| f.name.clone())
                            .filter(|n| !n.is_empty())
                    };
                    if let Some(name) = opening {
                        // Some local servers leave the id empty; the
                        // engine still needs one to pair the result.
                        let id = tc
                            .id
                            .filter(|id| !id.is_empty())
                            .unwrap_or_else(|| format!("call_{slot}"));
                        if reasoning_open {
                            reasoning_open = false;
                            yield StreamChunk::ReasoningFinished { signature: None };
                        }
                        calls.states.insert(
                            slot,
                            ToolCallState {
                                id: id.clone(),
                                name: name.clone(),
                                args_buf: String::new(),
                            },
                        );
                        calls.order.push(slot);
                        yield StreamChunk::ToolCallStarted { call_id: id, name };
                    }

                    if let Some(args) = function.and_then(|f| f.arguments)
                        && !args.is_empty()
                        && let Some(state) = calls.states.get_mut(&slot)
                    {
                        state.args_buf.push_str(&args);
                        yield StreamChunk::ToolCallArgsDelta {
                            call_id: state.id.clone(),
                            delta: args,
                        };
                    }
                }

                if let Some(reason) = choice.finish_reason {
                    final_finish = Some(reason);
                    if reasoning_open {
                        reasoning_open = false;
                        yield StreamChunk::ReasoningFinished { signature: None };
                    }
                    for chunk in calls.close_all() {
                        yield chunk;
                    }
                }
            }
        }

        // Without a finish reason or `[DONE]` the body ended early: the
        // connection closed cleanly but the answer is incomplete.
        if final_finish.is_none() && !saw_done {
            Err(OpenAIError::Interrupted {
                message: "the stream ended before the model finished".into(),
                source: None,
            })?;
        }

        if reasoning_open {
            yield StreamChunk::ReasoningFinished { signature: None };
        }
        for chunk in calls.close_all() {
            yield chunk;
        }

        let mut reason = map_finish_reason(final_finish.as_deref());
        // Older Ollama reports `stop` after calling tools.
        if matches!(reason, FinishReason::EndTurn) && calls.closed_this_turn > 0 {
            reason = FinishReason::ToolUse;
        }
        yield StreamChunk::TurnFinished {
            reason,
            usage,
            service_tier: None,
            model: None,
        };
    };

    Box::pin(stream)
}

pub(crate) fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") | Some("eos") | Some("end_turn") | None => FinishReason::EndTurn,
        Some("length") | Some("max_tokens") => FinishReason::MaxTokens,
        Some("tool_calls") | Some("function_call") | Some("tool_use") => FinishReason::ToolUse,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use serde_json::json;

    fn chunk(s: &str) -> Result<WireEvent, OpenAIError> {
        Ok(WireEvent::Chunk(serde_json::from_str(s).unwrap()))
    }

    async fn collect(
        events: Vec<Result<WireEvent, OpenAIError>>,
    ) -> (Vec<StreamChunk>, Option<OpenAIError>) {
        let mut out = Vec::new();
        let mut s = process_events(stream::iter(events), RequestFeatures::default());
        while let Some(item) = s.next().await {
            match item {
                Ok(c) => out.push(c),
                Err(e) => return (out, Some(e)),
            }
        }
        (out, None)
    }

    #[tokio::test]
    async fn text_turn() {
        let (chunks, err) = collect(vec![
            chunk(r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#),
            chunk(r#"{"choices":[{"index":0,"delta":{"content":"Hola"}}]}"#),
            chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
            chunk(r#"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}"#),
            Ok(WireEvent::Done),
        ])
        .await;
        assert!(err.is_none());
        assert!(matches!(&chunks[0], StreamChunk::TextDelta { delta } if delta == "Hola"));
        match &chunks[1] {
            StreamChunk::TurnFinished { reason, usage, .. } => {
                assert!(matches!(reason, FinishReason::EndTurn));
                assert_eq!((usage.input_tokens, usage.output_tokens), (5, 3));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn ending_without_finish_or_done_is_interrupted() {
        let (chunks, err) = collect(vec![chunk(
            r#"{"choices":[{"index":0,"delta":{"content":"Hol"}}]}"#,
        )])
        .await;
        assert_eq!(chunks.len(), 1);
        assert!(matches!(err, Some(OpenAIError::Interrupted { .. })));
    }

    #[tokio::test]
    async fn done_without_finish_reason_is_a_normal_end() {
        let (chunks, err) = collect(vec![
            chunk(r#"{"choices":[{"index":0,"delta":{"content":"ok"}}]}"#),
            Ok(WireEvent::Done),
        ])
        .await;
        assert!(err.is_none());
        assert!(matches!(
            chunks.last(),
            Some(StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn reasoning_is_closed_before_text() {
        let (chunks, _) = collect(vec![
            chunk(r#"{"choices":[{"index":0,"delta":{"reasoning":"pienso"}}]}"#),
            chunk(r#"{"choices":[{"index":0,"delta":{"content":"listo"}}]}"#),
            chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
        ])
        .await;
        assert!(matches!(&chunks[0], StreamChunk::ReasoningDelta { delta } if delta == "pienso"));
        assert!(matches!(
            &chunks[1],
            StreamChunk::ReasoningFinished { signature: None }
        ));
        assert!(matches!(&chunks[2], StreamChunk::TextDelta { .. }));
    }

    #[tokio::test]
    async fn whole_tool_calls_without_index_and_stop_reason() {
        // Older Ollama: each call whole, no `index`, finish_reason `stop`.
        let (chunks, err) = collect(vec![
            chunk(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"a","type":"function","function":{"name":"f1","arguments":"{\"x\":1}"}},{"id":"b","type":"function","function":{"name":"f2","arguments":"{}"}}]}}]}"#,
            ),
            chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
        ])
        .await;
        assert!(err.is_none());
        let finished: Vec<_> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::ToolCallFinished {
                    call_id,
                    name,
                    args,
                    ..
                } => Some((call_id.clone(), name.clone(), args.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            finished,
            vec![
                ("a".into(), "f1".into(), json!({"x": 1})),
                ("b".into(), "f2".into(), json!({})),
            ]
        );
        assert!(matches!(
            chunks.last(),
            Some(StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn missing_tool_call_id_gets_one() {
        let (chunks, _) = collect(vec![
            chunk(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"","function":{"name":"f","arguments":"{}"}}]}}]}"#,
            ),
            chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#),
        ])
        .await;
        assert!(
            matches!(&chunks[0], StreamChunk::ToolCallStarted { call_id, .. } if call_id == "call_0")
        );
    }

    #[tokio::test]
    async fn truncated_arguments_are_malformed() {
        let (chunks, _) = collect(vec![
            chunk(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"w","arguments":"{\"p\": \"a"}}]}}]}"#,
            ),
            chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#),
        ])
        .await;
        assert!(
            chunks
                .iter()
                .any(|c| matches!(c, StreamChunk::ToolCallMalformed { .. }))
        );
        assert!(matches!(
            chunks.last(),
            Some(StreamChunk::TurnFinished {
                reason: FinishReason::MaxTokens,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn error_event_string_or_object() {
        let (_, err) = collect(vec![chunk(r#"{"error":"model crashed"}"#)]).await;
        match err {
            Some(OpenAIError::Provider { message, .. }) => assert_eq!(message, "model crashed"),
            other => panic!("{other:?}"),
        }
        let (_, err) = collect(vec![chunk(
            r#"{"error":{"code":"context_length_exceeded","message":"too long"}}"#,
        )])
        .await;
        assert!(matches!(
            err,
            Some(OpenAIError::Provider {
                kind: crate::OpenAIApiErrorKind::ContextOverflow,
                ..
            })
        ));
    }

    #[test]
    fn finish_reasons() {
        assert!(matches!(map_finish_reason(None), FinishReason::EndTurn));
        assert!(matches!(
            map_finish_reason(Some("length")),
            FinishReason::MaxTokens
        ));
        assert!(matches!(
            map_finish_reason(Some("tool_calls")),
            FinishReason::ToolUse
        ));
        assert!(
            matches!(map_finish_reason(Some("content_filter")), FinishReason::Other(s) if s == "content_filter")
        );
    }
}
