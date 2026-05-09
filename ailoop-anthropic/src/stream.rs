use crate::errors::{AnthropicError, ApiErrorKind};
use crate::events::{AnthropicBlock, AnthropicDelta, AnthropicEvent};

use ailoop_core::{FinishReason, StreamChunk, Usage};
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures::Stream;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Response;
use std::collections::HashMap;

pub fn process_response(
    response: Response,
) -> BoxStream<'static, Result<StreamChunk, AnthropicError>> {
    let events = try_stream! {
        let mut sse = response.bytes_stream().eventsource();
        while let Some(event) = sse.next().await {
            let event = event.map_err(AnthropicError::Sse)?;
            if event.data.is_empty() { continue; }
            let parsed: AnthropicEvent = serde_json::from_str(&event.data)
                .map_err(AnthropicError::Json)?;
            yield parsed;
        }
    };
    process_events(Box::pin(events))
}

/// State-machine half of the SSE pipeline. Consumes a stream of parsed
/// `AnthropicEvent`s and emits `StreamChunk`s in wire order. Split from
/// `process_response` so it can be exercised with synthetic event sequences
/// in unit tests.
pub(crate) fn process_events<S>(
    events: S,
) -> BoxStream<'static, Result<StreamChunk, AnthropicError>>
where
    S: Stream<Item = Result<AnthropicEvent, AnthropicError>> + Send + Unpin + 'static,
{
    let stream = try_stream! {
        let mut events = events;
        let mut blocks: HashMap<u32, BlockState> = HashMap::new();
        let mut final_stop = FinishReason::EndTurn;
        let mut usage = Usage::default();

        while let Some(event) = events.next().await {
            let parsed = event?;

            match parsed {
                AnthropicEvent::MessageStart {message} => {
                    usage.input_tokens = message.usage.input_tokens;
                    usage.cached_input_tokens = message.usage.cache_read_input_tokens;
                    usage.cache_creation_input_tokens = message.usage.cache_creation_input_tokens;
                }
                AnthropicEvent::ContentBlockStart {index, content_block} => {
                    match content_block {
                        AnthropicBlock::Text {..} => {
                            blocks.insert(index, BlockState::Text);
                        }
                        AnthropicBlock::ToolUse {id, name, ..} => {
                            blocks.insert(index, BlockState::ToolUse { id: id.clone(), name: name.clone(), args_buf: String::new() });
                            yield StreamChunk::ToolCallStart { id, name };
                        }
                        AnthropicBlock::Thinking { thinking, signature } => {
                            // start values are usually empty; deltas fill them in.
                            blocks.insert(index, BlockState::Thinking {
                                signature: if signature.is_empty() { None } else { Some(signature) },
                            });
                            if !thinking.is_empty() {
                                yield StreamChunk::ReasoningDelta { delta: thinking };
                            }
                        }
                        AnthropicBlock::RedactedThinking { data } => {
                            // Atomic block: emit immediately, no state needed.
                            yield StreamChunk::RedactedReasoningBlock { data };
                        }
                        AnthropicBlock::Unknown => {
                            // Unknown block type — track as Text-like so deltas
                            // and stop events don't desync state, but emit
                            // nothing.
                            blocks.insert(index, BlockState::Unknown);
                        }
                    }
                }
                AnthropicEvent::ContentBlockDelta {index, delta } => match delta {
                    AnthropicDelta::TextDelta { text } => {
                        yield StreamChunk::TextDelta { delta: text };
                    }
                    AnthropicDelta::InputJsonDelta { partial_json } => {
                        if let Some(BlockState::ToolUse { id, args_buf, .. }) = blocks.get_mut(&index) {
                            args_buf.push_str(&partial_json);
                            yield StreamChunk::ToolCallArgsDelta {
                                id: id.clone(),
                                delta: partial_json,
                            };
                        }
                    }
                    AnthropicDelta::ThinkingDelta { thinking } => {
                        yield StreamChunk::ReasoningDelta { delta: thinking };
                    }
                    AnthropicDelta::SignatureDelta { signature } => {
                        if let Some(BlockState::Thinking { signature: sig_slot }) = blocks.get_mut(&index) {
                            *sig_slot = Some(signature);
                        }
                    }
                    AnthropicDelta::Unknown => {}
                }
                AnthropicEvent::ContentBlockStop { index } => {
                    match blocks.remove(&index) {
                        Some(BlockState::ToolUse { id, name, args_buf }) => {
                            let args = serde_json::from_str(&args_buf)
                                .unwrap_or(serde_json::json!({}));
                            yield StreamChunk::ToolCallEnd { id, name, args };
                        }
                        Some(BlockState::Thinking { signature }) => {
                            yield StreamChunk::ReasoningEnd { signature };
                        }
                        Some(BlockState::Text) | Some(BlockState::Unknown) | None => {}
                    }
                }
                AnthropicEvent::MessageDelta { delta, usage: u } => {
                    final_stop = map_stop_reason(&delta.stop_reason);
                    usage.output_tokens = u.output_tokens;
                }
                AnthropicEvent::MessageStop => {}
                AnthropicEvent::Ping => { }
                AnthropicEvent::Error { error } => {
                    Err(AnthropicError::Provider {
                        kind: ApiErrorKind::from_error_type(&error.error_type),
                        message: error.message,
                    })?;
                }
            }
        }
        yield StreamChunk::TurnFinished { reason: final_stop, usage };
    };

    Box::pin(stream)
}

pub fn map_stop_reason(reason: &Option<String>) -> FinishReason {
    match reason.as_deref() {
        Some("end_turn") => FinishReason::EndTurn,
        Some("tool_use") => FinishReason::ToolUse,
        Some("max_tokens") => FinishReason::MaxTokens,
        Some("stop_sequence") => FinishReason::StopSequence,
        Some(other) => FinishReason::Other(other.to_string()),
        None => FinishReason::EndTurn,
    }
}

pub enum BlockState {
    Text,
    ToolUse {
        id: String,
        name: String,
        args_buf: String,
    },
    Thinking {
        signature: Option<String>,
    },
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{
        AnthropicErrorPayload, MessageDeltaPayload, MessageStartPayload, MessageStartUsage,
        UsageDelta,
    };
    use futures::stream;
    use serde_json::json;

    fn ok(event: AnthropicEvent) -> Result<AnthropicEvent, AnthropicError> {
        Ok(event)
    }

    async fn run(events: Vec<Result<AnthropicEvent, AnthropicError>>) -> Vec<StreamChunk> {
        let mut out = Vec::new();
        let mut s = process_events(stream::iter(events));
        while let Some(chunk) = s.next().await {
            out.push(chunk.expect("state machine should not error on this fixture"));
        }
        out
    }

    /// Reproduces the canonical Anthropic stream for an extended-thinking
    /// turn that ends in a tool call: thinking → signature → tool_use →
    /// stop. Verifies that the engine sees ReasoningDelta×N + ReasoningEnd
    /// with the signature, then ToolCallStart, ToolCallArgsDelta×N,
    /// ToolCallEnd with parsed args, and finally TurnFinished{ToolUse}.
    #[tokio::test]
    async fn thinking_then_tool_use_round_trip() {
        let events = vec![
            ok(AnthropicEvent::MessageStart {
                message: MessageStartPayload {
                    usage: MessageStartUsage {
                        input_tokens: 12,
                        cache_read_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                    },
                },
            }),
            ok(AnthropicEvent::ContentBlockStart {
                index: 0,
                content_block: AnthropicBlock::Thinking {
                    thinking: String::new(),
                    signature: String::new(),
                },
            }),
            ok(AnthropicEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicDelta::ThinkingDelta {
                    thinking: "Let me ".into(),
                },
            }),
            ok(AnthropicEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicDelta::ThinkingDelta {
                    thinking: "check the weather.".into(),
                },
            }),
            ok(AnthropicEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicDelta::SignatureDelta {
                    signature: "sig-xyz".into(),
                },
            }),
            ok(AnthropicEvent::ContentBlockStop { index: 0 }),
            ok(AnthropicEvent::ContentBlockStart {
                index: 1,
                content_block: AnthropicBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "get_weather".into(),
                    input: json!({}),
                },
            }),
            ok(AnthropicEvent::ContentBlockDelta {
                index: 1,
                delta: AnthropicDelta::InputJsonDelta {
                    partial_json: "{\"location\":".into(),
                },
            }),
            ok(AnthropicEvent::ContentBlockDelta {
                index: 1,
                delta: AnthropicDelta::InputJsonDelta {
                    partial_json: "\"SF\"}".into(),
                },
            }),
            ok(AnthropicEvent::ContentBlockStop { index: 1 }),
            ok(AnthropicEvent::MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("tool_use".into()),
                    stop_sequence: None,
                },
                usage: UsageDelta { output_tokens: 50 },
            }),
            ok(AnthropicEvent::MessageStop),
        ];

        let chunks = run(events).await;

        // Walk the chunks in order. The state machine guarantees:
        //   thinking deltas → reasoning end (with sig) → tool call lifecycle → turn finished
        let mut iter = chunks.into_iter();

        match iter.next().expect("reasoning delta 1") {
            StreamChunk::ReasoningDelta { delta } => assert_eq!(delta, "Let me "),
            other => panic!("expected ReasoningDelta, got {other:?}"),
        }
        match iter.next().expect("reasoning delta 2") {
            StreamChunk::ReasoningDelta { delta } => assert_eq!(delta, "check the weather."),
            other => panic!("expected ReasoningDelta, got {other:?}"),
        }
        match iter.next().expect("reasoning end") {
            StreamChunk::ReasoningEnd { signature } => {
                assert_eq!(signature.as_deref(), Some("sig-xyz"));
            }
            other => panic!("expected ReasoningEnd, got {other:?}"),
        }
        match iter.next().expect("tool call start") {
            StreamChunk::ToolCallStart { id, name } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }
        match iter.next().expect("args delta 1") {
            StreamChunk::ToolCallArgsDelta { id, delta } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(delta, "{\"location\":");
            }
            other => panic!("expected ToolCallArgsDelta, got {other:?}"),
        }
        match iter.next().expect("args delta 2") {
            StreamChunk::ToolCallArgsDelta { id, delta } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(delta, "\"SF\"}");
            }
            other => panic!("expected ToolCallArgsDelta, got {other:?}"),
        }
        match iter.next().expect("tool call end") {
            StreamChunk::ToolCallEnd { id, name, args } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(args, json!({"location": "SF"}));
            }
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }
        match iter.next().expect("turn finished") {
            StreamChunk::TurnFinished { reason, usage } => {
                assert!(matches!(reason, FinishReason::ToolUse));
                assert_eq!(usage.input_tokens, 12);
                assert_eq!(usage.output_tokens, 50);
            }
            other => panic!("expected TurnFinished, got {other:?}"),
        }
        assert!(
            iter.next().is_none(),
            "no chunks should follow TurnFinished"
        );
    }

    /// Mid-stream `error` events must be classified into `ApiErrorKind`
    /// so `RetryingModel<M>` can match on `Overloaded` without parsing
    /// strings — same guarantee the HTTP path provides via `Api { kind }`.
    #[tokio::test]
    async fn mid_stream_error_event_is_classified() {
        let events = vec![
            ok(AnthropicEvent::MessageStart {
                message: MessageStartPayload {
                    usage: MessageStartUsage {
                        input_tokens: 1,
                        cache_read_input_tokens: 0,
                        cache_creation_input_tokens: 0,
                    },
                },
            }),
            ok(AnthropicEvent::Error {
                error: AnthropicErrorPayload {
                    error_type: "overloaded_error".into(),
                    message: "busy".into(),
                },
            }),
        ];

        let mut s = process_events(stream::iter(events));
        let mut last = None;
        while let Some(chunk) = s.next().await {
            last = Some(chunk);
        }

        match last.expect("stream must yield the error") {
            Err(AnthropicError::Provider { kind, message }) => {
                assert_eq!(kind, ApiErrorKind::Overloaded);
                assert_eq!(message, "busy");
            }
            other => panic!("expected Provider{{Overloaded, ..}}, got {other:?}"),
        }
    }
}
