use crate::errors::{AnthropicError, AnthropicApiErrorKind};
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
        let mut service_tier: Option<String> = None;

        while let Some(event) = events.next().await {
            let parsed = event?;

            match parsed {
                AnthropicEvent::MessageStart {message} => {
                    usage.input_tokens = message.usage.input_tokens;
                    usage.cached_input_tokens = message.usage.cache_read_input_tokens;
                    apply_cache_creation(
                        &mut usage,
                        message.usage.cache_creation_input_tokens,
                        message.usage.cache_creation.as_ref(),
                    );
                    if message.usage.service_tier.is_some() {
                        service_tier = message.usage.service_tier;
                    }
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
                    if u.cache_read_input_tokens > 0 {
                        usage.cached_input_tokens = u.cache_read_input_tokens;
                    }
                    if u.cache_creation_input_tokens > 0 || u.cache_creation.is_some() {
                        apply_cache_creation(
                            &mut usage,
                            u.cache_creation_input_tokens,
                            u.cache_creation.as_ref(),
                        );
                    }
                }
                AnthropicEvent::MessageStop => {}
                AnthropicEvent::Ping => { }
                AnthropicEvent::Error { error } => {
                    Err(AnthropicError::Provider {
                        kind: AnthropicApiErrorKind::from_error_type(&error.error_type),
                        message: error.message,
                    })?;
                }
            }
        }
        yield StreamChunk::TurnFinished {
            reason: final_stop,
            usage,
            service_tier,
        };
    };

    Box::pin(stream)
}

/// Reconcile Anthropic's two cache-creation reporting shapes into the
/// fields on [`Usage`]. When the API ships the new TTL breakdown we
/// trust it for both totals and per-bucket counters; when only the flat
/// counter is present we still populate the total. The legacy and new
/// shapes can coexist on the same payload — callers send both in
/// parallel for compat — so the breakdown wins when present because it
/// is strictly more informative.
fn apply_cache_creation(
    usage: &mut Usage,
    flat_input_tokens: u32,
    breakdown: Option<&crate::events::CacheCreationBreakdown>,
) {
    match breakdown {
        Some(b) => {
            usage.cache_creation_5m_tokens = b.ephemeral_5m_input_tokens;
            usage.cache_creation_1h_tokens = b.ephemeral_1h_input_tokens;
            usage.cache_creation_input_tokens =
                b.ephemeral_5m_input_tokens + b.ephemeral_1h_input_tokens;
        }
        None => {
            usage.cache_creation_input_tokens = flat_input_tokens;
        }
    }
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
                        ..Default::default()
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
                usage: UsageDelta {
                    output_tokens: 50,
                    ..Default::default()
                },
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
            StreamChunk::TurnFinished {
                reason,
                usage,
                service_tier,
            } => {
                assert!(matches!(reason, FinishReason::ToolUse));
                assert_eq!(usage.input_tokens, 12);
                assert_eq!(usage.output_tokens, 50);
                assert!(service_tier.is_none());
            }
            other => panic!("expected TurnFinished, got {other:?}"),
        }
        assert!(
            iter.next().is_none(),
            "no chunks should follow TurnFinished"
        );
    }

    /// Mid-stream `error` events must be classified into `AnthropicApiErrorKind`
    /// so `RetryingModel<M>` can match on `Overloaded` without parsing
    /// strings — same guarantee the HTTP path provides via `Api { kind }`.
    #[tokio::test]
    async fn mid_stream_error_event_is_classified() {
        let events = vec![
            ok(AnthropicEvent::MessageStart {
                message: MessageStartPayload {
                    usage: MessageStartUsage {
                        input_tokens: 1,
                        ..Default::default()
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
                assert_eq!(kind, AnthropicApiErrorKind::Overloaded);
                assert_eq!(message, "busy");
            }
            other => panic!("expected Provider{{Overloaded, ..}}, got {other:?}"),
        }
    }

    /// `message_start` may carry the new `cache_creation` breakdown and
    /// `service_tier`. Both must reach the engine via `TurnFinished`:
    /// the breakdown populates the per-TTL counters on `Usage` and the
    /// flat `cache_creation_input_tokens` is rederived as their sum;
    /// `service_tier` rides on `TurnFinished`.
    #[tokio::test]
    async fn message_start_breakdown_and_service_tier_reach_turn_finished() {
        use crate::events::CacheCreationBreakdown;

        let events = vec![
            ok(AnthropicEvent::MessageStart {
                message: MessageStartPayload {
                    usage: MessageStartUsage {
                        input_tokens: 100,
                        cache_read_input_tokens: 25,
                        cache_creation_input_tokens: 150,
                        cache_creation: Some(CacheCreationBreakdown {
                            ephemeral_5m_input_tokens: 100,
                            ephemeral_1h_input_tokens: 50,
                        }),
                        service_tier: Some("priority".into()),
                    },
                },
            }),
            ok(AnthropicEvent::MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".into()),
                    stop_sequence: None,
                },
                usage: UsageDelta {
                    output_tokens: 7,
                    ..Default::default()
                },
            }),
            ok(AnthropicEvent::MessageStop),
        ];

        let chunks = run(events).await;
        let last = chunks
            .into_iter()
            .last()
            .expect("stream emits at least TurnFinished");
        match last {
            StreamChunk::TurnFinished {
                reason,
                usage,
                service_tier,
            } => {
                assert!(matches!(reason, FinishReason::EndTurn));
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.cached_input_tokens, 25);
                // Total = sum of breakdown when breakdown is present.
                assert_eq!(usage.cache_creation_input_tokens, 150);
                assert_eq!(usage.cache_creation_5m_tokens, 100);
                assert_eq!(usage.cache_creation_1h_tokens, 50);
                assert_eq!(service_tier.as_deref(), Some("priority"));
            }
            other => panic!("expected TurnFinished, got {other:?}"),
        }
    }

    /// When the API provides only the legacy flat `cache_creation_input_tokens`
    /// (no breakdown object), the flat counter still propagates and the
    /// per-TTL fields stay at zero.
    #[tokio::test]
    async fn message_start_legacy_flat_counter_propagates() {
        let events = vec![
            ok(AnthropicEvent::MessageStart {
                message: MessageStartPayload {
                    usage: MessageStartUsage {
                        input_tokens: 10,
                        cache_creation_input_tokens: 42,
                        ..Default::default()
                    },
                },
            }),
            ok(AnthropicEvent::MessageDelta {
                delta: MessageDeltaPayload {
                    stop_reason: Some("end_turn".into()),
                    stop_sequence: None,
                },
                usage: UsageDelta {
                    output_tokens: 1,
                    ..Default::default()
                },
            }),
            ok(AnthropicEvent::MessageStop),
        ];

        let chunks = run(events).await;
        let last = chunks
            .into_iter()
            .last()
            .expect("stream emits at least TurnFinished");
        match last {
            StreamChunk::TurnFinished {
                usage,
                service_tier,
                ..
            } => {
                assert_eq!(usage.cache_creation_input_tokens, 42);
                assert_eq!(usage.cache_creation_5m_tokens, 0);
                assert_eq!(usage.cache_creation_1h_tokens, 0);
                assert!(service_tier.is_none());
            }
            other => panic!("expected TurnFinished, got {other:?}"),
        }
    }
}
