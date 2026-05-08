use crate::errors::AnthropicError;
use crate::events::{AnthropicBlock, AnthropicDelta, AnthropicEvent};

use ailoop_core::{FinishReason, StreamChunk, Usage};
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Response;
use std::collections::HashMap;

pub fn process_response(
    response: Response,
) -> BoxStream<'static, Result<StreamChunk, AnthropicError>> {
    let stream = try_stream! {
        let mut blocks: HashMap<u32, BlockState> = HashMap::new();
        let mut final_stop = FinishReason::EndTurn;
        let mut usage = Usage::default();

        let mut events = response.bytes_stream().eventsource();

        while let Some(event) = events.next().await {
            let event = event.map_err(AnthropicError::Sse)?;
            if event.data.is_empty() {continue;}

            let parsed: AnthropicEvent = serde_json::from_str(&event.data).map_err(AnthropicError::Json)?;

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
                    Err(AnthropicError::Provider { error_type: error.error_type, message: error.message })?;
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
