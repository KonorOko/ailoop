use crate::errors::AzureOpenAIError;
use crate::events::ChatCompletionsChunk;

use ailoop_core::{FinishReason, StreamChunk, Usage};
use async_stream::try_stream;
use eventsource_stream::Eventsource;
use futures::Stream;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Response;
use std::collections::{HashMap, HashSet};

pub fn process_response(
    response: Response,
) -> BoxStream<'static, Result<StreamChunk, AzureOpenAIError>> {
    let events = try_stream! {
        let mut sse = response.bytes_stream().eventsource();
        while let Some(event) = sse.next().await {
            let event = event.map_err(AzureOpenAIError::Sse)?;
            // Azure terminates the stream with the literal sentinel
            // `data: [DONE]` (not JSON). Skip it; the consumer-side state
            // machine emits TurnFinished from the recorded finish_reason.
            if event.data == "[DONE]" || event.data.is_empty() {
                continue;
            }
            let parsed: ChatCompletionsChunk = serde_json::from_str(&event.data)
                .map_err(AzureOpenAIError::Json)?;
            yield parsed;
        }
    };
    process_events(Box::pin(events))
}

struct ToolCallState {
    id: String,
    name: String,
    args_buf: String,
}

/// State-machine half of the SSE pipeline. Consumes parsed
/// `ChatCompletionsChunk`s and emits `StreamChunk`s in wire order. Split
/// from `process_response` so it can be exercised with synthetic chunk
/// vectors in unit tests.
pub(crate) fn process_events<S>(
    events: S,
) -> BoxStream<'static, Result<StreamChunk, AzureOpenAIError>>
where
    S: Stream<Item = Result<ChatCompletionsChunk, AzureOpenAIError>> + Send + Unpin + 'static,
{
    let stream = try_stream! {
        let mut events = events;
        let mut tool_calls: HashMap<u32, ToolCallState> = HashMap::new();
        let mut emitted_starts: HashSet<u32> = HashSet::new();
        let mut tool_call_order: Vec<u32> = Vec::new();
        let mut final_finish: Option<String> = None;
        let mut usage = Usage::default();

        while let Some(chunk) = events.next().await {
            let chunk = chunk?;

            // Usage chunks arrive after `finish_reason`, just before [DONE].
            // Intermediate chunks carry `usage: null`, which deserializes to
            // None and is skipped here.
            if let Some(u) = chunk.usage {
                usage.input_tokens = u.prompt_tokens;
                usage.output_tokens = u.completion_tokens;
                if let Some(details) = u.prompt_tokens_details {
                    usage.cached_input_tokens = details.cached_tokens;
                }
                // Azure does not report cache writes (Anthropic-only concept).
            }

            for choice in chunk.choices {
                if let Some(text) = choice.delta.content
                    && !text.is_empty()
                {
                    yield StreamChunk::TextDelta { delta: text };
                }

                for tc in choice.delta.tool_calls {
                    let idx = tc.index;

                    // First chunk for this tool_call index carries id + name.
                    // Subsequent chunks omit them and only stream args fragments.
                    if !emitted_starts.contains(&idx) {
                        let name = tc.function.as_ref().and_then(|f| f.name.as_ref());
                        if let (Some(id), Some(name)) = (tc.id.as_ref(), name) {
                            tool_calls.insert(
                                idx,
                                ToolCallState {
                                    id: id.clone(),
                                    name: name.clone(),
                                    args_buf: String::new(),
                                },
                            );
                            emitted_starts.insert(idx);
                            tool_call_order.push(idx);
                            yield StreamChunk::ToolCallStart {
                                id: id.clone(),
                                name: name.clone(),
                            };
                        }
                    }

                    if let Some(args_chunk) =
                        tc.function.as_ref().and_then(|f| f.arguments.as_ref())
                        && !args_chunk.is_empty()
                        && let Some(state) = tool_calls.get_mut(&idx)
                    {
                        state.args_buf.push_str(args_chunk);
                        yield StreamChunk::ToolCallArgsDelta {
                            id: state.id.clone(),
                            delta: args_chunk.clone(),
                        };
                    }
                }

                // Chat Completions has no explicit "tool call done" event;
                // finish_reason on the choice closes any pending tool calls.
                if let Some(reason) = choice.finish_reason {
                    final_finish = Some(reason);
                    let order = std::mem::take(&mut tool_call_order);
                    for idx in order {
                        if let Some(state) = tool_calls.remove(&idx) {
                            let args = serde_json::from_str(&state.args_buf)
                                .unwrap_or(serde_json::json!({}));
                            yield StreamChunk::ToolCallEnd {
                                id: state.id,
                                name: state.name,
                                args,
                            };
                        }
                    }
                }
            }
        }

        let reason = map_finish_reason(final_finish.as_deref());
        yield StreamChunk::TurnFinished {
            reason,
            usage,
            // Chat Completions does not surface a service tier; leave it
            // None so middlewares treat the absence the same way they
            // treat any other Chat Completions deployment.
            service_tier: None,
        };
    };

    Box::pin(stream)
}

pub(crate) fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") | None => FinishReason::EndTurn,
        Some("length") => FinishReason::MaxTokens,
        // `function_call` is the legacy single-call shape. Azure still
        // emits it for older deployments configured with the deprecated
        // `functions` parameter; treat it the same as `tool_calls`.
        Some("tool_calls") | Some("function_call") => FinishReason::ToolUse,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use serde_json::json;

    fn parse(s: &str) -> Result<ChatCompletionsChunk, AzureOpenAIError> {
        Ok(serde_json::from_str(s).unwrap())
    }

    async fn run(
        events: Vec<Result<ChatCompletionsChunk, AzureOpenAIError>>,
    ) -> Vec<StreamChunk> {
        let mut out = Vec::new();
        let mut s = process_events(stream::iter(events));
        while let Some(chunk) = s.next().await {
            out.push(chunk.expect("state machine should not error on this fixture"));
        }
        out
    }

    #[tokio::test]
    async fn text_turn_emits_deltas_and_endturn() {
        let events = vec![
            parse(r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#),
            parse(r#"{"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#),
            parse(r#"{"choices":[{"index":0,"delta":{"content":", world"}}]}"#),
            parse(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
            parse(r#"{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3}}"#),
        ];
        let chunks = run(events).await;
        let mut iter = chunks.into_iter();
        match iter.next().unwrap() {
            StreamChunk::TextDelta { delta } => assert_eq!(delta, "Hello"),
            other => panic!("expected TextDelta, got {other:?}"),
        }
        match iter.next().unwrap() {
            StreamChunk::TextDelta { delta } => assert_eq!(delta, ", world"),
            other => panic!("expected TextDelta, got {other:?}"),
        }
        match iter.next().unwrap() {
            StreamChunk::TurnFinished { reason, usage, .. } => {
                assert!(matches!(reason, FinishReason::EndTurn));
                assert_eq!(usage.input_tokens, 5);
                assert_eq!(usage.output_tokens, 3);
            }
            other => panic!("expected TurnFinished, got {other:?}"),
        }
        assert!(iter.next().is_none());
    }

    #[tokio::test]
    async fn tool_call_multiplexed_by_index() {
        let events = vec![
            parse(
                r#"{"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
            ),
            parse(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc\":"}}]}}]}"#,
            ),
            parse(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"SF\"}"}}]}}]}"#,
            ),
            parse(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#),
            parse(r#"{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7}}"#),
        ];
        let chunks = run(events).await;
        let mut iter = chunks.into_iter();
        match iter.next().unwrap() {
            StreamChunk::ToolCallStart { id, name } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }
        match iter.next().unwrap() {
            StreamChunk::ToolCallArgsDelta { id, delta } => {
                assert_eq!(id, "call_1");
                assert_eq!(delta, "{\"loc\":");
            }
            other => panic!("expected ToolCallArgsDelta, got {other:?}"),
        }
        match iter.next().unwrap() {
            StreamChunk::ToolCallArgsDelta { id, delta } => {
                assert_eq!(id, "call_1");
                assert_eq!(delta, "\"SF\"}");
            }
            other => panic!("expected ToolCallArgsDelta, got {other:?}"),
        }
        match iter.next().unwrap() {
            StreamChunk::ToolCallEnd { id, name, args } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(args, json!({ "loc": "SF" }));
            }
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }
        match iter.next().unwrap() {
            StreamChunk::TurnFinished { reason, usage, .. } => {
                assert!(matches!(reason, FinishReason::ToolUse));
                assert_eq!(usage.input_tokens, 12);
                assert_eq!(usage.output_tokens, 7);
            }
            other => panic!("expected TurnFinished, got {other:?}"),
        }
        assert!(iter.next().is_none());
    }

    #[tokio::test]
    async fn parallel_tool_calls_dont_mix_state() {
        // Azure can emit two tool_calls indexed 0 and 1 with their args
        // interleaved across chunks.
        let events = vec![
            parse(
                r#"{"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f1","arguments":""}}]}}]}"#,
            ),
            parse(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"f2","arguments":""}}]}}]}"#,
            ),
            parse(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":1}"}}]}}]}"#,
            ),
            parse(
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"y\":2}"}}]}}]}"#,
            ),
            parse(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#),
        ];
        let chunks = run(events).await;
        let mut starts = Vec::new();
        let mut args = Vec::new();
        let mut ends = Vec::new();
        let mut finished = 0;
        for chunk in chunks {
            match chunk {
                StreamChunk::ToolCallStart { id, name } => starts.push((id, name)),
                StreamChunk::ToolCallArgsDelta { id, delta } => args.push((id, delta)),
                StreamChunk::ToolCallEnd { id, args, .. } => ends.push((id, args)),
                StreamChunk::TurnFinished { .. } => finished += 1,
                other => panic!("unexpected chunk {other:?}"),
            }
        }
        assert_eq!(starts, vec![("a".into(), "f1".into()), ("b".into(), "f2".into())]);
        assert_eq!(
            args,
            vec![
                ("a".into(), "{\"x\":1}".into()),
                ("b".into(), "{\"y\":2}".into()),
            ]
        );
        assert_eq!(
            ends,
            vec![
                ("a".into(), json!({ "x": 1 })),
                ("b".into(), json!({ "y": 2 })),
            ]
        );
        assert_eq!(finished, 1);
    }

    #[tokio::test]
    async fn usage_in_final_chunk_populates_turn_finished() {
        // Intermediate chunks carry `usage: null` (which deserializes to
        // None); the final pre-[DONE] chunk has empty choices and the real
        // usage including prompt_tokens_details.cached_tokens.
        let events = vec![
            parse(r#"{"choices":[{"index":0,"delta":{"content":"hi"},"usage":null}]}"#),
            parse(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
            parse(
                r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":80}}}"#,
            ),
        ];
        let chunks = run(events).await;
        let last = chunks.last().unwrap();
        match last {
            StreamChunk::TurnFinished { usage, .. } => {
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 50);
                assert_eq!(usage.cached_input_tokens, 80);
                assert_eq!(usage.cache_creation_input_tokens, 0);
            }
            other => panic!("expected TurnFinished, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn length_finish_reason_maps_to_max_tokens() {
        let events = vec![
            parse(r#"{"choices":[{"index":0,"delta":{"content":"truncated"}}]}"#),
            parse(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}"#),
        ];
        let chunks = run(events).await;
        let last = chunks.last().unwrap();
        assert!(matches!(
            last,
            StreamChunk::TurnFinished {
                reason: FinishReason::MaxTokens,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn unknown_finish_reason_maps_to_other() {
        // `content_filter` is real but not one of the four canonical ailoop
        // variants, so it round-trips through `Other`.
        let events = vec![parse(
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}"#,
        )];
        let chunks = run(events).await;
        match chunks.last().unwrap() {
            StreamChunk::TurnFinished { reason, .. } => match reason {
                FinishReason::Other(s) => assert_eq!(s, "content_filter"),
                other => panic!("expected Other, got {other:?}"),
            },
            other => panic!("expected TurnFinished, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn emits_exactly_one_turn_finished() {
        let events = vec![
            parse(r#"{"choices":[{"index":0,"delta":{"content":"a"}}]}"#),
            parse(r#"{"choices":[{"index":0,"delta":{"content":"b"}}]}"#),
            parse(r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
            parse(r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}"#),
        ];
        let chunks = run(events).await;
        let count = chunks
            .iter()
            .filter(|c| matches!(c, StreamChunk::TurnFinished { .. }))
            .count();
        assert_eq!(count, 1);
    }

    fn parse_sse_fixture(text: &str) -> Vec<Result<ChatCompletionsChunk, AzureOpenAIError>> {
        let mut events = Vec::new();
        for line in text.lines() {
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload == "[DONE]" || payload.is_empty() {
                continue;
            }
            match serde_json::from_str::<ChatCompletionsChunk>(payload) {
                Ok(c) => events.push(Ok(c)),
                Err(e) => events.push(Err(AzureOpenAIError::Json(e))),
            }
        }
        events
    }

    #[tokio::test]
    async fn fixture_text_turn_emits_text_and_endturn() {
        let sse = include_str!("../tests/fixtures/chat_text.sse");
        let events = parse_sse_fixture(sse);
        let chunks = run(events).await;

        let text_count = chunks
            .iter()
            .filter(|c| matches!(c, StreamChunk::TextDelta { .. }))
            .count();
        assert!(
            text_count >= 1,
            "expected at least one TextDelta, got {chunks:?}"
        );

        let last = chunks.last().expect("non-empty chunks");
        match last {
            StreamChunk::TurnFinished { reason, usage, .. } => {
                assert!(
                    matches!(reason, FinishReason::EndTurn),
                    "expected EndTurn, got {reason:?}"
                );
                assert!(usage.input_tokens > 0, "input_tokens > 0");
                assert!(usage.output_tokens > 0, "output_tokens > 0");
            }
            other => panic!("last chunk should be TurnFinished, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fixture_tool_call_turn_emits_tool_lifecycle() {
        let sse = include_str!("../tests/fixtures/chat_tool_call.sse");
        let events = parse_sse_fixture(sse);
        let chunks = run(events).await;

        let mut starts = 0;
        let mut args_deltas = 0;
        let mut ends: Vec<serde_json::Value> = Vec::new();
        let mut finished: Option<FinishReason> = None;
        for chunk in &chunks {
            match chunk {
                StreamChunk::ToolCallStart { id, name } => {
                    assert!(!id.is_empty(), "tool call id must not be empty");
                    assert!(!name.is_empty(), "tool call name must not be empty");
                    starts += 1;
                }
                StreamChunk::ToolCallArgsDelta { .. } => args_deltas += 1,
                StreamChunk::ToolCallEnd { args, .. } => ends.push(args.clone()),
                StreamChunk::TurnFinished { reason, .. } => finished = Some(reason.clone()),
                _ => {}
            }
        }
        assert_eq!(starts, 1, "expected one ToolCallStart");
        assert!(args_deltas >= 1, "expected at least one args delta");
        assert_eq!(ends.len(), 1, "expected one ToolCallEnd");
        assert!(
            ends[0].is_object(),
            "tool call args should parse to a JSON object, got {:?}",
            ends[0]
        );
        let reason = finished.expect("missing TurnFinished");
        assert!(
            matches!(reason, FinishReason::ToolUse),
            "expected ToolUse, got {reason:?}"
        );
    }

    #[test]
    fn map_finish_reason_covers_all_known_variants() {
        assert!(matches!(
            map_finish_reason(Some("stop")),
            FinishReason::EndTurn
        ));
        assert!(matches!(map_finish_reason(None), FinishReason::EndTurn));
        assert!(matches!(
            map_finish_reason(Some("length")),
            FinishReason::MaxTokens
        ));
        assert!(matches!(
            map_finish_reason(Some("tool_calls")),
            FinishReason::ToolUse
        ));
        assert!(matches!(
            map_finish_reason(Some("function_call")),
            FinishReason::ToolUse
        ));
        match map_finish_reason(Some("content_filter")) {
            FinishReason::Other(s) => assert_eq!(s, "content_filter"),
            other => panic!("expected Other, got {other:?}"),
        }
    }
}
