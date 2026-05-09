use std::sync::Arc;

use crate::errors::EngineError;
use ailoop_core::{
    AssistantBlock, ChatMiddleware, ChatRequest, CompletionModel, FinishReason, HookAction,
    RunConfig, RunId, StepId, StreamChunk, ToolDecision, Usage, UserBlock,
};
use ailoop_tools::{ToolRegistry, errors::ToolRegistryError};
pub use async_stream::try_stream;

pub use ailoop_core::{Message, ToolResultContent};
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;

macro_rules! bail_with_hooks {
    ($result: expr, $chain: expr, $run_id: expr) => {
        match $result {
            Ok(v) => Ok(v),
            Err(e) => {
                let err: EngineError<_> = e.into();
                for mw in $chain {
                    mw.on_run_error($run_id, &err).await;
                }
                Err(err)
            }
        }
    };
}

pub async fn run_chat<'a, M: CompletionModel + Sync + Send>(
    model: &'a M,
    messages: Vec<Message>,
    tools: &'a ToolRegistry,
    config: RunConfig,
) -> Result<BoxStream<'a, Result<StreamChunk, EngineError<M::Error>>>, EngineError<M::Error>> {
    let run_id = config.run_id.clone().unwrap_or_default();
    let stream = try_stream! {
        for mw in &config.middlewares {
            match mw.on_run_start(&run_id, &messages, &config).await {
                HookAction::Continue => {},
                HookAction::Terminate {reason} => {
                    let chunk = StreamChunk::RunFinished {
                        run_id: run_id.clone(),
                        reason: FinishReason::Aborted(reason),
                        usage: Usage::default(),
                        new_messages: vec![],
                    };
                    for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                    yield chunk;
                    return;
                }
            };
        }

        let chunk = StreamChunk::RunStarted { run_id: run_id.clone() };
        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
        yield chunk;

        let mut iteration = 0;
        let mut current_messages = messages.to_vec();
        let mut finish_reason = FinishReason::EndTurn;
        let mut usage_run = Usage::default();


        loop {
            if iteration >= config.max_iterations {
                bail_with_hooks!(Err(EngineError::MaxIterationsExceeded(iteration)), &config.middlewares, &run_id)?;
            }

            let step_id = StepId::new();
            let chunk = StreamChunk::StepStarted { run_id: run_id.clone(), step_id: step_id.clone(), iteration };
            for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
            yield chunk;

            let mut assistant_blocks = Vec::new();
            let mut text_buf = String::new();
            let mut reasoning_buf = String::new();

            let mut tool_calls = Vec::new();

            let mut req = ChatRequest {
                messages: current_messages.clone(),
                tools: Some(tools.active_tools().map(|t| t.tool_definition()).collect()),
                system_prompt: config.system_prompt.clone(),
                max_tokens: config.max_tokens,
                additional_params: None,
                temperature: None,
                top_p: None,
                top_k: None,
                stop_sequences: Vec::new(),
            };

            for mw in &config.middlewares {
                mw.on_chat_request(&run_id, &step_id, &mut req).await;
            }

            let mut adapter_stream = bail_with_hooks!(model.chat_stream(req).await.map_err(EngineError::Model), &config.middlewares, &run_id)?;

            while let Some(chunk) = adapter_stream.next().await {
                let chunk = bail_with_hooks!(chunk.map_err(EngineError::Model), &config.middlewares, &run_id)?;

                for mw in &config.middlewares {
                    mw.on_chunk(&chunk).await;
                }

                match &chunk {
                    StreamChunk::TextDelta { delta } => {
                        text_buf.push_str(delta);
                    },
                    StreamChunk::ReasoningDelta { delta } => {
                        reasoning_buf.push_str(delta);
                    },
                    StreamChunk::ToolCallStart { .. } => {
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::Text(std::mem::take(&mut text_buf)));
                        }
                    },
                    StreamChunk::ToolCallEnd { id, name, args } => {
                        assistant_blocks.push(AssistantBlock::ToolCall { id: id.clone(), name: name.clone(), args: args.clone() });
                        tool_calls.push((id.clone(), name.clone(), args.clone()))
                    },
                    StreamChunk::ReasoningEnd { signature } => {
                        // Reasoning blocks must keep their original position
                        // relative to text and tool_use; flush any pending
                        // text first so order on replay matches the wire.
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::Text(std::mem::take(&mut text_buf)));
                        }
                        assistant_blocks.push(AssistantBlock::Reasoning {
                            text: std::mem::take(&mut reasoning_buf),
                            signature: signature.clone(),
                        });
                    },
                    StreamChunk::RedactedReasoningBlock { data } => {
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::Text(std::mem::take(&mut text_buf)));
                        }
                        assistant_blocks.push(AssistantBlock::RedactedReasoning {
                            data: data.clone(),
                        });
                    },
                    StreamChunk::TurnFinished { reason, usage } => {
                        finish_reason = reason.clone();
                        usage_run += *usage;
                        continue;
                    },
                    _=> ()
                }


                yield chunk;
            }

            if !text_buf.is_empty() {
                assistant_blocks.push(AssistantBlock::Text(text_buf));
            }

            if !assistant_blocks.is_empty() {
                current_messages.push(Message::Assistant { blocks: assistant_blocks });
            }

            let mut tools_result = Vec::new();
            for (id, name, args) in tool_calls {

                let decision = run_tool_chain(&config.middlewares, &run_id, &step_id, &name, &args).await;

                let content = match decision {
                    ToolDecision::Continue => {
                        match tools.tool_call(&name, args.clone()).await {
                            Ok(content) => content,
                            Err(ToolRegistryError::NotFound(_)) => {
                                let available_tools: Vec<String> = tools.active_tools().map(|t| t.tool_definition().name).collect();
                                ToolResultContent::Error(format!("Tool '{name}' not found. Available tools: [{}]", available_tools.join(", ")))},
                            Err(other) => bail_with_hooks!(Err(EngineError::Tool(other)), &config.middlewares, &run_id)?,
                        }
                    },
                    ToolDecision::Skip {reason} => {
                        ToolResultContent::Error(format!("Tool skipped: {reason}"))
                    },
                    ToolDecision::Terminate {reason} => {
                        let chunk = StreamChunk::RunFinished {
                            run_id: run_id.clone(),
                            reason: FinishReason::Aborted(reason),
                            usage: usage_run,
                            new_messages: current_messages.split_off(messages.len()),
                        };
                        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                        yield chunk;
                        return;
                    }
                };

                for mw in &config.middlewares {
                    mw.on_after_tool_call(&run_id, &step_id, &name, &args, &content).await;
                }

                let chunk = StreamChunk::ToolResult {
                    run_id: run_id.clone(),
                    step_id: step_id.clone(),
                    call_id: id.clone(),
                    content: content.clone(),
                };
                for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                yield chunk;

                tools_result.push(UserBlock::ToolResult { call_id: id, content });
            }

            if !tools_result.is_empty() {
                current_messages.push(Message::User { blocks: tools_result });
            }

            let chunk = StreamChunk::StepFinished {
                run_id: run_id.clone(),
                step_id: step_id.clone(),
                iteration,
                new_messages_so_far: Arc::new(current_messages[messages.len()..].to_vec()),
            };
            for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
            yield chunk;

            if !matches!(finish_reason, FinishReason::ToolUse) {
                break;
            }

            iteration += 1;
        }

        let new_messages = current_messages.split_off(messages.len());

        for mw in &config.middlewares {
            mw.on_run_finished(&run_id, &finish_reason, &usage_run, &new_messages).await;
        }

        let chunk = StreamChunk::RunFinished {
            run_id: run_id.clone(),
            reason: finish_reason,
            usage: usage_run,
            new_messages,
        };
        for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
        yield chunk;
    };

    Ok(Box::pin(stream))
}

async fn run_tool_chain(
    chain: &[Arc<dyn ChatMiddleware>],
    run_id: &RunId,
    step_id: &StepId,
    name: &str,
    args: &Value,
) -> ToolDecision {
    for mw in chain {
        match mw.on_before_tool_call(run_id, step_id, name, args).await {
            ToolDecision::Continue => continue,
            terminate_or_skip => return terminate_or_skip,
        }
    }
    ToolDecision::Continue
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::ToolDefinition;
    use ailoop_core::testing::ScriptedModel;
    use ailoop_tools::registry::ToolDyn;
    use serde_json::json;

    struct GetWeather;

    #[async_trait::async_trait]
    impl ToolDyn for GetWeather {
        fn name(&self) -> String {
            "get_weather".into()
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "get_weather".into(),
                description: "stub".into(),
                input_schema: json!({"type":"object","properties":{},"required":[]}),
                tags: vec![],
            }
        }
        async fn call(&self, _: serde_json::Value) -> ToolResultContent {
            ToolResultContent::Text("sunny".into())
        }
    }

    /// Engine-level counterpart to the state-machine test in
    /// `ailoop-anthropic`: a thinking turn that ends in a tool call must
    /// land in history as a single assistant message whose blocks are
    /// `[Reasoning{text, signature}, ToolCall{...}]` in that order.
    /// Anthropic rejects requests where the order does not match what was
    /// streamed, so this is load-bearing for tool-use chains.
    #[tokio::test]
    async fn assistant_message_preserves_reasoning_then_tool_call_order() {
        let turn1 = vec![
            StreamChunk::ReasoningDelta {
                delta: "thinking ".into(),
            },
            StreamChunk::ReasoningDelta {
                delta: "step.".into(),
            },
            StreamChunk::ReasoningEnd {
                signature: Some("sig-xyz".into()),
            },
            StreamChunk::ToolCallStart {
                id: "toolu_1".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallEnd {
                id: "toolu_1".into(),
                name: "get_weather".into(),
                args: json!({"location": "SF"}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
            },
        ];
        // Turn 2 just ends the run; we only care about the assistant
        // turn that issued the tool call.
        let turn2 = vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
        }];

        let model = ScriptedModel::new([turn1, turn2]);

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(GetWeather)).unwrap();

        let stream = run_chat(
            &model,
            vec![Message::user("what's the weather?")],
            &registry,
            RunConfig::default(),
        )
        .await
        .expect("run_chat should start");

        let chunks: Vec<_> = stream.collect().await;

        let new_messages = chunks
            .into_iter()
            .find_map(|c| match c {
                Ok(StreamChunk::RunFinished { new_messages, .. }) => Some(new_messages),
                _ => None,
            })
            .expect("run should emit RunFinished");

        let assistant_blocks = new_messages
            .iter()
            .find_map(|m| match m {
                Message::Assistant { blocks } => Some(blocks),
                _ => None,
            })
            .expect("new_messages should contain the assistant turn");

        assert_eq!(
            assistant_blocks.len(),
            2,
            "expected exactly Reasoning + ToolCall, got {assistant_blocks:?}"
        );
        match &assistant_blocks[0] {
            AssistantBlock::Reasoning { text, signature } => {
                assert_eq!(text, "thinking step.");
                assert_eq!(signature.as_deref(), Some("sig-xyz"));
            }
            other => panic!("expected Reasoning first, got {other:?}"),
        }
        match &assistant_blocks[1] {
            AssistantBlock::ToolCall { id, name, args } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(args, &json!({"location": "SF"}));
            }
            other => panic!("expected ToolCall second, got {other:?}"),
        }
    }

    /// All engine-emitted chunks of a run share the same `RunId`. Within
    /// a step, `StepId` matches across `StepStarted`, `ToolResult`, and
    /// `StepFinished`. Distinct iterations get distinct `StepId`s. This
    /// is the contract observability middlewares rely on to correlate
    /// concurrent runs and step-level spans.
    #[tokio::test]
    async fn engine_chunks_share_run_id_and_step_ids_match_per_iteration() {
        let turn1 = vec![
            StreamChunk::ToolCallStart {
                id: "toolu_1".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallEnd {
                id: "toolu_1".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
            },
        ];
        let turn2 = vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
        }];

        let model = ScriptedModel::new([turn1, turn2]);

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(GetWeather)).unwrap();

        let stream = run_chat(
            &model,
            vec![Message::user("hi")],
            &registry,
            RunConfig::default(),
        )
        .await
        .expect("run_chat should start");

        let chunks: Vec<StreamChunk> = stream.collect::<Vec<_>>().await.into_iter().map(|c| c.unwrap()).collect();

        let run_ids: Vec<&RunId> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::RunStarted { run_id }
                | StreamChunk::StepStarted { run_id, .. }
                | StreamChunk::StepFinished { run_id, .. }
                | StreamChunk::ToolResult { run_id, .. }
                | StreamChunk::RunFinished { run_id, .. } => Some(run_id),
                _ => None,
            })
            .collect();
        assert!(!run_ids.is_empty(), "expected engine chunks with RunId");
        let first = run_ids[0];
        for id in &run_ids[1..] {
            assert_eq!(*id, first, "all engine-emitted chunks must share RunId");
        }

        let mut step_ids_by_iter: std::collections::HashMap<usize, Vec<&StepId>> =
            std::collections::HashMap::new();
        for c in &chunks {
            match c {
                StreamChunk::StepStarted { step_id, iteration, .. }
                | StreamChunk::StepFinished { step_id, iteration, .. } => {
                    step_ids_by_iter.entry(*iteration).or_default().push(step_id);
                }
                StreamChunk::ToolResult { step_id, .. } => {
                    step_ids_by_iter.entry(0).or_default().push(step_id);
                }
                _ => {}
            }
        }
        for (iter, ids) in &step_ids_by_iter {
            let s = ids[0];
            for id in &ids[1..] {
                assert_eq!(*id, s, "iteration {iter} step_ids must match");
            }
        }

        let iter_step_ids: Vec<&StepId> = chunks
            .iter()
            .filter_map(|c| match c {
                StreamChunk::StepStarted { step_id, .. } => Some(step_id),
                _ => None,
            })
            .collect();
        if iter_step_ids.len() >= 2 {
            assert_ne!(
                iter_step_ids[0], iter_step_ids[1],
                "distinct iterations must mint distinct StepIds"
            );
        }
    }
}
