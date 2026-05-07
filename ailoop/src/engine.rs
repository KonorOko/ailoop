use std::sync::Arc;

use crate::errors::EngineError;
use ailoop_core::{
    AssistantBlock, ChatMiddleware, ChatRequest, CompletionModel, FinishReason, HookAction,
    RunConfig, StreamChunk, ToolDecision, Usage, UserBlock,
};
use ailoop_tools::{ToolRegistry, errors::ToolRegistryError};
pub use async_stream::try_stream;

pub use ailoop_core::{Message, ToolResultContent};
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;

macro_rules! bail_with_hooks {
    ($result: expr, $chain: expr) => {
        match $result {
            Ok(v) => Ok(v),
            Err(e) => {
                let err: EngineError<_> = e.into();
                for mw in $chain {
                    mw.on_run_error(&err).await;
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
    let stream = try_stream! {
        for mw in &config.middlewares {
            match mw.on_run_start(&messages, &config).await {
                HookAction::Continue => {},
                HookAction::Terminate {reason} => {
                    yield StreamChunk::RunFinished { reason: FinishReason::Aborted(reason), usage: Usage::default(), new_messages: vec![] };
                    return;
                }
            };
        }

        yield StreamChunk::RunStarted;

        let mut iteration = 0;
        let mut current_messages = messages.to_vec();
        let mut finish_reason = FinishReason::EndTurn;
        let mut usage_run = Usage::default();


        loop {
            if iteration >= config.max_iterations {
                bail_with_hooks!(Err(EngineError::MaxIterationsExceeded(iteration)), &config.middlewares)?;
            }

            yield StreamChunk::StepStarted { iteration };

            let mut assistant_blocks = Vec::new();
            let mut text_buf = String::new();


            let mut tool_calls = Vec::new();

            let mut req = ChatRequest {
                messages: current_messages.clone(),
                tools: Some(tools.active_tools().map(|t| t.tool_definition()).collect()),
                system_prompt: config.system_prompt.clone(),
                max_tokens: config.max_tokens,
                aditional_params: None,
                temperature: None
            };

            for mw in &config.middlewares {
                mw.on_chat_request(&mut req).await;
            }

            let mut adapter_stream = bail_with_hooks!(model.chat_stream(req).await.map_err(EngineError::Model), &config.middlewares)?;

            while let Some(chunk) = adapter_stream.next().await {
                let chunk = bail_with_hooks!(chunk.map_err(EngineError::Model), &config.middlewares)?;

                for mw in &config.middlewares {
                    mw.on_chunk(&chunk).await;
                }

                match &chunk {
                    StreamChunk::TextDelta { delta } => {
                        text_buf.push_str(delta);
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

                let decision = run_tool_chain(&config.middlewares, &name, &args).await;

                let content = match decision {
                    ToolDecision::Continue => {
                        match tools.tool_call(&name, args.clone()).await {
                            Ok(content) => content,
                            Err(ToolRegistryError::NotFound(_)) => {
                                let available_tools: Vec<String> = tools.active_tools().map(|t| t.tool_definition().name).collect();
                                ToolResultContent::Error(format!("Tool '{name}' not found. Available tools: [{}]", available_tools.join(", ")))},
                            Err(other) => bail_with_hooks!(Err(EngineError::Tool(other)), &config.middlewares)?,
                        }
                    },
                    ToolDecision::Skip {reason} => {
                        ToolResultContent::Error(format!("Tool skipped: {reason}"))
                    },
                    ToolDecision::Terminate {reason} => {
                        yield StreamChunk::RunFinished { reason: FinishReason::Aborted(reason), usage: usage_run, new_messages: current_messages.split_off(messages.len()) };
                        return;
                    }
                };

                for mw in &config.middlewares {
                    mw.on_after_tool_call(&name, &args, &content).await;
                }

                yield StreamChunk::ToolResult {
                    call_id: id.clone(),
                    content: content.clone()
                };

                tools_result.push(UserBlock::ToolResult { call_id: id, content });
            }

            if !tools_result.is_empty() {
                current_messages.push(Message::User { blocks: tools_result });
            }

            yield StreamChunk::StepFinished {
                iteration,
                new_messages_so_far: Arc::new(current_messages[messages.len()..].to_vec()),
            };

            if !matches!(finish_reason, FinishReason::ToolUse) {
                break;
            }

            iteration += 1;
        }

        let new_messages = current_messages.split_off(messages.len());

        for mw in &config.middlewares {
            mw.on_run_finished(&finish_reason, &usage_run, &new_messages).await;
        }

        yield StreamChunk::RunFinished {
            reason: finish_reason,
            usage: usage_run,
            new_messages,
        };
    };

    Ok(Box::pin(stream))
}

async fn run_tool_chain(
    chain: &[Arc<dyn ChatMiddleware>],
    name: &str,
    args: &Value,
) -> ToolDecision {
    for mw in chain {
        match mw.on_before_tool_call(name, args).await {
            ToolDecision::Continue => continue,
            terminate_or_skip => return terminate_or_skip,
        }
    }
    ToolDecision::Continue
}
