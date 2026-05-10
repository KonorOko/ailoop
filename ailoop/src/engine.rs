use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use crate::errors::EngineError;
use ailoop_core::{
    AssistantBlock, CancellationToken, ChatMiddleware, ChatRequest, CompletionModel, FinishReason,
    HookAction, Message, RunConfig, RunId, StepId, StreamChunk, ToolDecision, ToolResultContent,
    Usage, UserBlock,
};
use ailoop_tools::{ToolRegistry, errors::ToolRegistryError};
use async_stream::try_stream;
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;

type AbortFuture = Pin<Box<dyn Future<Output = String> + Send>>;

/// Builds the abort future that resolves with a textual reason when the
/// configured timeout elapses or the [`CancellationToken`] is fired.
/// Resolves to a never-completing future when neither is configured.
fn build_abort_future(
    timeout: Option<Duration>,
    cancellation: Option<CancellationToken>,
) -> AbortFuture {
    Box::pin(async move {
        let cancel_fut = async move {
            match cancellation {
                Some(token) => {
                    token.cancelled().await;
                    "cancelled by caller".to_string()
                }
                None => std::future::pending::<String>().await,
            }
        };
        let timer_fut = async move {
            match timeout {
                Some(d) => {
                    tokio::time::sleep(d).await;
                    format!("timeout exceeded after {d:?}")
                }
                None => std::future::pending::<String>().await,
            }
        };
        // Cancel takes priority on simultaneous fire so callers can rely
        // on the "cancelled by caller" reason in a configured race.
        tokio::select! {
            biased;
            reason = cancel_fut => reason,
            reason = timer_fut => reason,
        }
    })
}

/// Polls `fut` against the abort future. If the abort wins the race the
/// caller receives `Err(reason)` and `fut` is dropped — which cancels
/// any in-flight HTTP request, retry-backoff sleep, or tool execution
/// behind it.
async fn race_abort<F, T>(fut: F, abort: &mut AbortFuture) -> Result<T, String>
where
    F: Future<Output = T>,
{
    tokio::select! {
        biased;
        reason = &mut *abort => Err(reason),
        value = fut => Ok(value),
    }
}

/// Fires the `on_run_finished` + `on_chunk` hook pair for an aborted
/// run and returns the `RunFinished` chunk for the caller to yield.
/// Centralised so every abort site (hook terminate, tool terminate,
/// timeout, cancellation) follows the same persistence discipline.
async fn fire_abort_hooks(
    middlewares: &[Arc<dyn ChatMiddleware>],
    run_id: &RunId,
    reason: String,
    usage: Usage,
    new_messages: Vec<Message>,
) -> StreamChunk {
    let finish_reason = FinishReason::Aborted(reason);
    for mw in middlewares {
        mw.on_run_finished(run_id, &finish_reason, &usage, &new_messages)
            .await;
    }
    let mut chunk = StreamChunk::RunFinished {
        run_id: run_id.clone(),
        reason: finish_reason,
        usage,
        new_messages,
    };
    for mw in middlewares {
        mw.on_chunk_mut(&mut chunk).await;
    }
    for mw in middlewares {
        mw.on_chunk(&chunk).await;
    }
    chunk
}

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
        // The abort future resolves with a textual reason when either
        // the timeout elapses or the cancellation token fires; until
        // then it is `pending`, so wrapping any await with
        // `race_abort(_, &mut abort_fut)` is a no-op on the happy path.
        let mut abort_fut: AbortFuture = build_abort_future(
            config.timeout,
            config.cancellation.clone(),
        );

        for mw in &config.middlewares {
            let action = match race_abort(
                mw.on_run_start(&run_id, &messages, &config),
                &mut abort_fut,
            ).await {
                Ok(a) => a,
                Err(reason) => {
                    let chunk = fire_abort_hooks(
                        &config.middlewares, &run_id, reason, Usage::default(), vec![],
                    ).await;
                    yield chunk;
                    return;
                }
            };
            match action {
                HookAction::Continue => {},
                HookAction::Terminate {reason} => {
                    let chunk = fire_abort_hooks(
                        &config.middlewares, &run_id, reason, Usage::default(), vec![],
                    ).await;
                    yield chunk;
                    return;
                }
                _ => {}
            };
        }

        let mut chunk = StreamChunk::RunStarted { run_id: run_id.clone() };
        for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
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
            let mut chunk = StreamChunk::StepStarted { run_id: run_id.clone(), step_id: step_id.clone(), iteration };
            for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
            for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
            yield chunk;

            let mut assistant_blocks = Vec::new();
            let mut text_buf = String::new();
            let mut reasoning_buf = String::new();

            let mut tool_calls = Vec::new();

            let mut req = ChatRequest::new(current_messages.clone(), config.max_tokens);
            req.tools = Some(tools.active_tools().map(|t| t.tool_definition()).collect());
            req.system_prompt = config.system_prompt.clone();

            for mw in &config.middlewares {
                if let Err(reason) = race_abort(
                    mw.on_chat_request(&run_id, &step_id, &mut req),
                    &mut abort_fut,
                ).await {
                    let new_messages = current_messages.split_off(messages.len());
                    let chunk = fire_abort_hooks(
                        &config.middlewares, &run_id, reason, usage_run, new_messages,
                    ).await;
                    yield chunk;
                    return;
                }
            }

            let chat_stream_result = race_abort(model.chat_stream(req), &mut abort_fut).await;
            let mut adapter_stream = match chat_stream_result {
                Ok(r) => bail_with_hooks!(r.map_err(EngineError::Model), &config.middlewares, &run_id)?,
                Err(reason) => {
                    let new_messages = current_messages.split_off(messages.len());
                    let chunk = fire_abort_hooks(
                        &config.middlewares, &run_id, reason, usage_run, new_messages,
                    ).await;
                    yield chunk;
                    return;
                }
            };

            loop {
                let next = match race_abort(adapter_stream.next(), &mut abort_fut).await {
                    Ok(n) => n,
                    Err(reason) => {
                        // Preserve any complete blocks the assistant has
                        // produced before the abort so history stays
                        // consistent — only blocks closed by their `*End`
                        // chunk are in `assistant_blocks`, partial tool
                        // calls (start without end) are not.
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::text(text_buf));
                        }
                        if !assistant_blocks.is_empty() {
                            current_messages.push(Message::Assistant { blocks: assistant_blocks });
                        }
                        let new_messages = current_messages.split_off(messages.len());
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                };
                let chunk = match next {
                    Some(c) => c,
                    None => break,
                };
                let chunk = bail_with_hooks!(chunk.map_err(EngineError::Model), &config.middlewares, &run_id)?;

                // Mutating phase first: every `_mut` runs before any
                // observer, so the engine itself, the assistant-history
                // builder below, and the stream consumer all see the
                // same fully-mutated chunk. See the trait doc on
                // `on_chunk_mut` for the contract.
                let mut chunk = chunk;
                for mw in &config.middlewares {
                    mw.on_chunk_mut(&mut chunk).await;
                }
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
                            assistant_blocks.push(AssistantBlock::text(std::mem::take(&mut text_buf)));
                        }
                    },
                    StreamChunk::ToolCallEnd { id, name, args } => {
                        assistant_blocks.push(AssistantBlock::tool_call(id.clone(), name.clone(), args.clone()));
                        tool_calls.push((id.clone(), name.clone(), args.clone()))
                    },
                    StreamChunk::ReasoningEnd { signature } => {
                        // Reasoning blocks must keep their original position
                        // relative to text and tool_use; flush any pending
                        // text first so order on replay matches the wire.
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::text(std::mem::take(&mut text_buf)));
                        }
                        assistant_blocks.push(AssistantBlock::Reasoning {
                            text: std::mem::take(&mut reasoning_buf),
                            signature: signature.clone(),
                        });
                    },
                    StreamChunk::RedactedReasoningBlock { data } => {
                        if !text_buf.is_empty() {
                            assistant_blocks.push(AssistantBlock::text(std::mem::take(&mut text_buf)));
                        }
                        assistant_blocks.push(AssistantBlock::RedactedReasoning {
                            data: data.clone(),
                        });
                    },
                    StreamChunk::TurnFinished { reason, usage, .. } => {
                        finish_reason = reason.clone();
                        usage_run += *usage;
                        continue;
                    },
                    _=> ()
                }


                yield chunk;
            }

            if !text_buf.is_empty() {
                assistant_blocks.push(AssistantBlock::text(text_buf));
            }

            if !assistant_blocks.is_empty() {
                current_messages.push(Message::Assistant { blocks: assistant_blocks });
            }

            let mut tools_result = Vec::new();
            for (id, name, mut args) in tool_calls {

                // Input-transform phase: every `_mut` runs before any
                // gating decision so a sanitizer can rewrite args before
                // an `ApprovalMiddleware` sees them. Mutated `args` flow
                // through to the tool invocation below.
                let mut aborted = false;
                let mut abort_reason = String::new();
                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_before_tool_call_mut(&run_id, &step_id, &name, &mut args),
                        &mut abort_fut,
                    ).await {
                        aborted = true;
                        abort_reason = reason;
                        break;
                    }
                }
                if aborted {
                    if !tools_result.is_empty() {
                        current_messages.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                    }
                    let new_messages = current_messages.split_off(messages.len());
                    let chunk = fire_abort_hooks(
                        &config.middlewares, &run_id, abort_reason, usage_run, new_messages,
                    ).await;
                    yield chunk;
                    return;
                }

                let decision = match race_abort(
                    run_tool_chain(&config.middlewares, &run_id, &step_id, &name, &args),
                    &mut abort_fut,
                ).await {
                    Ok(d) => d,
                    Err(reason) => {
                        if !tools_result.is_empty() {
                            current_messages.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                        }
                        let new_messages = current_messages.split_off(messages.len());
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                };

                let mut content = match decision {
                    ToolDecision::Continue => {
                        let call_result = race_abort(
                            tools.tool_call(&name, args.clone()),
                            &mut abort_fut,
                        ).await;
                        match call_result {
                            Ok(Ok(content)) => content,
                            Ok(Err(ToolRegistryError::NotFound(_))) => {
                                let available_tools: Vec<String> = tools.active_tools().map(|t| t.tool_definition().name).collect();
                                ToolResultContent::Error(format!("Tool '{name}' not found. Available tools: [{}]", available_tools.join(", ")))
                            },
                            Ok(Err(other)) => bail_with_hooks!(Err(EngineError::Tool(other)), &config.middlewares, &run_id)?,
                            Err(reason) => {
                                if !tools_result.is_empty() {
                                    current_messages.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                                }
                                let new_messages = current_messages.split_off(messages.len());
                                let chunk = fire_abort_hooks(
                                    &config.middlewares, &run_id, reason, usage_run, new_messages,
                                ).await;
                                yield chunk;
                                return;
                            }
                        }
                    },
                    ToolDecision::Skip {reason} => {
                        ToolResultContent::Error(format!("Tool skipped: {reason}"))
                    },
                    ToolDecision::Terminate {reason} => {
                        if !tools_result.is_empty() {
                            current_messages.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                        }
                        let new_messages = current_messages.split_off(messages.len());
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                    _ => ToolResultContent::Error("unsupported ToolDecision variant".into()),
                };

                // Output-transform phase: every `_mut` runs before any
                // observer, so observers and the engine's emitted
                // `ToolResult` chunk all see the same mutated result.
                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_after_tool_call_mut(&run_id, &step_id, &name, &args, &mut content),
                        &mut abort_fut,
                    ).await {
                        // Same persistence discipline as the observer
                        // path below: the just-completed tool's result
                        // (whatever the partially-applied transforms
                        // left it as) must land in history so the next
                        // assistant turn isn't missing a tool_result.
                        tools_result.push(UserBlock::tool_result(id.clone(), content.clone()));
                        current_messages.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                        let new_messages = current_messages.split_off(messages.len());
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                }

                for mw in &config.middlewares {
                    if let Err(reason) = race_abort(
                        mw.on_after_tool_call(&run_id, &step_id, &name, &args, &content),
                        &mut abort_fut,
                    ).await {
                        // Preserve the just-completed tool's result so
                        // history isn't left with a tool_call missing
                        // its tool_result on the next assistant turn.
                        tools_result.push(UserBlock::tool_result(id.clone(), content.clone()));
                        current_messages.push(Message::User { blocks: std::mem::take(&mut tools_result) });
                        let new_messages = current_messages.split_off(messages.len());
                        let chunk = fire_abort_hooks(
                            &config.middlewares, &run_id, reason, usage_run, new_messages,
                        ).await;
                        yield chunk;
                        return;
                    }
                }

                let mut chunk = StreamChunk::ToolResult {
                    run_id: run_id.clone(),
                    step_id: step_id.clone(),
                    call_id: id.clone(),
                    content: content.clone(),
                };
                for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
                for mw in &config.middlewares { mw.on_chunk(&chunk).await; }
                yield chunk;

                tools_result.push(UserBlock::tool_result(id, content));
            }

            if !tools_result.is_empty() {
                current_messages.push(Message::User { blocks: tools_result });
            }

            let mut chunk = StreamChunk::StepFinished {
                run_id: run_id.clone(),
                step_id: step_id.clone(),
                iteration,
                new_messages_so_far: Arc::new(current_messages[messages.len()..].to_vec()),
            };
            for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
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

        let mut chunk = StreamChunk::RunFinished {
            run_id: run_id.clone(),
            reason: finish_reason,
            usage: usage_run,
            new_messages,
        };
        for mw in &config.middlewares { mw.on_chunk_mut(&mut chunk).await; }
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
            ToolDefinition::new(
                "get_weather",
                "stub",
                json!({"type":"object","properties":{},"required":[]}),
                vec![],
            )
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
                service_tier: None,
            },
        ];
        // Turn 2 just ends the run; we only care about the assistant
        // turn that issued the tool call.
        let turn2 = vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
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
            AssistantBlock::ToolCall { id, name, args, .. } => {
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
                service_tier: None,
            },
        ];
        let turn2 = vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
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

        let chunks: Vec<StreamChunk> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|c| c.unwrap())
            .collect();

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
                StreamChunk::StepStarted {
                    step_id, iteration, ..
                }
                | StreamChunk::StepFinished {
                    step_id, iteration, ..
                } => {
                    step_ids_by_iter
                        .entry(*iteration)
                        .or_default()
                        .push(step_id);
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

    /// `HookAction::Terminate` from `on_run_start` must still drive the planned
    /// termination contract: every middleware sees `on_run_finished` once with
    /// `FinishReason::Aborted`. Observers like `TokenBudget` accumulate the
    /// final turn there; if the engine emits `RunFinished` without firing the
    /// hook, those middlewares miss aborted runs entirely.
    #[tokio::test]
    async fn on_run_finished_fires_on_hook_terminate() {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct AbortingMw {
            finished_count: AtomicUsize,
            last_reason: Mutex<Option<FinishReason>>,
        }

        #[async_trait::async_trait]
        impl ChatMiddleware for AbortingMw {
            async fn on_run_start(
                &self,
                _run_id: &RunId,
                _messages: &[Message],
                _config: &RunConfig,
            ) -> HookAction {
                HookAction::Terminate {
                    reason: "budget exceeded".into(),
                }
            }
            async fn on_run_finished(
                &self,
                _run_id: &RunId,
                reason: &FinishReason,
                _usage: &Usage,
                _new_messages: &[Message],
            ) {
                self.finished_count.fetch_add(1, Ordering::SeqCst);
                *self.last_reason.lock().unwrap() = Some(reason.clone());
            }
        }

        let mw = Arc::new(AbortingMw {
            finished_count: AtomicUsize::new(0),
            last_reason: Mutex::new(None),
        });
        let model = ScriptedModel::new(Vec::<Vec<StreamChunk>>::new());
        let registry = ToolRegistry::new();
        let mut config = RunConfig::default();
        config.middlewares = vec![mw.clone()];

        let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
            .await
            .expect("run_chat should start");
        let chunks: Vec<_> = stream.collect().await;

        assert_eq!(
            mw.finished_count.load(Ordering::SeqCst),
            1,
            "on_run_finished must fire exactly once on HookAction::Terminate"
        );
        match mw.last_reason.lock().unwrap().as_ref() {
            Some(FinishReason::Aborted(r)) => assert_eq!(r, "budget exceeded"),
            other => panic!("expected Aborted reason, got {other:?}"),
        }

        let finished = chunks
            .into_iter()
            .find_map(|c| match c {
                Ok(StreamChunk::RunFinished { reason, .. }) => Some(reason),
                _ => None,
            })
            .expect("run should emit RunFinished");
        assert!(
            matches!(finished, FinishReason::Aborted(ref r) if r == "budget exceeded"),
            "RunFinished.reason mismatch: {finished:?}"
        );
    }

    /// When middleware aborts mid-step on the second of two tool calls, the
    /// `User { ToolResult }` message for the already-executed first tool must
    /// land in `RunFinished.new_messages`. Otherwise `Conversation::stream`
    /// extends history with an assistant message carrying tool_uses but no
    /// matching tool_results — the next provider call rejects with HTTP 400.
    #[tokio::test]
    async fn tool_terminate_preserves_prior_tool_results_in_history() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct TerminateOnSecondToolMw {
            calls: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl ChatMiddleware for TerminateOnSecondToolMw {
            async fn on_before_tool_call(
                &self,
                _run_id: &RunId,
                _step_id: &StepId,
                _name: &str,
                _args: &Value,
            ) -> ToolDecision {
                let n = self.calls.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    ToolDecision::Continue
                } else {
                    ToolDecision::Terminate {
                        reason: "policy".into(),
                    }
                }
            }
        }

        let turn = vec![
            StreamChunk::ToolCallStart {
                id: "toolu_a".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallEnd {
                id: "toolu_a".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::ToolCallStart {
                id: "toolu_b".into(),
                name: "get_weather".into(),
            },
            StreamChunk::ToolCallEnd {
                id: "toolu_b".into(),
                name: "get_weather".into(),
                args: json!({}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
                service_tier: None,
            },
        ];
        let model = ScriptedModel::new([turn]);

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(GetWeather)).unwrap();

        let mw = Arc::new(TerminateOnSecondToolMw {
            calls: AtomicUsize::new(0),
        });
        let mut config = RunConfig::default();
        config.middlewares = vec![mw.clone()];

        let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
            .await
            .expect("run_chat should start");
        let chunks: Vec<_> = stream.collect().await;

        let new_messages = chunks
            .into_iter()
            .find_map(|c| match c {
                Ok(StreamChunk::RunFinished {
                    reason: FinishReason::Aborted(_),
                    new_messages,
                    ..
                }) => Some(new_messages),
                _ => None,
            })
            .expect("run should emit RunFinished{Aborted}");

        let assistant_tool_call_ids: Vec<&str> = new_messages
            .iter()
            .filter_map(|m| match m {
                Message::Assistant { blocks } => Some(blocks),
                _ => None,
            })
            .flat_map(|blocks| blocks.iter())
            .filter_map(|b| match b {
                AssistantBlock::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            assistant_tool_call_ids,
            vec!["toolu_a", "toolu_b"],
            "assistant turn must carry both tool_calls"
        );

        let user_tool_result_ids: Vec<&str> = new_messages
            .iter()
            .filter_map(|m| match m {
                Message::User { blocks } => Some(blocks),
                _ => None,
            })
            .flat_map(|blocks| blocks.iter())
            .filter_map(|b| match b {
                UserBlock::ToolResult { call_id, .. } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            user_tool_result_ids.contains(&"toolu_a"),
            "first tool's ToolResult must be preserved in history, got {user_tool_result_ids:?}"
        );
    }
}
