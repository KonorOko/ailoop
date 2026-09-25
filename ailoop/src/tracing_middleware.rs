//! `tracing` integration for ailoop. Compiled only when the `tracing`
//! feature is enabled.
//!
//! `TracingMiddleware` is a `ChatMiddleware` that emits `tracing` events
//! at every relevant lifecycle point with `RunId` / `StepId` attached so
//! a subscriber can correlate events from concurrent runs. Spans that
//! span an `await` boundary are deliberately avoided — the trait does
//! not expose per-run state, and faking it via `Arc<Mutex<HashMap<...>>>`
//! is more complexity than it earns. Callers can still build their own
//! span topology around `Conversation::stream` if they need a single
//! span for the whole run.

use ailoop_core::{
    ChatMiddleware, ChatRequest, HookAction, RunErrorInfo, RunFinishedInfo, RunStartInfo, StepInfo,
    StreamChunk, ToolCallInfo, ToolDecision, ToolResultContent,
};
use serde_json::Value;

/// Drop-in middleware that logs every run/step/tool/chunk event through
/// the `tracing` crate. Register a subscriber (e.g.
/// `tracing_subscriber::fmt`) to consume the output.
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct TracingMiddleware;

impl TracingMiddleware {
    /// Construct a new middleware. The struct is unit-shaped; this is
    /// equivalent to [`TracingMiddleware::default`] and either spelling
    /// works at registration sites.
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for TracingMiddleware {
    async fn on_run_started(&self, run: &RunStartInfo<'_>) -> HookAction {
        tracing::info!(
            target: "ailoop.run",
            run_id = %run.run_id,
            messages = run.messages.len(),
            "run started",
        );
        HookAction::Continue
    }

    async fn on_chat_request(&self, step: &StepInfo, req: &mut ChatRequest) {
        tracing::debug!(
            target: "ailoop.step",
            run_id = %step.run_id,
            step_id = %step.step_id,
            tools = req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            messages = req.messages.len(),
            "sending chat request",
        );
    }

    async fn on_chunk(&self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::TextDelta { delta } => {
                tracing::trace!(target: "ailoop.chunk", chars = delta.len(), "text delta");
            }
            StreamChunk::ReasoningDelta { delta } => {
                tracing::trace!(target: "ailoop.chunk", chars = delta.len(), "reasoning delta");
            }
            StreamChunk::ReasoningFinished { .. } => {
                tracing::trace!(target: "ailoop.chunk", "reasoning finished");
            }
            StreamChunk::RedactedReasoningBlock { .. } => {
                tracing::debug!(target: "ailoop.chunk", "redacted reasoning block");
            }
            StreamChunk::ToolCallStarted { id, name } => {
                tracing::info!(
                    target: "ailoop.chunk",
                    call_id = %id,
                    name = %name,
                    "tool call started",
                );
            }
            StreamChunk::ToolCallArgsDelta { .. } => {}
            StreamChunk::ToolCallFinished { id, name, .. } => {
                tracing::debug!(
                    target: "ailoop.chunk",
                    call_id = %id,
                    name = %name,
                    "tool call finished",
                );
            }
            StreamChunk::ToolCallMalformed {
                id,
                name,
                raw,
                error,
            } => {
                tracing::warn!(
                    target: "ailoop.chunk",
                    call_id = %id,
                    name = %name,
                    error = %error,
                    raw_bytes = raw.len(),
                    "tool call malformed",
                );
            }
            StreamChunk::TurnFinished {
                reason,
                usage,
                service_tier,
            } => {
                tracing::debug!(
                    target: "ailoop.chunk",
                    reason = ?reason,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    cached_input_tokens = usage.cached_input_tokens,
                    cache_creation_input_tokens = usage.cache_creation_input_tokens,
                    cache_creation_5m_tokens = usage.cache_creation_5m_tokens,
                    cache_creation_1h_tokens = usage.cache_creation_1h_tokens,
                    service_tier = service_tier.as_deref().unwrap_or(""),
                    "turn finished",
                );
            }
            StreamChunk::HistoryCompacted {
                run_id,
                before_count,
                after_count,
                strategy,
            } => {
                tracing::info!(
                    target: "ailoop.compaction",
                    run_id = %run_id,
                    before = *before_count,
                    after = *after_count,
                    strategy = *strategy,
                    "history compacted",
                );
            }
            StreamChunk::StepFinished {
                run_id,
                step_id,
                iteration,
                ..
            } => {
                tracing::debug!(
                    target: "ailoop.step",
                    run_id = %run_id,
                    step_id = %step_id,
                    iteration = *iteration,
                    "step finished",
                );
            }
            // RunStarted/StepStarted/RunFinished/ToolResult are handled in
            // their dedicated hooks where we have richer context. Avoid
            // double-logging them through `on_chunk`.
            _ => {}
        }
    }

    async fn on_run_finished(&self, run: &RunFinishedInfo<'_>) {
        tracing::info!(
            target: "ailoop.run",
            run_id = %run.run_id,
            reason = ?run.reason,
            input_tokens = run.usage.input_tokens,
            output_tokens = run.usage.output_tokens,
            new_messages = run.new_messages.len(),
            "run finished",
        );
    }

    async fn on_run_error(&self, run: &RunErrorInfo<'_>) {
        tracing::error!(
            target: "ailoop.run",
            run_id = %run.run_id,
            error = %run.error,
            input_tokens = run.usage.input_tokens,
            output_tokens = run.usage.output_tokens,
            partial_messages = run.partial_messages.len(),
            "run errored",
        );
    }

    async fn on_before_tool_call(&self, call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        tracing::info!(
            target: "ailoop.tool",
            run_id = %call.run_id,
            step_id = %call.step_id,
            call_id = %call.call_id,
            name = %call.name,
            "tool call starting",
        );
        ToolDecision::Continue
    }

    async fn on_after_tool_call(
        &self,
        call: &ToolCallInfo,
        _args: &Value,
        result: &ToolResultContent,
    ) {
        let outcome = if result.is_error { "error" } else { "text" };
        tracing::debug!(
            target: "ailoop.tool",
            run_id = %call.run_id,
            step_id = %call.step_id,
            call_id = %call.call_id,
            name = %call.name,
            outcome,
            "tool call finished",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{FinishReason, Message, RunConfig, RunId, StepId, Usage};
    use std::io;
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::fmt::MakeWriter;

    /// `MakeWriter` impl that appends every emitted byte into a shared
    /// buffer so the test can read what `tracing_subscriber::fmt` wrote.
    #[derive(Clone)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for BufferWriter {
        type Writer = BufferWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[tokio::test]
    async fn middleware_emits_run_id_in_tracing_output() {
        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer = BufferWriter(buffer.clone());

        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .finish();

        let mw = TracingMiddleware::new();
        let run_id = RunId::new();
        let step_id = StepId::new();

        tracing::subscriber::with_default(subscriber, || {
            futures::executor::block_on(async {
                mw.on_run_started(&RunStartInfo::new(&run_id, &[], &RunConfig::default()))
                    .await;
                mw.on_before_tool_call(
                    &ToolCallInfo::new(
                        run_id.clone(),
                        step_id.clone(),
                        "toolu_test",
                        "get_weather",
                    ),
                    &serde_json::Value::Null,
                )
                .await;
                mw.on_run_finished(&RunFinishedInfo::new(
                    &run_id,
                    &FinishReason::EndTurn,
                    &Usage::default(),
                    &[],
                ))
                .await;
                mw.on_chunk(&StreamChunk::HistoryCompacted {
                    run_id: run_id.clone(),
                    before_count: 12,
                    after_count: 4,
                    strategy: "truncate",
                })
                .await;
            });
        });

        let log = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        let run_id_str = run_id.to_string();
        assert!(
            log.contains(&run_id_str),
            "expected log to contain run_id `{run_id_str}`, got:\n{log}"
        );
        assert!(log.contains("run started"), "missing on_run_started event");
        assert!(
            log.contains("tool call starting"),
            "missing on_before_tool_call event"
        );
        assert!(
            log.contains("run finished"),
            "missing on_run_finished event"
        );
        assert!(
            log.contains("history compacted"),
            "missing HistoryCompacted event"
        );
        assert!(
            log.contains("strategy=\"truncate\""),
            "missing strategy field"
        );
    }

    /// Drives a real `run_chat` with `TracingMiddleware` registered and a
    /// `ScriptedModel` that completes in a single turn. Asserts the
    /// engine's `StepFinished` chunk reaches `on_chunk` (no dedicated
    /// hook covers it) and the middleware logs it.
    #[tokio::test]
    async fn run_chat_emits_step_finished_through_subscriber() {
        use crate::engine::run_chat;
        use ailoop_core::testing::ScriptedModel;
        use ailoop_tools::ToolRegistry;
        use futures::StreamExt;

        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer = BufferWriter(buffer.clone());

        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::DEBUG)
            .with_ansi(false)
            .finish();

        let model = ScriptedModel::new([vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        }]]);
        let registry = ToolRegistry::new();
        let mw: Arc<dyn ChatMiddleware> = Arc::new(TracingMiddleware::new());
        let run_id = RunId::new();
        let mut config = RunConfig::default();
        config.middlewares = vec![mw];
        config.run_id = Some(run_id.clone());

        tracing::subscriber::with_default(subscriber, || {
            futures::executor::block_on(async {
                let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
                    .await
                    .expect("run_chat should start");
                let _: Vec<_> = stream.collect().await;
            });
        });

        let log = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        assert!(
            log.contains("step finished"),
            "expected log to contain `step finished`, got:\n{log}"
        );
        let run_id_str = run_id.to_string();
        assert!(
            log.contains(&run_id_str),
            "expected step finished log to carry run_id `{run_id_str}`, got:\n{log}"
        );
    }
}
