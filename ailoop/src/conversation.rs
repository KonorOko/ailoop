use ailoop_context::ContextManager;
use ailoop_core::{
    AssistantBlock, ChatMiddleware, CompletionModel, FinishReason, Message, RunConfig, RunId,
    StreamChunk, ToolTag, Usage,
};
use ailoop_prompts::{Prompt, PromptSection};
use ailoop_tools::{ToolRegistry, registry::ToolDyn};
use futures::{Stream, StreamExt, stream::BoxStream};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
    task::Poll,
};

use crate::{
    errors::{BuildError, EngineError},
    middleware::{ApprovalCallback, ApprovalMiddleware, SystemPromptMiddleware, wrap_callback},
    run_chat,
};

pub struct Conversation<M: CompletionModel> {
    model: M,
    history: ContextManager,
    tools: ToolRegistry,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
}

/// Summary of a completed (or aborted) run, returned by
/// [`Conversation::run`].
///
/// `final_text` concatenates every [`AssistantBlock::Text`] of the last
/// assistant message in `new_messages` — what most callers think of as
/// "the answer". It is `None` when the run aborted before the assistant
/// produced any text, or when the last assistant message contains only
/// non-text blocks (`ToolCall`, `Reasoning`, `RedactedReasoning`).
#[derive(Debug)]
pub struct RunOutcome {
    pub run_id: RunId,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub new_messages: Vec<Message>,
    pub final_text: Option<String>,
}

impl<M: CompletionModel + Send + Sync> Conversation<M> {
    pub fn builder(model: M) -> ConversationBuilder<M> {
        ConversationBuilder::new(model)
    }

    /// Read-only view of the conversation history. Useful for asserting
    /// in tests and for callers who want to inspect or persist the
    /// conversation state outside of [`Conversation::run`] /
    /// [`Conversation::stream`].
    pub fn history(&self) -> &[Message] {
        self.history.messages()
    }

    /// Names of every tool currently active for this conversation.
    ///
    /// Useful for asserting in tests and for surfacing the effective tool
    /// set after `with_capabilities` filtering.
    pub fn active_tool_names(&self) -> Vec<String> {
        self.tools
            .active_tools()
            .map(|t| t.tool_definition().name)
            .collect()
    }

    /// Non-streaming convenience for one-question / one-answer flows
    /// (CLI tools, notebooks, batch evaluation). Drains the underlying
    /// [`Conversation::stream`] and materialises a [`RunOutcome`]
    /// summarising the run.
    ///
    /// Errors from the model, tools, or context management surface as
    /// `Err(EngineError)`, exactly as they would on the streaming path.
    /// Aborted runs (timeout, cancellation, hook/tool `Terminate`) are
    /// **not** errors — they return `Ok(RunOutcome)` with
    /// `finish_reason = FinishReason::Aborted(_)`. The caller decides
    /// whether to treat that as success or failure.
    ///
    /// History is extended with the run's `new_messages` exactly once,
    /// the same way it is on the streaming path.
    pub async fn run(
        &mut self,
        user_input: impl Into<String>,
    ) -> Result<RunOutcome, EngineError<M::Error>> {
        let mut stream = self.stream(user_input).await?;

        let mut finished: Option<(RunId, FinishReason, Usage, Vec<Message>)> = None;
        while let Some(chunk) = stream.next().await {
            if let StreamChunk::RunFinished {
                run_id,
                reason,
                usage,
                new_messages,
            } = chunk?
            {
                finished = Some((run_id, reason, usage, new_messages));
            }
        }

        let (run_id, finish_reason, usage, new_messages) = finished
            .expect("engine guarantees a RunFinished chunk before the stream terminates");

        let final_text = new_messages
            .iter()
            .rev()
            .find_map(|m| match m {
                Message::Assistant { blocks } => Some(blocks),
                _ => None,
            })
            .and_then(|blocks| {
                let mut buf = String::new();
                for b in blocks {
                    if let AssistantBlock::Text(t) = b {
                        buf.push_str(t);
                    }
                }
                if buf.is_empty() { None } else { Some(buf) }
            });

        Ok(RunOutcome {
            run_id,
            finish_reason,
            usage,
            new_messages,
            final_text,
        })
    }

    pub async fn stream(
        &mut self,
        user_msg: impl Into<String>,
    ) -> Result<RunStream<'_, M>, EngineError<M::Error>> {
        self.history.add_message(Message::user(user_msg));
        let report = self.history.compact_if_needed()?;

        let snapshot = self.history.messages().to_vec();
        let run_id = RunId::new();

        let inner = run_chat(
            &self.model,
            snapshot,
            &self.tools,
            RunConfig {
                middlewares: self.middlewares.clone(),
                run_id: Some(run_id.clone()),
                ..Default::default()
            },
        )
        .await?;

        let prelude: BoxStream<'_, Result<StreamChunk, EngineError<M::Error>>> = match report {
            Some(r) => {
                let chunk = StreamChunk::HistoryCompacted {
                    run_id,
                    before_count: r.before,
                    after_count: r.after,
                    strategy: r.strategy,
                };
                let middlewares = self.middlewares.clone();
                Box::pin(futures::stream::once(async move {
                    for mw in &middlewares {
                        mw.on_chunk(&chunk).await;
                    }
                    Ok(chunk)
                }))
            }
            None => Box::pin(futures::stream::empty()),
        };

        Ok(RunStream {
            inner: Box::pin(prelude.chain(inner)),
            history: &mut self.history,
        })
    }
}

struct ApprovalSpec {
    callback: ApprovalCallback,
    tags: Option<Vec<ToolTag>>,
}

pub struct ConversationBuilder<M: CompletionModel> {
    model: M,
    prompt: Prompt,
    history: ContextManager,
    tools: ToolRegistry,
    tool_prompts: HashMap<String, PromptSection>,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
    capabilities: Option<Vec<ToolTag>>,
    approval: Option<ApprovalSpec>,
    errors: Vec<BuildError>,
}

impl<M: CompletionModel> ConversationBuilder<M> {
    pub fn new(model: M) -> Self {
        let history = ContextManager::builder(460).build();
        let tools = ToolRegistry::new();
        let tool_prompts = HashMap::new();
        let prompt = Prompt::new();

        Self {
            model,
            prompt,
            tool_prompts,
            history,
            tools,
            middlewares: vec![],
            capabilities: None,
            approval: None,
            errors: vec![],
        }
    }

    pub fn tool<T>(mut self, tool: T) -> Self
    where
        T: ToolDyn + 'static,
    {
        if let Err(e) = self.tools.register(Arc::new(tool)) {
            self.errors.push(e.into());
        }
        self
    }

    pub fn tool_with_prompt_file<T>(mut self, tool: T, prompt_path: impl AsRef<Path>) -> Self
    where
        T: ToolDyn + 'static,
    {
        let arc_tool = Arc::new(tool);
        let name = arc_tool.name();

        match PromptSection::from_file(prompt_path) {
            Ok(section) => {
                self.tool_prompts.insert(name, section);
            }

            Err(e) => self.errors.push(e.into()),
        }

        if let Err(e) = self.tools.register(arc_tool) {
            self.errors.push(e.into());
        }

        self
    }

    pub fn middleware(mut self, middleware: Arc<dyn ChatMiddleware>) -> Self {
        self.middlewares.push(middleware);
        self
    }

    /// Restrict the active tool set to those whose declared tags overlap
    /// with `capabilities`. Applied at `build()` time, so call order
    /// relative to `tool(...)` does not matter.
    ///
    /// **Default-deny.** Tools with no declared tags are excluded under
    /// any non-empty `capabilities` filter — capability mode treats
    /// unknown tools as unknown risk. If you call this with an empty
    /// slice the result is *no* active tools.
    ///
    /// Successive calls overwrite the previous filter.
    pub fn with_capabilities(mut self, capabilities: &[ToolTag]) -> Self {
        self.capabilities = Some(capabilities.to_vec());
        self
    }

    /// Run `callback` before every Destructive or WritesFiles tool call;
    /// the callback's [`ToolDecision`] is forwarded to the engine.
    ///
    /// Tool name resolution happens at `build()` time and respects the
    /// capability filter, so untagged or filtered-out tools never trigger
    /// the callback.
    ///
    /// [`ToolDecision`]: ailoop_core::ToolDecision
    pub fn with_approval<F, Fut>(self, callback: F) -> Self
    where
        F: Fn(String, serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ailoop_core::ToolDecision> + Send + 'static,
    {
        self.with_approval_for_tags(&[ToolTag::Destructive, ToolTag::WritesFiles], callback)
    }

    /// Same as [`with_approval`](Self::with_approval) but with a custom
    /// tag set. The callback fires for tool calls whose declared tags
    /// overlap with `tags`. Pass an empty slice to disable the gate.
    pub fn with_approval_for_tags<F, Fut>(mut self, tags: &[ToolTag], callback: F) -> Self
    where
        F: Fn(String, serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ailoop_core::ToolDecision> + Send + 'static,
    {
        self.approval = Some(ApprovalSpec {
            callback: wrap_callback(callback),
            tags: Some(tags.to_vec()),
        });
        self
    }

    /// Run `callback` before every tool call (untagged or otherwise).
    /// Useful for `Conversation::run` flows where every action is
    /// surfaced to the human.
    pub fn with_approval_for_all<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(String, serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ailoop_core::ToolDecision> + Send + 'static,
    {
        self.approval = Some(ApprovalSpec {
            callback: wrap_callback(callback),
            tags: None,
        });
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt.add_section(PromptSection::new(prompt));
        self
    }

    pub fn system_prompt_file(mut self, path: impl AsRef<Path>) -> Self {
        match PromptSection::from_file(path) {
            Ok(section) => self.prompt.add_section(section),
            Err(e) => self.errors.push(e.into()),
        }

        self
    }

    pub fn build(self) -> Result<Conversation<M>, BuildError> {
        if let Some(first) = self.errors.into_iter().next() {
            return Err(first);
        }

        let mut middlewares = self.middlewares;

        let sp_mw = SystemPromptMiddleware {
            base: self.prompt,
            tools_sections: self.tool_prompts,
        };

        middlewares.push(Arc::new(sp_mw));

        let mut tools = self.tools;
        if let Some(capabilities) = self.capabilities {
            tools.deactivate_all();
            tools.activate_by_tags(&capabilities);
        }

        if let Some(spec) = self.approval {
            let approval_mw = match spec.tags {
                None => ApprovalMiddleware::from_parts_all(spec.callback),
                Some(tags) => {
                    let names: HashSet<String> = tools
                        .active_tools()
                        .filter(|tool| {
                            tool.tool_definition()
                                .tags
                                .iter()
                                .any(|tag| tags.contains(tag))
                        })
                        .map(|tool| tool.tool_definition().name)
                        .collect();
                    ApprovalMiddleware::from_parts(spec.callback, names)
                }
            };
            middlewares.push(Arc::new(approval_mw));
        }

        Ok(Conversation {
            model: self.model,
            history: self.history,
            tools,
            middlewares,
        })
    }
}

pub struct RunStream<'a, M: CompletionModel> {
    inner: BoxStream<'a, Result<StreamChunk, EngineError<M::Error>>>,
    history: &'a mut ContextManager,
}

impl<'a, M: CompletionModel> Stream for RunStream<'a, M> {
    type Item = Result<StreamChunk, EngineError<M::Error>>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();

        let polled = this.inner.poll_next_unpin(cx);

        if let Poll::Ready(Some(Ok(StreamChunk::RunFinished { new_messages, .. }))) = &polled {
            this.history.extend(new_messages.clone());
        }

        polled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{ChatRequest, CompletionModel, StreamChunk, ToolDecision};
    use ailoop_tools::registry::ToolDyn as _;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockModel;

    #[async_trait::async_trait]
    impl CompletionModel for MockModel {
        type Error = std::convert::Infallible;
        fn name(&self) -> &str {
            "mock"
        }
        fn model(&self) -> &str {
            "mock"
        }
        async fn chat_stream(
            &self,
            _req: ChatRequest,
        ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[derive(Default)]
    struct FakeTool {
        name: &'static str,
        tags: Vec<ToolTag>,
    }

    #[async_trait::async_trait]
    impl ToolDyn for FakeTool {
        fn name(&self) -> String {
            self.name.into()
        }
        fn tool_definition(&self) -> ailoop_core::ToolDefinition {
            ailoop_core::ToolDefinition {
                name: self.name.into(),
                description: "fake".into(),
                input_schema: json!({"type":"object","properties":{},"required":[]}),
                tags: self.tags.clone(),
            }
        }
        async fn call(&self, _args: serde_json::Value) -> ailoop_core::ToolResultContent {
            ailoop_core::ToolResultContent::Text(String::new())
        }
    }

    async fn dispatch_through_chain(
        chat: &Conversation<MockModel>,
        name: &str,
        args: &serde_json::Value,
    ) -> ToolDecision {
        let run_id = ailoop_core::RunId::new();
        let step_id = ailoop_core::StepId::new();
        for mw in &chat.middlewares {
            match mw
                .on_before_tool_call(&run_id, &step_id, name, args)
                .await
            {
                ToolDecision::Continue => continue,
                other => return other,
            }
        }
        ToolDecision::Continue
    }

    #[tokio::test]
    async fn with_approval_gates_only_destructive_writesfiles_tools() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_cb = counter.clone();
        let chat = Conversation::builder(MockModel)
            .tool(FakeTool {
                name: "list_dir",
                tags: vec![ToolTag::ReadOnly],
            })
            .tool(FakeTool {
                name: "delete_file",
                tags: vec![ToolTag::Destructive, ToolTag::WritesFiles],
            })
            .tool(FakeTool {
                name: "untagged",
                tags: vec![],
            })
            .with_approval(move |_name, _args| {
                let c = counter_cb.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    ToolDecision::Continue
                }
            })
            .build()
            .unwrap();

        dispatch_through_chain(&chat, "list_dir", &json!({})).await;
        dispatch_through_chain(&chat, "delete_file", &json!({})).await;
        dispatch_through_chain(&chat, "untagged", &json!({})).await;

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn with_approval_for_tags_uses_custom_set() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_cb = counter.clone();
        let chat = Conversation::builder(MockModel)
            .tool(FakeTool {
                name: "list_dir",
                tags: vec![ToolTag::ReadOnly],
            })
            .tool(FakeTool {
                name: "delete_file",
                tags: vec![ToolTag::Destructive],
            })
            .with_approval_for_tags(&[ToolTag::ReadOnly], move |_name, _args| {
                let c = counter_cb.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    ToolDecision::Continue
                }
            })
            .build()
            .unwrap();

        dispatch_through_chain(&chat, "list_dir", &json!({})).await;
        dispatch_through_chain(&chat, "delete_file", &json!({})).await;

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn with_approval_for_all_fires_for_untagged() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_cb = counter.clone();
        let chat = Conversation::builder(MockModel)
            .tool(FakeTool {
                name: "untagged",
                tags: vec![],
            })
            .with_approval_for_all(move |_name, _args| {
                let c = counter_cb.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    ToolDecision::Continue
                }
            })
            .build()
            .unwrap();

        dispatch_through_chain(&chat, "untagged", &json!({})).await;

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn approval_skip_decision_short_circuits_chain() {
        let chat = Conversation::builder(MockModel)
            .tool(FakeTool {
                name: "delete_file",
                tags: vec![ToolTag::Destructive],
            })
            .with_approval(|_name, _args| async move {
                ToolDecision::Skip {
                    reason: "user denied".into(),
                }
            })
            .build()
            .unwrap();

        let decision = dispatch_through_chain(&chat, "delete_file", &json!({})).await;
        match decision {
            ToolDecision::Skip { reason } => assert_eq!(reason, "user denied"),
            _ => panic!("expected Skip"),
        }
    }

    #[tokio::test]
    async fn capabilities_filter_runs_before_approval_resolution() {
        // delete_file is filtered out by capabilities; the approval gate
        // should not register it as a gated name (and the dispatch
        // wouldn't reach it anyway because the engine wouldn't expose it).
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_cb = counter.clone();
        let chat = Conversation::builder(MockModel)
            .tool(FakeTool {
                name: "list_dir",
                tags: vec![ToolTag::ReadOnly],
            })
            .tool(FakeTool {
                name: "delete_file",
                tags: vec![ToolTag::Destructive],
            })
            .with_capabilities(&[ToolTag::ReadOnly])
            .with_approval(move |_name, _args| {
                let c = counter_cb.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    ToolDecision::Continue
                }
            })
            .build()
            .unwrap();

        // delete_file is inactive (filtered out), so even if dispatched
        // the approval gate wouldn't have it in its set.
        dispatch_through_chain(&chat, "delete_file", &json!({})).await;
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "filtered-out tool should not be in approval gate"
        );
    }

    /// `Conversation::stream` runs registered middlewares' `on_chunk`
    /// against the `HistoryCompacted` prelude before yielding it, so a
    /// subscriber sees the event under a real run (not just when a test
    /// pokes `on_chunk` directly). Companion to the engine-side test
    /// that asserts `StepFinished` reaches `on_chunk`.
    #[cfg(feature = "tracing")]
    #[tokio::test]
    async fn stream_logs_history_compacted_through_subscriber() {
        use crate::TracingMiddleware;
        use ailoop_core::testing::ScriptedModel;
        use ailoop_core::{FinishReason, Usage};
        use std::io;
        use std::sync::Mutex;
        use tracing_subscriber::fmt::MakeWriter;

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

        let buffer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let writer = BufferWriter(buffer.clone());

        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .finish();

        let model = ScriptedModel::new([vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
        }]]);

        let mut chat = Conversation::builder(model)
            .middleware(Arc::new(TracingMiddleware::new()))
            .build()
            .expect("builder should succeed");

        // Pre-seed enough history to overflow `max_tokens=460` (CharEstimator
        // = len()/4) so `compact_if_needed` fires on `stream`.
        let big = "x".repeat(200);
        for _ in 0..15 {
            chat.history.add_message(Message::user(big.clone()));
            chat.history.add_message(Message::assistant_text(big.clone()));
        }

        tracing::subscriber::with_default(subscriber, || {
            futures::executor::block_on(async {
                let mut stream = chat.stream("trigger run").await.expect("stream should start");
                while (stream.next().await).is_some() {}
            });
        });

        let log = String::from_utf8(buffer.lock().unwrap().clone()).unwrap();
        assert!(
            log.contains("history compacted"),
            "expected log to contain `history compacted`, got:\n{log}"
        );
    }

    /// When `compact_if_needed` runs, `Conversation::stream` must emit
    /// `HistoryCompacted` as the first chunk, carrying the same `RunId`
    /// the engine then uses for its own chunks. This is the contract
    /// observability middlewares rely on to attribute compaction to a
    /// specific run.
    #[tokio::test]
    async fn stream_emits_history_compacted_with_matching_run_id() {
        let mut chat = Conversation::builder(MockModel)
            .build()
            .expect("builder should succeed");

        // Default `max_tokens` for the builder is 460 (CharEstimator =
        // len()/4). Stuff enough text to overshoot the budget so the
        // call to `stream` triggers compaction. Pre-seed messages
        // directly into the private history field; we want compaction
        // to fire on the call we observe, not on `add_message`.
        let big = "x".repeat(200);
        for _ in 0..15 {
            chat.history.add_message(Message::user(big.clone()));
            chat.history
                .add_message(Message::assistant_text(big.clone()));
        }

        let mut stream = chat.stream("trigger run").await.expect("stream should start");

        let first = stream.next().await.expect("expected at least one chunk").unwrap();
        let compacted_run_id = match first {
            StreamChunk::HistoryCompacted {
                run_id,
                before_count,
                after_count,
                strategy,
            } => {
                assert!(after_count < before_count, "compaction must shrink history");
                assert_eq!(strategy, "truncate");
                run_id
            }
            other => panic!("expected HistoryCompacted first, got {other:?}"),
        };

        // Every subsequent engine-emitted chunk must carry the same RunId.
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            match &chunk {
                StreamChunk::RunStarted { run_id }
                | StreamChunk::StepStarted { run_id, .. }
                | StreamChunk::StepFinished { run_id, .. }
                | StreamChunk::ToolResult { run_id, .. }
                | StreamChunk::RunFinished { run_id, .. } => {
                    assert_eq!(*run_id, compacted_run_id, "engine RunId must match HistoryCompacted RunId");
                }
                _ => {}
            }
        }
    }
}
