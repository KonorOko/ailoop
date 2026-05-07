use ailoop_context::ContextManager;
use ailoop_core::{ChatMiddleware, CompletionModel, Message, RunConfig, StreamChunk};
use ailoop_prompts::{Prompt, PromptSection};
use ailoop_tools::{ToolRegistry, registry::ToolDyn};
use futures::{Stream, StreamExt, stream::BoxStream};
use std::{collections::HashMap, path::Path, sync::Arc, task::Poll};

use crate::{
    errors::{BuildError, EngineError},
    middleware::SystemPromptMiddleware,
    run_chat,
};

pub struct Conversation<M: CompletionModel> {
    model: M,
    history: ContextManager,
    tools: ToolRegistry,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
}

impl<M: CompletionModel + Send + Sync> Conversation<M> {
    pub fn builder(model: M) -> ConversationBuilder<M> {
        ConversationBuilder::new(model)
    }

    pub async fn stream(
        &mut self,
        user_msg: impl Into<String>,
    ) -> Result<RunStream<'_, M>, EngineError<M::Error>> {
        self.history.add_message(Message::user(user_msg));
        self.history.compact_if_needed()?;

        let snapshot = self.history.messages().to_vec();

        let inner = run_chat(
            &self.model,
            snapshot,
            &self.tools,
            RunConfig {
                middlewares: self.middlewares.clone(),
                ..Default::default()
            },
        )
        .await?;

        Ok(RunStream {
            inner,
            history: &mut self.history,
        })
    }
}

pub struct ConversationBuilder<M: CompletionModel> {
    model: M,
    prompt: Prompt,
    history: ContextManager,
    tools: ToolRegistry,
    tool_prompts: HashMap<String, PromptSection>,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
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

        Ok(Conversation {
            model: self.model,
            history: self.history,
            tools: self.tools,
            middlewares: middlewares,
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
