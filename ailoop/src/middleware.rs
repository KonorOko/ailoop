use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ailoop_core::{ChatMiddleware, ChatRequest, RunId, StepId, ToolDecision};
use futures::future::BoxFuture;
use serde_json::Value;

use crate::{Prompt, PromptSection};

pub struct SystemPromptMiddleware {
    pub base: Prompt,
    pub tools_sections: HashMap<String, PromptSection>,
}

#[async_trait::async_trait]
impl ChatMiddleware for SystemPromptMiddleware {
    async fn on_chat_request(&self, _run_id: &RunId, _step_id: &StepId, req: &mut ChatRequest) {
        let mut prompt = self.base.clone();

        if let Some(tools) = &req.tools {
            for tool in tools {
                if let Some(tool_prompt) = self.tools_sections.get(&tool.name) {
                    prompt.add_section(tool_prompt.clone());
                }
            }
        }

        req.system_prompt = Some(prompt.render().into());
    }
}

pub(crate) type ApprovalCallback =
    Arc<dyn Fn(String, Value) -> BoxFuture<'static, ToolDecision> + Send + Sync>;

enum GatePolicy {
    All,
    ByName(HashSet<String>),
}

/// Middleware that asks a user-supplied async callback whether each tool
/// call should proceed, returning the callback's [`ToolDecision`] to the
/// engine.
///
/// Construct via [`approve_all`](Self::approve_all) for an unconditional
/// gate, or via [`for_named`](Self::for_named) for an explicit set of
/// tool names. For tag-based gating, use the builder method
/// `ConversationBuilder::with_approval`.
pub struct ApprovalMiddleware {
    callback: ApprovalCallback,
    policy: GatePolicy,
}

impl ApprovalMiddleware {
    /// Wire the callback for every tool call, regardless of tags.
    pub fn approve_all<F, Fut>(callback: F) -> Self
    where
        F: Fn(String, Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ToolDecision> + Send + 'static,
    {
        Self {
            callback: wrap_callback(callback),
            policy: GatePolicy::All,
        }
    }

    /// Wire the callback for tool calls whose name appears in `names`.
    /// Other tool calls pass through with `Continue`.
    pub fn for_named<I, S, F, Fut>(names: I, callback: F) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        F: Fn(String, Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ToolDecision> + Send + 'static,
    {
        Self {
            callback: wrap_callback(callback),
            policy: GatePolicy::ByName(names.into_iter().map(Into::into).collect()),
        }
    }

    pub(crate) fn from_parts(callback: ApprovalCallback, names: HashSet<String>) -> Self {
        Self {
            callback,
            policy: GatePolicy::ByName(names),
        }
    }

    pub(crate) fn from_parts_all(callback: ApprovalCallback) -> Self {
        Self {
            callback,
            policy: GatePolicy::All,
        }
    }

    fn should_gate(&self, name: &str) -> bool {
        match &self.policy {
            GatePolicy::All => true,
            GatePolicy::ByName(set) => set.contains(name),
        }
    }
}

pub(crate) fn wrap_callback<F, Fut>(callback: F) -> ApprovalCallback
where
    F: Fn(String, Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ToolDecision> + Send + 'static,
{
    Arc::new(move |name, args| Box::pin(callback(name, args)) as BoxFuture<'static, ToolDecision>)
}

#[async_trait::async_trait]
impl ChatMiddleware for ApprovalMiddleware {
    async fn on_before_tool_call(
        &self,
        _run_id: &RunId,
        _step_id: &StepId,
        name: &str,
        args: &Value,
    ) -> ToolDecision {
        if !self.should_gate(name) {
            return ToolDecision::Continue;
        }
        (self.callback)(name.to_string(), args.clone()).await
    }
}
