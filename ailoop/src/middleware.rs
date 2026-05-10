use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ailoop_core::{ChatMiddleware, ChatRequest, RunId, StepId, ToolChoice, ToolDecision};
use futures::future::BoxFuture;
use serde_json::Value;

use crate::{Prompt, PromptSection};

pub(crate) struct SystemPromptMiddleware {
    pub(crate) base: Prompt,
    pub(crate) tools_sections: HashMap<String, PromptSection>,
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

pub(crate) type RequestOverlay = Arc<dyn Fn(&mut ChatRequest) + Send + Sync>;

/// Builder-supplied defaults for per-request controls, applied via
/// [`RequestDefaultsMiddleware`] at the head of the middleware chain.
#[derive(Default, Clone)]
pub(crate) struct RequestDefaults {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) stop_sequences: Vec<String>,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) disable_parallel_tool_use: Option<bool>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) additional_params: Option<Value>,
    pub(crate) overlay: Option<RequestOverlay>,
}

impl RequestDefaults {
    pub(crate) fn has_overrides(&self) -> bool {
        self.temperature.is_some()
            || self.top_p.is_some()
            || self.top_k.is_some()
            || !self.stop_sequences.is_empty()
            || self.tool_choice.is_some()
            || self.disable_parallel_tool_use.is_some()
            || self.max_tokens.is_some()
            || self.additional_params.is_some()
            || self.overlay.is_some()
    }
}

/// Internal middleware that applies the [`RequestDefaults`] captured by
/// `ConversationBuilder` to every outgoing [`ChatRequest`]. Inserted at
/// the head of the chain so user-supplied middlewares run *after* it
/// and can override unconditionally — the builder defaults are a floor,
/// not a ceiling.
pub(crate) struct RequestDefaultsMiddleware {
    pub(crate) defaults: RequestDefaults,
}

#[async_trait::async_trait]
impl ChatMiddleware for RequestDefaultsMiddleware {
    async fn on_chat_request(&self, _run_id: &RunId, _step_id: &StepId, req: &mut ChatRequest) {
        if req.temperature.is_none() {
            req.temperature = self.defaults.temperature;
        }
        if req.top_p.is_none() {
            req.top_p = self.defaults.top_p;
        }
        if req.top_k.is_none() {
            req.top_k = self.defaults.top_k;
        }
        if req.stop_sequences.is_empty() && !self.defaults.stop_sequences.is_empty() {
            req.stop_sequences = self.defaults.stop_sequences.clone();
        }
        if req.tool_choice.is_none() {
            req.tool_choice = self.defaults.tool_choice.clone();
        }
        if req.disable_parallel_tool_use.is_none() {
            req.disable_parallel_tool_use = self.defaults.disable_parallel_tool_use;
        }
        if let Some(mt) = self.defaults.max_tokens {
            req.max_tokens = mt;
        }
        if req.additional_params.is_none() {
            req.additional_params = self.defaults.additional_params.clone();
        }
        if let Some(overlay) = &self.defaults.overlay {
            overlay(req);
        }
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
