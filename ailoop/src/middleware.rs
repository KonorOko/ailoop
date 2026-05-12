use std::collections::HashSet;
use std::sync::Arc;

use ailoop_core::{
    ChatMiddleware, ChatRequest, ReasoningEffort, RunId, StepId, ToolChoice, ToolDecision,
};
use futures::future::BoxFuture;
use serde_json::Value;

use crate::{Prompt, PromptSection};

/// A `PromptSection` shared by a named set of tools. The section is
/// appended to the system prompt at most once per turn when at least
/// one of `tools` is active in the request — see
/// [`SystemPromptMiddleware::on_chat_request`].
pub(crate) struct ToolPromptGroup {
    pub(crate) tools: HashSet<String>,
    pub(crate) section: PromptSection,
}

pub(crate) struct SystemPromptMiddleware {
    pub(crate) base: Prompt,
    pub(crate) tools_sections: Vec<ToolPromptGroup>,
}

#[async_trait::async_trait]
impl ChatMiddleware for SystemPromptMiddleware {
    async fn on_chat_request(&self, _run_id: &RunId, _step_id: &StepId, req: &mut ChatRequest) {
        let mut prompt = self.base.clone();

        if let Some(tools) = &req.tools {
            // Walk groups in registration order. Each group whose tool
            // set intersects the request's active tools contributes its
            // section exactly once, regardless of how many of the
            // group's tools are active — that's the whole point of the
            // grouping API (no per-tool duplication of a shared guide).
            for group in &self.tools_sections {
                let active = tools.iter().any(|tool| group.tools.contains(&tool.name));
                if active {
                    prompt.add_section(group.section.clone());
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
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
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
            || self.reasoning_effort.is_some()
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
        if req.reasoning_effort.is_none() {
            req.reasoning_effort = self.defaults.reasoning_effort;
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
