use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use ailoop_core::{
    ChatMiddleware, ChatRequest, Message, ReasoningEffort, RunErrorInfo, RunFinishedInfo, RunId,
    StepId, StepInfo, SystemBlock, SystemPrompt, ToolCallInfo, ToolChoice, ToolDecision, ToolTag,
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

/// Renders the builder's system prompt (base + active tool-group
/// sections) and composes it with whatever a user middleware already
/// wrote to `req.system_prompt`. Runs after user middlewares so it sees
/// the final `req.tools`; see [`compose_system_prompt`] for the merge
/// rule.
pub(crate) struct SystemPromptMiddleware {
    pub(crate) base: Prompt,
    pub(crate) tools_sections: Vec<ToolPromptGroup>,
}

#[async_trait::async_trait]
impl ChatMiddleware for SystemPromptMiddleware {
    async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
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

        req.system_prompt = Some(compose_system_prompt(
            prompt.render(),
            req.system_prompt.take(),
        ));
    }
}

/// Merge the builder-rendered prompt with a user-supplied one.
///
/// Inside a `Conversation` the engine hands middlewares a request with
/// `system_prompt: None`, so a `Some(_)` here was put there by a user
/// middleware. The builder prompt goes first (stable prefix, better
/// prompt-cache hits) and the user's goes after it:
///
/// - no user prompt → the builder prompt, as before;
/// - empty builder prompt → the user prompt, untouched;
/// - `Plain + Plain` → one string, separated by a blank line;
/// - user `Blocks` → `Blocks`, with the builder prompt as a leading
///   block without cache breakpoint and the user's blocks (and their
///   `cache_control`) preserved as-is.
fn compose_system_prompt(builder: String, user: Option<SystemPrompt>) -> SystemPrompt {
    let Some(user) = user else {
        return builder.into();
    };
    let builder = builder.trim_end();
    if builder.is_empty() {
        return user;
    }
    match user {
        SystemPrompt::Plain(user) => SystemPrompt::Plain(format!("{builder}\n\n{user}")),
        SystemPrompt::Blocks(user) => {
            let mut blocks = Vec::with_capacity(user.len() + 1);
            blocks.push(SystemBlock::new(builder));
            blocks.extend(user);
            SystemPrompt::Blocks(blocks)
        }
        // `SystemPrompt` is `#[non_exhaustive]`; fall back to flattening
        // any future variant to text rather than dropping it.
        other => SystemPrompt::Plain(format!("{builder}\n\n{}", other.as_text())),
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
    pub(crate) parallel_tool_use: Option<bool>,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
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
            || self.parallel_tool_use.is_some()
            || self.reasoning_effort.is_some()
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
    async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
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
        if req.parallel_tool_use.is_none() {
            req.parallel_tool_use = self.defaults.parallel_tool_use;
        }
        if req.reasoning_effort.is_none() {
            req.reasoning_effort = self.defaults.reasoning_effort;
        }
        if req.additional_params.is_none() {
            req.additional_params = self.defaults.additional_params.clone();
        }
        if let Some(overlay) = &self.defaults.overlay {
            overlay(req);
        }
    }
}

/// What an approval callback receives for one gated tool call.
///
/// Built by [`ApprovalMiddleware`] right before the engine runs the
/// tool. Fields are public so a callback can read or move them out
/// directly (`req.name`, `req.args`); the type is
/// `#[non_exhaustive]` so new context can be added without breaking
/// callbacks. Use [`new`](Self::new) plus the `with_*` setters to build
/// one outside the crate, e.g. to unit-test a verifier.
///
/// # Model-based verifier
///
/// The callback is async and can call any
/// [`CompletionModel`](ailoop_core::CompletionModel), so a risky-action
/// verifier fits here. Recommended shape:
///
/// 1. **Tags decide what gets reviewed.** Gate only what needs it with
///    [`approval_for_tags`](crate::ConversationBuilder::approval_for_tags)
///    (or the default [`approval`](crate::ConversationBuilder::approval)
///    for `Destructive` / `WritesFiles`); everything else runs without
///    paying for a verifier call.
/// 2. **The verifier judges the call against the user's intent** and
///    answers allow, deny or escalate. Allow → [`ToolDecision::Continue`];
///    deny → [`ToolDecision::Skip`] with a reason the model can read.
/// 3. **A human handles what the verifier escalates.**
///
/// Fail closed: bound the verifier with a timeout and treat a model
/// error, a timeout or an unparseable answer as deny or escalate,
/// never as `Continue`.
///
/// Take intent only from what the user wrote. [`messages`](Self::messages)
/// also carries tool results (`UserBlock::ToolResult`) and earlier
/// assistant turns; those can contain text injected by a web page, a
/// file or an MCP server, so hand them to the verifier as untrusted
/// data, not as instructions. `rm -rf build/` is reasonable after "clean
/// the build" and alarming after "summarize this file" — the user's
/// text, not a tool's output, is what tells the two apart.
///
/// ```no_run
/// use std::time::Duration;
/// use ailoop::{ApprovalRequest, Message, ToolDecision, UserBlock};
///
/// enum Verdict { Allow, Deny(String), Escalate }
///
/// // Your verifier: prompt a model with `intent` and the call, parse
/// // its answer. Errors surface as `Err`.
/// async fn verify(intent: &str, req: &ApprovalRequest) -> Result<Verdict, String> {
///     # let _ = (intent, req);
///     # unimplemented!()
/// }
/// async fn ask_human(req: &ApprovalRequest) -> ToolDecision {
///     # let _ = req;
///     # unimplemented!()
/// }
///
/// async fn gate(req: ApprovalRequest) -> ToolDecision {
///     // User-authored text only; tool results are not intent.
///     let intent: Vec<&str> = req
///         .messages
///         .iter()
///         .filter_map(|m| match m {
///             Message::User { blocks } => Some(blocks),
///             _ => None,
///         })
///         .flatten()
///         .filter_map(|b| match b {
///             UserBlock::Text { text, .. } => Some(text.as_str()),
///             _ => None,
///         })
///         .collect();
///     let intent = intent.join("\n");
///
///     match tokio::time::timeout(Duration::from_secs(10), verify(&intent, &req)).await {
///         Ok(Ok(Verdict::Allow)) => ToolDecision::Continue,
///         Ok(Ok(Verdict::Deny(reason))) => ToolDecision::Skip { reason },
///         // Escalation, verifier error and timeout all go to a human.
///         Ok(Ok(Verdict::Escalate)) | Ok(Err(_)) | Err(_) => ask_human(&req).await,
///     }
/// }
///
/// # fn wire(builder: ailoop::ConversationBuilder<impl ailoop::CompletionModel>) {
/// let builder = builder.approval(gate);
/// # let _ = builder;
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ApprovalRequest {
    /// Run the call belongs to.
    pub run_id: RunId,
    /// Step (model turn) that produced the call.
    pub step_id: StepId,
    /// Provider-assigned id of the call, the same as `call_id` on
    /// [`ToolCallInfo`] and on the `ToolResult` chunk.
    pub call_id: String,
    /// Wire name of the tool.
    pub name: String,
    /// Arguments the tool will run with, after every earlier
    /// middleware's `on_before_tool_call_mut`.
    pub args: Value,
    /// Tags the tool declares. Filled when the gate was installed with
    /// a `ConversationBuilder::approval*` method; empty for
    /// [`ApprovalMiddleware::approve_all`] and
    /// [`ApprovalMiddleware::approve_named`], which do not see the
    /// registry.
    pub tags: Arc<[ToolTag]>,
    /// Context sent to the model on the step that produced this call,
    /// as left by every middleware's `on_chat_request` (it can be
    /// compacted or rewritten relative to the stored history). The call
    /// being approved is not in it — see `name` / `args`. Shared,
    /// not copied, between the gated calls of a step.
    pub messages: Arc<[Message]>,
}

impl ApprovalRequest {
    /// A request for the call `call` with arguments `args`. The run,
    /// step, call id and tool name come from `call`; `tags` and
    /// `messages` start empty, set them with
    /// [`with_tags`](Self::with_tags) and
    /// [`with_messages`](Self::with_messages).
    pub fn new(call: ToolCallInfo, args: Value) -> Self {
        Self {
            run_id: call.run_id,
            step_id: call.step_id,
            call_id: call.call_id,
            name: call.name,
            args,
            tags: Arc::from([]),
            messages: Arc::from([]),
        }
    }

    /// Replace `tags`.
    pub fn with_tags(mut self, tags: impl Into<Arc<[ToolTag]>>) -> Self {
        self.tags = tags.into();
        self
    }

    /// Replace `messages`.
    pub fn with_messages(mut self, messages: impl Into<Arc<[Message]>>) -> Self {
        self.messages = messages.into();
        self
    }
}

pub(crate) type ApprovalCallback =
    Arc<dyn Fn(ApprovalRequest) -> BoxFuture<'static, ToolDecision> + Send + Sync>;

enum GatePolicy {
    All,
    ByName(HashSet<String>),
}

/// Middleware that asks a user-supplied async callback whether each tool
/// call should proceed, returning the callback's [`ToolDecision`] to the
/// engine.
///
/// Construct via [`approve_all`](Self::approve_all) for an unconditional
/// gate, or via [`approve_named`](Self::approve_named) for an explicit set of
/// tool names. For tag-based gating, use the builder method
/// `ConversationBuilder::approval`.
///
/// The callback receives an [`ApprovalRequest`] with the call and the
/// context the model saw on that step; see its docs for the
/// model-based verifier pattern. To fill `messages`, the middleware
/// records each step's request in `on_chat_request`, so it must sit
/// after any middleware that rewrites `req.messages` (the builder
/// always puts it last). That state is keyed by [`RunId`] — one
/// instance can be shared across concurrent runs — and dropped in
/// `on_run_finished`, `on_run_error` or `on_run_dropped`.
pub struct ApprovalMiddleware {
    callback: ApprovalCallback,
    policy: GatePolicy,
    tags: HashMap<String, Arc<[ToolTag]>>,
    contexts: Mutex<HashMap<RunId, Arc<[Message]>>>,
}

impl ApprovalMiddleware {
    /// Wire the callback for every tool call, regardless of tags.
    pub fn approve_all<F, Fut>(callback: F) -> Self
    where
        F: Fn(ApprovalRequest) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ToolDecision> + Send + 'static,
    {
        Self::new(wrap_callback(callback), GatePolicy::All, HashMap::new())
    }

    /// Wire the callback for tool calls whose name appears in `names`.
    /// Other tool calls pass through with `Continue`.
    ///
    /// Matching is by exact wire name at call time, independent of
    /// whether the tool is currently active — so listing a deferred
    /// tool here gates it once a handler activates it mid-run.
    pub fn approve_named<I, S, F, Fut>(names: I, callback: F) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        F: Fn(ApprovalRequest) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ToolDecision> + Send + 'static,
    {
        Self::new(
            wrap_callback(callback),
            GatePolicy::ByName(names.into_iter().map(Into::into).collect()),
            HashMap::new(),
        )
    }

    pub(crate) fn from_parts(
        callback: ApprovalCallback,
        names: HashSet<String>,
        tags: HashMap<String, Arc<[ToolTag]>>,
    ) -> Self {
        Self::new(callback, GatePolicy::ByName(names), tags)
    }

    pub(crate) fn from_parts_all(
        callback: ApprovalCallback,
        tags: HashMap<String, Arc<[ToolTag]>>,
    ) -> Self {
        Self::new(callback, GatePolicy::All, tags)
    }

    fn new(
        callback: ApprovalCallback,
        policy: GatePolicy,
        tags: HashMap<String, Arc<[ToolTag]>>,
    ) -> Self {
        Self {
            callback,
            policy,
            tags,
            contexts: Mutex::new(HashMap::new()),
        }
    }

    fn should_gate(&self, name: &str) -> bool {
        match &self.policy {
            GatePolicy::All => true,
            GatePolicy::ByName(set) => set.contains(name),
        }
    }

    fn contexts(&self) -> std::sync::MutexGuard<'_, HashMap<RunId, Arc<[Message]>>> {
        // Poisoning only means another thread panicked mid-insert; the
        // map itself is still consistent.
        self.contexts.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(test)]
    pub(crate) fn tracked_runs(&self) -> usize {
        self.contexts().len()
    }
}

pub(crate) fn wrap_callback<F, Fut>(callback: F) -> ApprovalCallback
where
    F: Fn(ApprovalRequest) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = ToolDecision> + Send + 'static,
{
    Arc::new(move |req| Box::pin(callback(req)) as BoxFuture<'static, ToolDecision>)
}

#[async_trait::async_trait]
impl ChatMiddleware for ApprovalMiddleware {
    async fn on_chat_request(&self, step: &StepInfo, req: &mut ChatRequest) {
        // One copy per step, shared by every gated call of the step.
        // Overwrites the previous step (and a same-step retry after
        // context-overflow recovery).
        let messages: Arc<[Message]> = Arc::from(req.messages.as_slice());
        self.contexts().insert(step.run_id.clone(), messages);
    }

    async fn on_before_tool_call(&self, call: &ToolCallInfo, args: &Value) -> ToolDecision {
        if !self.should_gate(&call.name) {
            return ToolDecision::Continue;
        }
        let messages = self
            .contexts()
            .get(&call.run_id)
            .cloned()
            .unwrap_or_else(|| Arc::from([]));
        let tags = self
            .tags
            .get(&call.name)
            .cloned()
            .unwrap_or_else(|| Arc::from([]));
        let req = ApprovalRequest::new(call.clone(), args.clone())
            .with_tags(tags)
            .with_messages(messages);
        (self.callback)(req).await
    }

    async fn on_run_finished(&self, run: &RunFinishedInfo<'_>) {
        self.contexts().remove(run.run_id);
    }

    async fn on_run_error(&self, run: &RunErrorInfo<'_>) {
        self.contexts().remove(run.run_id);
    }

    fn on_run_dropped(&self, run_id: &RunId) {
        self.contexts().remove(run_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{FinishReason, Usage};

    fn gate() -> ApprovalMiddleware {
        ApprovalMiddleware::approve_all(|_req| async { ToolDecision::Continue })
    }

    async fn record_step(mw: &ApprovalMiddleware, run_id: &RunId) {
        let mut req = ChatRequest::new(vec![Message::user("hi")], 1024);
        mw.on_chat_request(&StepInfo::new(run_id.clone(), StepId::new()), &mut req)
            .await;
    }

    #[test]
    fn approval_request_new_takes_identity_from_call() {
        let call = ToolCallInfo::new(RunId::new(), StepId::new(), "toolu_1", "rm");
        let req = ApprovalRequest::new(call.clone(), serde_json::json!({"path": "/tmp"}));
        assert_eq!(req.run_id, call.run_id);
        assert_eq!(req.step_id, call.step_id);
        assert_eq!(req.call_id, "toolu_1");
        assert_eq!(req.name, "rm");
        assert!(req.tags.is_empty());
        assert!(req.messages.is_empty());
    }

    #[tokio::test]
    async fn run_context_is_dropped_on_run_finished() {
        let mw = gate();
        let run_id = RunId::new();
        record_step(&mw, &run_id).await;
        assert_eq!(mw.tracked_runs(), 1);

        mw.on_run_finished(&RunFinishedInfo::new(
            &run_id,
            &FinishReason::EndTurn,
            &Usage::default(),
            &[],
        ))
        .await;
        assert_eq!(mw.tracked_runs(), 0);
    }

    #[tokio::test]
    async fn run_context_is_dropped_on_run_error() {
        let mw = gate();
        let run_id = RunId::new();
        record_step(&mw, &run_id).await;
        let other = RunId::new();
        record_step(&mw, &other).await;
        assert_eq!(mw.tracked_runs(), 2);

        let err = std::io::Error::other("boom");
        mw.on_run_error(&RunErrorInfo::new(&run_id, &err, &Usage::default(), &[]))
            .await;
        assert_eq!(mw.tracked_runs(), 1, "only the failed run is dropped");
    }
}
