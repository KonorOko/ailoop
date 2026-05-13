//! `SubAgentTool` — wrap a [`Conversation`] so a parent agent can
//! delegate to it as a regular tool. Pure composition: nothing in the
//! engine or registry changes.
//!
//! The sub-agent's history persists across calls — each invocation sees
//! prior turns. For stateless behavior reconstruct the `SubAgentTool`
//! (or its inner `Conversation`) per call.

use std::time::Duration;

use ailoop_core::{CompletionModel, FinishReason, ToolDefinition, ToolResultContent};
use ailoop_tools::{ToolContext, ToolDyn};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{Conversation, RunOptions};

/// Per-invocation budget overrides applied to every child run dispatched
/// by a [`SubAgentTool`] built through [`SubAgentTool::with_config`].
///
/// Each field is `Option<T>`; `None` (the default) means "fall back to
/// the child [`Conversation`]'s defaults for this run". The values are
/// layered onto the [`RunOptions`] the wrapper assembles per
/// [`ToolDyn::call`], on top of the [`ToolContext`] cancellation
/// inheritance — the cancellation token always comes from the parent
/// context, never from `SubAgentConfig`.
///
/// `SubAgentConfig` is deliberately narrower than [`RunOptions`]:
/// `cancellation` and `run_id` are not exposed because they belong to
/// the parent's dispatch context (cancellation rides through
/// `ToolContext`; the engine mints a fresh `RunId` per child run).
/// Drop down to a hand-built [`RunOptions`] via a custom wrapper if you
/// need that surface.
///
/// Construct fluently — mirrors the [`RunOptions`] builder style so the
/// per-field semantics line up one-to-one:
///
/// ```
/// use std::time::Duration;
/// let config = ailoop::SubAgentConfig::new()
///     .timeout(Duration::from_secs(30))
///     .max_iterations(5);
/// ```
#[derive(Default, Clone, Debug)]
#[non_exhaustive]
pub struct SubAgentConfig {
    /// Wall-clock deadline applied to every child run. Mapped to
    /// [`RunOptions::timeout`] — the engine checks the deadline at
    /// every await boundary and aborts with
    /// [`FinishReason::Aborted`] on expiry, which the wrapper surfaces
    /// as a text-only [`ToolResultContent`] with `is_error: true`.
    pub timeout: Option<Duration>,
    /// Cap on the number of provider turns inside the child run.
    /// Mapped to [`RunOptions::max_iterations`]. Hitting the cap
    /// surfaces as
    /// [`EngineError::MaxIterationsExceeded`](crate::EngineError::MaxIterationsExceeded),
    /// which the wrapper renders as a `"sub-agent error: …"` text body
    /// with `is_error: true`.
    pub max_iterations: Option<usize>,
    /// Per-turn `max_tokens` override for every [`ChatRequest`] the
    /// child run builds. Mapped to [`RunOptions::max_tokens`].
    ///
    /// [`ChatRequest`]: ailoop_core::ChatRequest
    pub max_tokens: Option<u32>,
}

impl SubAgentConfig {
    /// Fresh `SubAgentConfig` with every field unset (equivalent to
    /// [`Default::default`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the wall-clock deadline applied to every child run. See
    /// [`Self::timeout`] for semantics.
    pub fn timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    /// Cap the child's per-run iteration count. See
    /// [`Self::max_iterations`] for semantics.
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = Some(n);
        self
    }

    /// Override `max_tokens` for every [`ChatRequest`] the child
    /// builds. See [`Self::max_tokens`] for semantics.
    ///
    /// [`ChatRequest`]: ailoop_core::ChatRequest
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
}

/// Wraps a [`Conversation`] so a parent agent can delegate to it as a
/// regular tool. Pure composition: nothing in the engine or registry
/// changes — the child runs on its own [`CompletionModel`], history,
/// and middleware chain.
///
/// The child's history persists across calls (each invocation sees
/// prior turns); rebuild the `SubAgentTool` per call if you need
/// stateless behavior. Child errors and aborts are surfaced as a
/// text-only [`ToolResultContent`] (with an `"sub-agent error: …"` /
/// `"sub-agent aborted: …"` prefix and `is_error: true`) — never as a
/// tool-registry error — so the parent's loop continues and the model
/// can distinguish failure from a normal reply.
///
/// The child run inherits a [`child_token`] of the parent's
/// [`ToolContext::cancellation`], so cancelling or timing out the
/// parent run cancels the in-flight sub-agent at the next await
/// boundary.
///
/// Per-invocation budget overrides (`timeout`, `max_iterations`,
/// `max_tokens`) live on [`SubAgentConfig`] and are applied through
/// [`SubAgentTool::with_config`]. The parent's
/// [`ToolContext::cancellation`] always wins over anything in the
/// config — the cancellation handle is wired from the context, not the
/// config, so a parent abort still cuts a child mid-run regardless of
/// per-call budget.
///
/// [`child_token`]: tokio_util::sync::CancellationToken::child_token
/// [`ToolContext::cancellation`]: ailoop_tools::ToolContext::cancellation
///
/// # Examples
///
/// ```no_run
/// # use std::sync::Arc;
/// # async fn build<M>(researcher_model: M, parent_model: M)
/// # -> Result<(), Box<dyn std::error::Error>>
/// # where M: ailoop::CompletionModel + Send + Sync + 'static {
/// // 1. Build the child conversation (its own model, history, prompt).
/// let researcher = ailoop::Conversation::builder(researcher_model)
///     .system_prompt("You are a focused research sub-agent.")
///     .build()?;
///
/// // 2. Wrap it as a tool so the parent can dispatch to it by name.
/// let tool = ailoop::SubAgentTool::new(
///     "researcher",
///     "Delegate a research question to the focused sub-agent.",
///     researcher,
/// );
///
/// // 3. Register on the parent like any other dynamic tool.
/// let _parent = ailoop::Conversation::builder(parent_model)
///     .tool_dyn(Arc::new(tool))
///     .build()?;
/// # Ok(()) }
/// ```
///
/// Cap the child's per-invocation budget when the parent needs to
/// bound runaway loops or long-running sub-tasks:
///
/// ```no_run
/// # use std::sync::Arc;
/// # use std::time::Duration;
/// # async fn build<M>(researcher_model: M)
/// # -> Result<(), Box<dyn std::error::Error>>
/// # where M: ailoop::CompletionModel + Send + Sync + 'static {
/// let researcher = ailoop::Conversation::builder(researcher_model).build()?;
/// let tool = ailoop::SubAgentTool::with_config(
///     "researcher",
///     "Delegate a research question, capped to 5 turns / 30s.",
///     researcher,
///     ailoop::SubAgentConfig::new()
///         .timeout(Duration::from_secs(30))
///         .max_iterations(5),
/// );
/// # let _ = Arc::new(tool);
/// # Ok(()) }
/// ```
pub struct SubAgentTool<M: CompletionModel> {
    name: String,
    description: String,
    conversation: Mutex<Conversation<M>>,
    config: SubAgentConfig,
}

impl<M> SubAgentTool<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    /// Wrap `conversation` as a tool exposing `name` /
    /// `description` to the parent's [`CompletionModel`]. Use
    /// [`Arc::new`](std::sync::Arc::new) when registering through
    /// [`tool_dyn`](crate::ConversationBuilder::tool_dyn).
    ///
    /// Equivalent to
    /// [`Self::with_config(name, description, conversation, SubAgentConfig::default())`](Self::with_config):
    /// no per-invocation budget overrides are applied, so the child
    /// runs under its own builder defaults.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        conversation: Conversation<M>,
    ) -> Self {
        Self::with_config(name, description, conversation, SubAgentConfig::default())
    }

    /// Wrap `conversation` with explicit per-invocation budget
    /// overrides. The `config` is stored on the wrapper and layered
    /// onto the [`RunOptions`] of every child run dispatched through
    /// [`ToolDyn::call`]. The parent's `ToolContext::cancellation`
    /// always wins over the config — only `timeout`, `max_iterations`,
    /// and `max_tokens` are configurable per-invocation.
    pub fn with_config(
        name: impl Into<String>,
        description: impl Into<String>,
        conversation: Conversation<M>,
        config: SubAgentConfig,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            conversation: Mutex::new(conversation),
            config,
        }
    }
}

#[async_trait::async_trait]
impl<M> ToolDyn for SubAgentTool<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    fn name(&self) -> String {
        self.name.clone()
    }

    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            &self.name,
            &self.description,
            json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Instruction or question to delegate to the sub-agent."
                    }
                },
                "required": ["prompt"]
            }),
            vec![],
        )
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> ToolResultContent {
        let prompt = args
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();

        let mut options = RunOptions::new().cancellation(ctx.cancellation().child_token());
        if let Some(timeout) = self.config.timeout {
            options = options.timeout(timeout);
        }
        if let Some(max_iterations) = self.config.max_iterations {
            options = options.max_iterations(max_iterations);
        }
        if let Some(max_tokens) = self.config.max_tokens {
            options = options.max_tokens(max_tokens);
        }

        let mut conv = self.conversation.lock().await;
        match conv.run_with_options(prompt, options).await {
            Ok(outcome) => {
                let text = outcome.final_text.unwrap_or_default();
                match outcome.finish_reason {
                    FinishReason::Aborted(reason) if text.is_empty() => {
                        ToolResultContent::text(format!("sub-agent aborted: {reason}"))
                            .with_is_error(true)
                    }
                    FinishReason::Aborted(reason) => {
                        ToolResultContent::text(format!("sub-agent aborted ({reason}): {text}"))
                            .with_is_error(true)
                    }
                    _ => ToolResultContent::text(text),
                }
            }
            Err(e) => ToolResultContent::text(format!("sub-agent error: {e}")).with_is_error(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::testing::{ScriptedError, ScriptedModel};
    use ailoop_core::{
        CancellationToken, ChatMiddleware, ChatRequest, HookAction, Message, RunConfig, RunId,
        StepId, StreamChunk, Usage,
    };
    use ailoop_tools::ToolActivation;
    use std::sync::{Arc, Mutex as StdMutex};

    fn one_text_turn(text: &str) -> Vec<StreamChunk> {
        vec![
            StreamChunk::TextDelta { delta: text.into() },
            StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                usage: Usage::default(),
                service_tier: None,
            },
        ]
    }

    /// The supervisor sees the sub-agent's final text as the tool result.
    #[tokio::test]
    async fn sub_agent_tool_returns_final_text() {
        let model = ScriptedModel::new([one_text_turn("delegated answer")]);
        let conv = Conversation::builder(model).build().expect("build");
        let tool = SubAgentTool::new("delegate", "delegate to a sub-agent", conv);

        let result = tool
            .call(json!({"prompt": "do the thing"}), &ToolContext::detached())
            .await;
        assert_eq!(result.as_text(), Some("delegated answer"));
        assert!(!result.is_error);
    }

    /// History persists between calls: the second invocation's
    /// `ChatRequest` carries the first prompt + first reply.
    #[tokio::test]
    async fn sub_agent_history_persists_between_calls() {
        struct Recorder {
            captures: Arc<StdMutex<Vec<Vec<Message>>>>,
        }
        #[async_trait::async_trait]
        impl ChatMiddleware for Recorder {
            async fn on_chat_request(&self, _: &RunId, _: &StepId, req: &mut ChatRequest) {
                self.captures.lock().unwrap().push(req.messages.clone());
            }
        }

        let model = ScriptedModel::new([one_text_turn("first"), one_text_turn("second")]);
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let conv = Conversation::builder(model)
            .middleware(Arc::new(Recorder {
                captures: captures.clone(),
            }))
            .build()
            .expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let first = tool
            .call(json!({"prompt": "P1"}), &ToolContext::detached())
            .await;
        let second = tool
            .call(json!({"prompt": "P2"}), &ToolContext::detached())
            .await;

        assert_eq!(first.as_text(), Some("first"));
        assert_eq!(second.as_text(), Some("second"));

        let captures = captures.lock().unwrap();
        assert_eq!(captures.len(), 2, "expected one capture per turn");

        // First request: only the first user prompt.
        let user_texts_turn1: Vec<String> = captures[0]
            .iter()
            .filter_map(|m| match m {
                Message::User { blocks } => Some(
                    blocks
                        .iter()
                        .filter_map(|b| match b {
                            ailoop_core::UserBlock::Text { text, .. } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                ),
                _ => None,
            })
            .collect();
        assert_eq!(user_texts_turn1, vec!["P1".to_string()]);

        // Second request: P1, assistant reply "first", P2 — proves the
        // sub-agent kept its history across tool invocations.
        assert_eq!(
            captures[1].len(),
            3,
            "second turn should see P1 + assistant + P2"
        );
        let last_user = captures[1]
            .iter()
            .rev()
            .find_map(|m| match m {
                Message::User { blocks } => blocks.iter().find_map(|b| match b {
                    ailoop_core::UserBlock::Text { text, .. } => Some(text.clone()),
                    _ => None,
                }),
                _ => None,
            })
            .expect("user block");
        assert_eq!(last_user, "P2");
    }

    /// An aborted sub-agent run surfaces as `ToolResultContent::Text`
    /// with a meaningful message and `is_error: true` — never as `Err`.
    #[tokio::test]
    async fn sub_agent_aborted_run_surfaces_as_text() {
        struct AbortMw;
        #[async_trait::async_trait]
        impl ChatMiddleware for AbortMw {
            async fn on_run_started(&self, _: &RunId, _: &[Message], _: &RunConfig) -> HookAction {
                HookAction::Terminate {
                    reason: "policy".into(),
                }
            }
        }

        let model = ScriptedModel::new(Vec::<Vec<StreamChunk>>::new());
        let conv = Conversation::builder(model)
            .middleware(Arc::new(AbortMw))
            .build()
            .expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let result = tool
            .call(json!({"prompt": "anything"}), &ToolContext::detached())
            .await;
        let text = result.as_text().expect("expected text body on abort");
        assert!(
            text.contains("aborted") && text.contains("policy"),
            "expected abort reason in text, got {text:?}"
        );
        assert!(result.is_error, "aborted runs must mark is_error: true");
    }

    /// Cancelling the parent's run-wide token aborts the in-flight
    /// sub-agent: the child receives a `child_token()` of the parent's
    /// cancellation handle, so `cancel()` on the parent fires inside
    /// the child run too. The wrapper surfaces the abort as text with
    /// `is_error: true`.
    #[tokio::test]
    async fn sub_agent_parent_cancellation_aborts_child() {
        let model = ScriptedModel::new([one_text_turn("never delivered")]);
        let conv = Conversation::builder(model).build().expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let parent_token = CancellationToken::new();
        let ctx = ToolContext::new(
            RunId::new(),
            StepId::new(),
            ToolActivation::detached(),
            parent_token.clone(),
        );

        // Pre-cancel before calling — the run aborts at the first await
        // boundary, before the model emits anything.
        parent_token.cancel();

        let result = tool.call(json!({"prompt": "stop me"}), &ctx).await;
        let text = result.as_text().expect("expected text body on abort");
        assert!(
            text.contains("aborted") && text.contains("cancelled by caller"),
            "expected cancelled-by-caller in abort text, got {text:?}"
        );
        assert!(
            result.is_error,
            "parent cancellation must mark the child result is_error: true"
        );
    }

    /// `SubAgentConfig::max_iterations` per-call caps the child even when
    /// the child's [`ConversationBuilder`] would otherwise allow more
    /// turns. With `max_iterations(0)`, the engine bails on the first
    /// iteration check with [`EngineError::MaxIterationsExceeded`], which
    /// the wrapper surfaces as `"sub-agent error: …"` + `is_error: true`.
    ///
    /// [`ConversationBuilder`]: crate::ConversationBuilder
    /// [`EngineError::MaxIterationsExceeded`]: crate::EngineError::MaxIterationsExceeded
    #[tokio::test]
    async fn sub_agent_config_max_iterations_caps_child() {
        // Child builder leaves max_iterations at the engine default
        // (10); the per-call cap of 0 must still win.
        let model = ScriptedModel::new([one_text_turn("never reached")]);
        let conv = Conversation::builder(model).build().expect("build");
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            conv,
            SubAgentConfig::new().max_iterations(0),
        );

        let result = tool
            .call(json!({"prompt": "anything"}), &ToolContext::detached())
            .await;
        let text = result
            .as_text()
            .expect("expected text body when max_iterations is exceeded");
        assert!(
            text.starts_with("sub-agent error:") && text.contains("max iterations"),
            "expected max-iterations error body, got {text:?}"
        );
        assert!(
            result.is_error,
            "max_iterations exceeded must mark is_error: true"
        );
    }

    /// `SubAgentConfig::timeout` per-call aborts the child run. The
    /// engine races the abort future against every await — here a
    /// sleeping `on_run_started` middleware never returns, but the
    /// 50ms timeout fires first and surfaces as a text-only result
    /// with `is_error: true`.
    #[tokio::test]
    async fn sub_agent_config_timeout_aborts_child() {
        struct SlowMw;
        #[async_trait::async_trait]
        impl ChatMiddleware for SlowMw {
            async fn on_run_started(&self, _: &RunId, _: &[Message], _: &RunConfig) -> HookAction {
                tokio::time::sleep(Duration::from_secs(60)).await;
                HookAction::Continue
            }
        }

        let model = ScriptedModel::new([one_text_turn("never reached")]);
        let conv = Conversation::builder(model)
            .middleware(Arc::new(SlowMw))
            .build()
            .expect("build");
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            conv,
            SubAgentConfig::new().timeout(Duration::from_millis(50)),
        );

        let result = tool
            .call(json!({"prompt": "anything"}), &ToolContext::detached())
            .await;
        let text = result
            .as_text()
            .expect("expected text body on timeout abort");
        assert!(
            text.contains("aborted") && text.contains("timeout exceeded"),
            "expected timeout-exceeded abort text, got {text:?}"
        );
        assert!(
            result.is_error,
            "timeout-aborted runs must mark is_error: true"
        );
    }

    /// `SubAgentConfig::max_tokens` per-call reaches the per-turn
    /// [`ChatRequest`] the engine builds. Spying via `on_chat_request`
    /// shows the override taking effect on the first dispatch.
    #[tokio::test]
    async fn sub_agent_config_max_tokens_reaches_chat_request() {
        struct ReqSpy {
            captured: Arc<StdMutex<Option<u32>>>,
        }
        #[async_trait::async_trait]
        impl ChatMiddleware for ReqSpy {
            async fn on_chat_request(&self, _: &RunId, _: &StepId, req: &mut ChatRequest) {
                *self.captured.lock().unwrap() = Some(req.max_tokens);
            }
        }

        let captured = Arc::new(StdMutex::new(None));
        let model = ScriptedModel::new([one_text_turn("ok")]);
        let conv = Conversation::builder(model)
            .middleware(Arc::new(ReqSpy {
                captured: captured.clone(),
            }))
            .build()
            .expect("build");
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            conv,
            SubAgentConfig::new().max_tokens(321),
        );

        let result = tool
            .call(json!({"prompt": "anything"}), &ToolContext::detached())
            .await;
        assert!(!result.is_error, "happy path must not be flagged as error");
        assert_eq!(
            *captured.lock().unwrap(),
            Some(321),
            "ChatRequest.max_tokens must reflect SubAgentConfig.max_tokens"
        );
    }

    /// `SubAgentTool::new` (no per-invocation config) leaves the
    /// engine's [`RunConfig`] untouched: budget knobs stay at their
    /// defaults, exactly as before this PR. Anchors the no-breaking-
    /// change contract.
    #[tokio::test]
    async fn sub_agent_new_without_config_leaves_run_config_at_defaults() {
        struct ConfigSpy {
            max_iterations: Arc<StdMutex<Option<usize>>>,
            max_tokens: Arc<StdMutex<Option<u32>>>,
            timeout: Arc<StdMutex<Option<Option<Duration>>>>,
        }
        #[async_trait::async_trait]
        impl ChatMiddleware for ConfigSpy {
            async fn on_run_started(
                &self,
                _: &RunId,
                _: &[Message],
                config: &RunConfig,
            ) -> HookAction {
                *self.max_iterations.lock().unwrap() = Some(config.max_iterations);
                *self.max_tokens.lock().unwrap() = Some(config.max_tokens);
                *self.timeout.lock().unwrap() = Some(config.timeout);
                HookAction::Continue
            }
        }

        let max_iterations = Arc::new(StdMutex::new(None));
        let max_tokens = Arc::new(StdMutex::new(None));
        let timeout = Arc::new(StdMutex::new(None));
        let model = ScriptedModel::new([one_text_turn("ok")]);
        let conv = Conversation::builder(model)
            .middleware(Arc::new(ConfigSpy {
                max_iterations: max_iterations.clone(),
                max_tokens: max_tokens.clone(),
                timeout: timeout.clone(),
            }))
            .build()
            .expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let _ = tool
            .call(json!({"prompt": "anything"}), &ToolContext::detached())
            .await;

        // RunConfig defaults — see `ailoop-core/src/config.rs`.
        assert_eq!(*max_iterations.lock().unwrap(), Some(10));
        assert_eq!(*max_tokens.lock().unwrap(), Some(4096));
        assert_eq!(
            *timeout.lock().unwrap(),
            Some(None),
            "no per-call timeout when SubAgentTool::new is used"
        );
    }

    /// `Conversation::run_with_options` returning `Err(_)` surfaces as
    /// `ToolResultContent::Text` with an `"sub-agent error: …"` prefix
    /// and `is_error: true` — the parent can react without the tool
    /// dispatch itself failing.
    #[tokio::test]
    async fn sub_agent_engine_error_surfaces_as_is_error_text() {
        let model = ScriptedModel::with_turns([Err(ScriptedError("permanent: bad auth".into()))]);
        let conv = Conversation::builder(model).build().expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let result = tool
            .call(json!({"prompt": "anything"}), &ToolContext::detached())
            .await;
        let text = result.as_text().expect("expected text body on error");
        assert!(
            text.starts_with("sub-agent error:") && text.contains("bad auth"),
            "expected sub-agent error prefix and message, got {text:?}"
        );
        assert!(
            result.is_error,
            "engine errors must mark the result is_error: true"
        );
    }
}
