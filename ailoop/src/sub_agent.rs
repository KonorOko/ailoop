//! `SubAgentTool` — wrap a [`Conversation`] so a parent agent can
//! delegate to it as a regular tool. Pure composition: nothing in the
//! engine or registry changes.
//!
//! The sub-agent's history persists across calls — each invocation sees
//! prior turns. For stateless behavior reconstruct the `SubAgentTool`
//! (or its inner `Conversation`) per call.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use ailoop_core::{
    AbortReason, ChatMiddleware, ChatRequest, CompletionModel, FinishReason, Message,
    ProviderError, RunStartInfo, Source, StepInfo, StreamChunk, ToolChoice, ToolDefinition,
    ToolResultContent, UserBlock,
};
use ailoop_tools::{ToolContext, ToolDyn};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::time::Instant;

use crate::{Conversation, RunOptions};

/// Parse a single entry from the JSON schema's `attachments` array into
/// a [`UserBlock`]. The wire form mirrors Anthropic's content blocks
/// (`{"type": "image"|"document", "source": {"type": "base64",
/// "media_type": "…", "data": "…"} | {"type": "url", "url": "…"} |
/// {"type": "file_id", "id": "…"}}`) so a model fluent in that shape
/// can produce attachments natively. The variant tag dispatches to
/// [`UserBlock::Image`] or [`UserBlock::Document`]; `source` flows
/// through [`Source`]'s own `Deserialize` impl, so the `base64` / `url`
/// / `file_id` discriminator is the same one providers use.
///
/// Done by hand rather than via a `#[derive(Deserialize)]` enum to
/// avoid pulling `serde` into `ailoop`'s runtime deps just for this
/// one shape.
fn parse_attachment(value: &Value) -> Result<UserBlock, String> {
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "attachment missing required `type` field".to_string())?;
    let source_value = value
        .get("source")
        .ok_or_else(|| "attachment missing required `source` field".to_string())?;
    let source: Source = serde_json::from_value(source_value.clone())
        .map_err(|e| format!("invalid `source`: {e}"))?;
    match kind {
        "image" => Ok(UserBlock::image(source)),
        "document" => Ok(UserBlock::document(source)),
        other => Err(format!(
            "unknown attachment `type` {other:?}: expected \"image\" or \"document\""
        )),
    }
}

fn parse_attachments(value: &Value) -> Result<Vec<UserBlock>, String> {
    let entries = value
        .as_array()
        .ok_or_else(|| "`attachments` must be an array".to_string())?;
    entries
        .iter()
        .enumerate()
        .map(|(i, v)| parse_attachment(v).map_err(|e| format!("attachments[{i}]: {e}")))
        .collect()
}

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
    /// as a text-only [`ToolResultContent`] with `is_error: true`
    /// (unless [`Self::wrap_up`] got a summary out first).
    pub timeout: Option<Duration>,
    /// Cap on the number of provider turns inside the child run.
    /// Mapped to [`RunOptions::max_iterations`]. Hitting the cap
    /// aborts the child with [`FinishReason::Aborted`], which the
    /// wrapper renders as `"sub-agent aborted (agent loop exceeded max
    /// iterations (n)): <partial text>"` with `is_error: true`, so
    /// whatever the child wrote before the cap reaches the parent.
    /// With [`Self::wrap_up`] set, the last allowed iteration becomes
    /// a no-tools summary turn instead. `None` leaves the child at the
    /// engine default (25 iterations, see
    /// [`RunConfig::max_iterations`](ailoop_core::RunConfig::max_iterations)).
    pub max_iterations: Option<usize>,
    /// Per-turn `max_tokens` override for every [`ChatRequest`] the
    /// child run builds. Mapped to [`RunOptions::max_tokens`], so it
    /// takes precedence over the child's
    /// [`ConversationBuilder::max_tokens`](crate::ConversationBuilder::max_tokens);
    /// `None` falls through to the child's builder default (or 4096).
    /// A middleware on the child that rewrites `req.max_tokens` still
    /// wins.
    ///
    /// [`ChatRequest`]: ailoop_core::ChatRequest
    pub max_tokens: Option<u32>,
    /// Opt-in graceful cutoff. When set, the child is forced to spend
    /// its last turn summarizing — no tool calls — before the hard
    /// `timeout` / `max_iterations` cutoff, and the summary reaches the
    /// parent marked as partial with `is_error: false`. `None` (the
    /// default) keeps the plain abort behavior. See [`WrapUp`].
    pub wrap_up: Option<WrapUp>,
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

    /// Enable the graceful cutoff. See [`Self::wrap_up`] and
    /// [`WrapUp`] for semantics.
    pub fn wrap_up(mut self, wrap_up: WrapUp) -> Self {
        self.wrap_up = Some(wrap_up);
        self
    }
}

/// Default instruction injected into the child's wrap-up turn. See
/// [`WrapUp::instruction`].
pub const DEFAULT_WRAP_UP_INSTRUCTION: &str = "You are about to run out of budget for this task. \
Do not call any more tools. Reply now with your final answer: summarize what you have found so \
far and state clearly what remains unverified or unfinished.";

/// Graceful-cutoff settings for a [`SubAgentTool`], enabled through
/// [`SubAgentConfig::wrap_up`].
///
/// Without it, a child that hits its `timeout` or `max_iterations` is
/// cut off and the parent receives `"sub-agent aborted (…)"` with
/// `is_error: true` — whatever the child learned after its last text
/// block is lost. With it, the wrapper forces one **wrap-up turn**
/// before the hard cutoff:
///
/// - **Iteration budget:** the last allowed iteration
///   (`max_iterations - 1`, zero-based) is always the wrap-up turn.
/// - **Time budget:** once `timeout * time_fraction` has elapsed since
///   the run started, the next request is the wrap-up turn. The check
///   runs when a request is about to be sent, so a model turn or tool
///   call already in flight is not interrupted; pick a fraction that
///   leaves room for one tool call plus one model turn before the hard
///   deadline. Without a timeout only the iteration trigger applies.
///
/// Both budgets are read from the child run's effective
/// [`RunConfig`](ailoop_core::RunConfig), so they apply whether they come from
/// [`SubAgentConfig`] or from the child's builder defaults.
///
/// The wrap-up request keeps its tool definitions (so the prompt cache
/// is not invalidated) but sets `tool_choice` to
/// [`ToolChoice::None`], and [`Self::instruction`] is appended as a
/// text block to the request's last user message. The instruction is
/// request-only: it is never written to the child's history. Once
/// triggered, every later request of the same run is forced too. The
/// wrap-up middleware runs after every builder middleware, so a user
/// middleware cannot re-enable tool calls.
///
/// If the wrap-up turn finishes with text, the parent receives
///
/// ```text
/// [partial: sub-agent reached its time budget]
/// <summary>
/// ```
///
/// (`iteration budget` for the iteration trigger) with
/// `is_error: false`. The hard `timeout` stays absolute: if it fires
/// before the wrap-up turn completes — or the child aborts for any
/// other reason — the result is the usual `"sub-agent aborted (…)"`
/// with `is_error: true`. A wrap-up turn that yields no text is also
/// reported with `is_error: true`.
///
/// A completion gate on the child ([`ChatMiddleware::on_turn_end`]
/// returning `Continue`) runs after the wrap-up turn and wins: the
/// extra iteration hits `max_iterations` (or the run keeps going until
/// the hard `timeout`), and the parent gets the abort. Gates on a
/// child with wrap-up should stop asking to continue near the budget.
///
/// ```
/// use std::time::Duration;
/// let config = ailoop::SubAgentConfig::new()
///     .timeout(Duration::from_secs(60))
///     .max_iterations(8)
///     .wrap_up(ailoop::WrapUp::new().time_fraction(0.75));
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct WrapUp {
    /// Fraction of the child's `timeout` after which the next request
    /// becomes the wrap-up turn. Clamped to `[0.0, 1.0]`. Default
    /// `0.8`.
    pub time_fraction: f64,
    /// Text appended to the wrap-up request. Default
    /// [`DEFAULT_WRAP_UP_INSTRUCTION`].
    pub instruction: String,
}

impl Default for WrapUp {
    fn default() -> Self {
        Self {
            time_fraction: 0.8,
            instruction: DEFAULT_WRAP_UP_INSTRUCTION.to_owned(),
        }
    }
}

impl WrapUp {
    /// Default settings: wrap up at 80% of the timeout or on the last
    /// allowed iteration, with [`DEFAULT_WRAP_UP_INSTRUCTION`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the fraction of the timeout after which the wrap-up turn is
    /// forced. See [`Self::time_fraction`].
    pub fn time_fraction(mut self, fraction: f64) -> Self {
        self.time_fraction = fraction;
        self
    }

    /// Replace the instruction appended to the wrap-up request. See
    /// [`Self::instruction`].
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = instruction.into();
        self
    }
}

/// Budgets of the child run, captured in `on_run_started`.
struct WrapUpBudget {
    started: Instant,
    soft_deadline: Option<Duration>,
    max_iterations: usize,
}

#[derive(Default)]
struct WrapUpState {
    budget: Option<WrapUpBudget>,
    iteration: usize,
    /// Which budget forced the wrap-up turn, as the abort it pre-empts.
    triggered: Option<AbortReason>,
}

/// Run-scoped middleware installed by [`SubAgentTool::call`] when
/// [`SubAgentConfig::wrap_up`] is set. See [`WrapUp`].
struct WrapUpMiddleware {
    time_fraction: f64,
    instruction: String,
    state: Arc<StdMutex<WrapUpState>>,
}

#[async_trait::async_trait]
impl ChatMiddleware for WrapUpMiddleware {
    async fn on_run_started(&self, run: &RunStartInfo<'_>) -> ailoop_core::HookAction {
        let fraction = self.time_fraction.clamp(0.0, 1.0);
        self.state.lock().expect("wrap-up state").budget = Some(WrapUpBudget {
            started: Instant::now(),
            soft_deadline: run.config.timeout.map(|t| t.mul_f64(fraction)),
            max_iterations: run.config.max_iterations,
        });
        ailoop_core::HookAction::Continue
    }

    async fn on_chunk(&self, chunk: &StreamChunk) {
        if let StreamChunk::StepStarted { iteration, .. } = chunk {
            self.state.lock().expect("wrap-up state").iteration = *iteration;
        }
    }

    async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
        {
            let mut state = self.state.lock().expect("wrap-up state");
            if state.triggered.is_none() {
                let Some(budget) = &state.budget else { return };
                let trigger = if state.iteration + 1 >= budget.max_iterations {
                    Some(AbortReason::MaxIterations(budget.max_iterations))
                } else {
                    budget
                        .soft_deadline
                        .filter(|soft| budget.started.elapsed() >= *soft)
                        .map(AbortReason::Timeout)
                };
                match trigger {
                    Some(reason) => state.triggered = Some(reason),
                    None => return,
                }
            }
        }

        req.tool_choice = Some(ToolChoice::None);
        let instruction = UserBlock::text(self.instruction.clone());
        match req.messages.last_mut() {
            Some(Message::User { blocks }) => blocks.push(instruction),
            _ => req
                .messages
                .push(Message::user_with_blocks(vec![instruction])),
        }
    }
}

/// Human label for the budget a wrap-up pre-empted.
fn budget_label(reason: &AbortReason) -> &'static str {
    match reason {
        AbortReason::MaxIterations(_) => "iteration",
        _ => "time",
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
/// To keep what the child found when it runs out of budget, opt in to
/// [`SubAgentConfig::wrap_up`]: the child is forced to summarize (no
/// tool calls) before the hard cutoff, and the parent receives the
/// summary prefixed with `[partial: …]` and `is_error: false`. See
/// [`WrapUp`].
///
/// Per-invocation budget overrides (`timeout`, `max_iterations`,
/// `max_tokens`) live on [`SubAgentConfig`] and are applied through
/// [`SubAgentTool::with_config`]. The parent's
/// [`ToolContext::cancellation`] always wins over anything in the
/// config — the cancellation handle is wired from the context, not the
/// config, so a parent abort still cuts a child mid-run regardless of
/// per-call budget.
///
/// # Usage
///
/// Every token the child spends — its own provider turns and those of
/// any sub-agent it calls in turn — is reported to the parent through
/// [`ToolContext::report_usage`] as it is spent, so the parent's
/// [`RunFinished::usage`] / [`RunOutcome::usage`](crate::RunOutcome::usage)
/// include it. That holds when the child aborts (its own `timeout`,
/// `max_iterations`, or a wrap-up) and when the parent aborts while
/// the child is still running: the turns completed before the cutoff
/// count. Don't add the child's usage yourself; it is already in the
/// parent's total.
///
/// # Multimodal input
///
/// The JSON schema accepts an optional `attachments` array alongside
/// `prompt`. Each entry mirrors Anthropic's content shape:
///
/// ```json
/// {
///   "type": "image",
///   "source": {"type": "base64", "media_type": "image/png", "data": "..."}
/// }
/// ```
///
/// `type` selects [`UserBlock::Image`] or [`UserBlock::Document`];
/// `source` accepts `base64`, `url`, or `file_id` (matching [`Source`]).
/// The wrapper combines the text `prompt` (if non-empty) with the
/// parsed attachment blocks into a single [`Message::user_with_blocks`]
/// kickoff. Malformed attachments surface as a `"sub-agent error:
/// invalid attachments: …"` reply with `is_error: true` — never an
/// `Err` to the engine.
///
/// Output stays text-only: the engine's [`AssistantBlock`] surface has
/// no image or document variants today, so the wrapper continues to
/// relay [`RunOutcome::final_text`](crate::RunOutcome::final_text) as
/// the tool result. Multimodal-out would require adding inline-media
/// variants to [`AssistantBlock`] first — tracked as a separate decision
/// in `dev-notes/sub-agent-improvements.md`.
///
/// [`AssistantBlock`]: ailoop_core::AssistantBlock
///
/// [`child_token`]: tokio_util::sync::CancellationToken::child_token
/// [`ToolContext::cancellation`]: ailoop_tools::ToolContext::cancellation
/// [`ToolContext::report_usage`]: ailoop_tools::ToolContext::report_usage
/// [`RunFinished::usage`]: ailoop_core::StreamChunk::RunFinished::usage
///
/// # Examples
///
/// ```no_run
/// # use std::sync::Arc;
/// # async fn build<M>(researcher_model: M, parent_model: M)
/// # -> Result<(), Box<dyn std::error::Error>>
/// # where M: ailoop::CompletionModel + 'static, M::Error: ailoop::ProviderError {
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
/// # where M: ailoop::CompletionModel + 'static, M::Error: ailoop::ProviderError {
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
    M: CompletionModel + 'static,
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
    /// `max_tokens` and the [`WrapUp`] cutoff are configurable
    /// per-invocation.
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
    M: CompletionModel + 'static,
    M::Error: ProviderError,
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
                    },
                    "attachments": {
                        "type": "array",
                        "description": "Optional images or documents to attach to the prompt. Each entry mirrors Anthropic's content shape: {\"type\": \"image\"|\"document\", \"source\": {\"type\": \"base64\", \"media_type\": \"…\", \"data\": \"…\"} | {\"type\": \"url\", \"url\": \"…\"} | {\"type\": \"file_id\", \"id\": \"…\"}}.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["image", "document"]},
                                "source": {
                                    "type": "object",
                                    "oneOf": [
                                        {
                                            "properties": {
                                                "type": {"const": "base64"},
                                                "media_type": {"type": "string"},
                                                "data": {"type": "string"}
                                            },
                                            "required": ["type", "media_type", "data"]
                                        },
                                        {
                                            "properties": {
                                                "type": {"const": "url"},
                                                "url": {"type": "string"}
                                            },
                                            "required": ["type", "url"]
                                        },
                                        {
                                            "properties": {
                                                "type": {"const": "file_id"},
                                                "id": {"type": "string"}
                                            },
                                            "required": ["type", "id"]
                                        }
                                    ]
                                }
                            },
                            "required": ["type", "source"]
                        }
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

        let attachment_blocks: Vec<UserBlock> = match args.get("attachments") {
            None | Some(Value::Null) => Vec::new(),
            Some(v) => match parse_attachments(v) {
                Ok(parsed) => parsed,
                Err(e) => {
                    return ToolResultContent::text(format!(
                        "sub-agent error: invalid attachments: {e}"
                    ))
                    .with_is_error(true);
                }
            },
        };

        let mut options = RunOptions::new().cancellation(ctx.cancellation().child_token());
        // The child's engine forwards every token it spends (its own
        // turns and its nested sub-agents) into our sink as it happens,
        // so the parent's total includes it even on aborts. Reporting
        // `outcome.usage` again here would double count.
        options.usage_parent = Some(ctx.usage_sink().clone());
        if let Some(timeout) = self.config.timeout {
            options = options.timeout(timeout);
        }
        if let Some(max_iterations) = self.config.max_iterations {
            options = options.max_iterations(max_iterations);
        }
        if let Some(max_tokens) = self.config.max_tokens {
            options = options.max_tokens(max_tokens);
        }
        let wrap_up_state = self.config.wrap_up.as_ref().map(|wrap_up| {
            let state = Arc::new(StdMutex::new(WrapUpState::default()));
            options.extra_middlewares.push(Arc::new(WrapUpMiddleware {
                time_fraction: wrap_up.time_fraction,
                instruction: wrap_up.instruction.clone(),
                state: state.clone(),
            }));
            state
        });

        let mut conv = self.conversation.lock().await;
        let run_result = if attachment_blocks.is_empty() {
            // Text-only fast path: preserve the bit-for-bit shape the
            // wrapper has always produced (single `UserBlock::Text`),
            // so existing snapshots and tests stay green.
            conv.run_with_options(prompt, options).await
        } else {
            let mut blocks: Vec<UserBlock> = Vec::with_capacity(attachment_blocks.len() + 1);
            if !prompt.is_empty() {
                blocks.push(UserBlock::text(prompt));
            }
            blocks.extend(attachment_blocks);
            conv.run_with_options(Message::user_with_blocks(blocks), options)
                .await
        };

        let wrapped_up =
            wrap_up_state.and_then(|state| state.lock().expect("wrap-up state").triggered.take());

        match run_result {
            Ok(outcome) => {
                let text = outcome.final_text.unwrap_or_default();
                match (outcome.finish_reason, wrapped_up) {
                    (FinishReason::Aborted(reason), _) if text.is_empty() => {
                        ToolResultContent::text(format!("sub-agent aborted: {reason}"))
                            .with_is_error(true)
                    }
                    (FinishReason::Aborted(reason), _) => {
                        ToolResultContent::text(format!("sub-agent aborted ({reason}): {text}"))
                            .with_is_error(true)
                    }
                    (_, Some(budget)) if text.is_empty() => ToolResultContent::text(format!(
                        "sub-agent aborted: reached its {} budget without producing a summary",
                        budget_label(&budget)
                    ))
                    .with_is_error(true),
                    (_, Some(budget)) => ToolResultContent::text(format!(
                        "[partial: sub-agent reached its {} budget]\n{text}",
                        budget_label(&budget)
                    )),
                    (_, None) => ToolResultContent::text(text),
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
        CancellationToken, ChatMiddleware, ChatRequest, HookAction, Message, RunId, Source, StepId,
        StreamChunk, Usage,
    };
    use ailoop_tools::{ToolActivation, UsageSink};
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
            async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
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
            async fn on_run_started(&self, _run: &RunStartInfo<'_>) -> HookAction {
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
            "toolu_parent",
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
    /// turns. With `max_iterations(0)`, the engine aborts on the first
    /// iteration check with [`AbortReason::MaxIterations`], which the
    /// wrapper surfaces as `"sub-agent aborted: …"` + `is_error: true`.
    ///
    /// [`ConversationBuilder`]: crate::ConversationBuilder
    /// [`AbortReason::MaxIterations`]: ailoop_core::AbortReason::MaxIterations
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
        assert_eq!(
            result.as_text(),
            Some("sub-agent aborted: agent loop exceeded max iterations (0)")
        );
        assert!(
            result.is_error,
            "max_iterations exceeded must mark is_error: true"
        );
    }

    /// A child that is still calling tools when it hits
    /// `max_iterations` hands its partial text to the parent instead of
    /// losing it behind an error.
    #[tokio::test]
    async fn sub_agent_max_iterations_preserves_partial_text() {
        // The tool is not registered: the engine answers with an
        // in-band "not found" result and keeps looping, which is all
        // this test needs to reach the cap.
        let tool_turn = vec![
            StreamChunk::TextDelta {
                delta: "found two candidates so far".into(),
            },
            StreamChunk::ToolCallStarted {
                call_id: "toolu_1".into(),
                name: "lookup".into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_1".into(),
                name: "lookup".into(),
                args: json!({}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
                service_tier: None,
            },
        ];
        let model = ScriptedModel::new([tool_turn]);
        let conv = Conversation::builder(model).build().expect("build");
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            conv,
            SubAgentConfig::new().max_iterations(1),
        );

        let result = tool
            .call(json!({"prompt": "research"}), &ToolContext::detached())
            .await;
        assert_eq!(
            result.as_text(),
            Some(
                "sub-agent aborted (agent loop exceeded max iterations (1)): \
                 found two candidates so far"
            )
        );
        assert!(result.is_error);
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
            async fn on_run_started(&self, _run: &RunStartInfo<'_>) -> HookAction {
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
            async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
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

    /// `SubAgentConfig::max_tokens` beats the child conversation's
    /// builder default.
    #[tokio::test]
    async fn sub_agent_config_max_tokens_overrides_child_builder_default() {
        struct ReqSpy {
            captured: Arc<StdMutex<Option<u32>>>,
        }
        #[async_trait::async_trait]
        impl ChatMiddleware for ReqSpy {
            async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
                *self.captured.lock().unwrap() = Some(req.max_tokens);
            }
        }

        let captured = Arc::new(StdMutex::new(None));
        let model = ScriptedModel::new([one_text_turn("ok")]);
        let conv = Conversation::builder(model)
            .max_tokens(1000)
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
            "SubAgentConfig.max_tokens must win over the child's builder default"
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
            async fn on_run_started(&self, run: &RunStartInfo<'_>) -> HookAction {
                *self.max_iterations.lock().unwrap() = Some(run.config.max_iterations);
                *self.max_tokens.lock().unwrap() = Some(run.config.max_tokens);
                *self.timeout.lock().unwrap() = Some(run.config.timeout);
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
        assert_eq!(*max_iterations.lock().unwrap(), Some(25));
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

    /// Recorder middleware capturing every [`ChatRequest`] the child
    /// sees, so attachment tests can assert the exact block sequence
    /// the engine dispatched.
    struct MessageRecorder {
        captures: Arc<StdMutex<Vec<Vec<Message>>>>,
    }
    #[async_trait::async_trait]
    impl ChatMiddleware for MessageRecorder {
        async fn on_chat_request(&self, _step: &StepInfo, req: &mut ChatRequest) {
            self.captures.lock().unwrap().push(req.messages.clone());
        }
    }

    /// Image attachment lands in the child's `ChatRequest` as a
    /// [`UserBlock::Image`] alongside the prompt text — proves the
    /// `attachments` field parses end-to-end into multimodal user
    /// blocks, not into a stringified placeholder.
    #[tokio::test]
    async fn sub_agent_image_attachment_reaches_chat_request() {
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let model = ScriptedModel::new([one_text_turn("looked at the image")]);
        let conv = Conversation::builder(model)
            .middleware(Arc::new(MessageRecorder {
                captures: captures.clone(),
            }))
            .build()
            .expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let result = tool
            .call(
                json!({
                    "prompt": "what is this?",
                    "attachments": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AAAA"
                            }
                        }
                    ]
                }),
                &ToolContext::detached(),
            )
            .await;
        assert!(
            !result.is_error,
            "happy multimodal path must not flag error"
        );
        assert_eq!(result.as_text(), Some("looked at the image"));

        let captures = captures.lock().unwrap();
        let blocks = captures
            .first()
            .and_then(|msgs| msgs.last())
            .and_then(|m| match m {
                Message::User { blocks } => Some(blocks),
                _ => None,
            })
            .expect("expected a user message in the first ChatRequest");
        assert_eq!(
            blocks.len(),
            2,
            "expected text + image blocks, got {blocks:?}"
        );
        match &blocks[0] {
            UserBlock::Text { text, .. } => assert_eq!(text, "what is this?"),
            other => panic!("expected Text first, got {other:?}"),
        }
        match &blocks[1] {
            UserBlock::Image { source, .. } => assert!(matches!(
                source,
                Source::Base64 { media_type, data }
                    if media_type == "image/png" && data == "AAAA"
            )),
            other => panic!("expected Image second, got {other:?}"),
        }
    }

    /// Document attachment uses [`UserBlock::Document`] (not Image) —
    /// proves the variant tag routes correctly.
    #[tokio::test]
    async fn sub_agent_document_attachment_reaches_chat_request() {
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let model = ScriptedModel::new([one_text_turn("ok")]);
        let conv = Conversation::builder(model)
            .middleware(Arc::new(MessageRecorder {
                captures: captures.clone(),
            }))
            .build()
            .expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let _ = tool
            .call(
                json!({
                    "prompt": "summarize",
                    "attachments": [
                        {
                            "type": "document",
                            "source": {"type": "url", "url": "https://example.com/x.pdf"}
                        }
                    ]
                }),
                &ToolContext::detached(),
            )
            .await;

        let captures = captures.lock().unwrap();
        let blocks = captures
            .first()
            .and_then(|msgs| msgs.last())
            .and_then(|m| match m {
                Message::User { blocks } => Some(blocks),
                _ => None,
            })
            .expect("expected a user message");
        assert!(
            blocks.iter().any(|b| matches!(
                b,
                UserBlock::Document { source: Source::Url { url }, .. } if url == "https://example.com/x.pdf"
            )),
            "expected a Document block with the URL source, got {blocks:?}"
        );
    }

    /// Attachment with an unknown source `type` fails to deserialize
    /// and surfaces as a tool-reported error — not as an engine `Err`,
    /// not as a panic. The error body should mention attachments so
    /// the model can correct itself on the next call.
    #[tokio::test]
    async fn sub_agent_invalid_attachment_surfaces_as_is_error_text() {
        let model = ScriptedModel::new([one_text_turn("never reached")]);
        let conv = Conversation::builder(model).build().expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let result = tool
            .call(
                json!({
                    "prompt": "what?",
                    "attachments": [
                        {"type": "image", "source": {"type": "bogus", "data": "xxx"}}
                    ]
                }),
                &ToolContext::detached(),
            )
            .await;
        let text = result
            .as_text()
            .expect("expected text body on malformed attachment");
        assert!(
            text.starts_with("sub-agent error: invalid attachments:"),
            "expected attachment error prefix, got {text:?}"
        );
        assert!(
            result.is_error,
            "malformed attachments must mark is_error: true"
        );
    }

    /// Attachments-only kickoff (empty `prompt`, one image): the
    /// `attachments` path drops the text block when `prompt` is empty,
    /// so the child receives a single-block user turn — matching the
    /// `Conversation::run(UserBlock::image(...))` ergonomics that
    /// already exist on the public API.
    #[tokio::test]
    async fn sub_agent_attachments_only_omits_empty_prompt_block() {
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let model = ScriptedModel::new([one_text_turn("ok")]);
        let conv = Conversation::builder(model)
            .middleware(Arc::new(MessageRecorder {
                captures: captures.clone(),
            }))
            .build()
            .expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", conv);

        let _ = tool
            .call(
                json!({
                    "prompt": "",
                    "attachments": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AAAA"
                            }
                        }
                    ]
                }),
                &ToolContext::detached(),
            )
            .await;

        let captures = captures.lock().unwrap();
        let blocks = captures
            .first()
            .and_then(|msgs| msgs.last())
            .and_then(|m| match m {
                Message::User { blocks } => Some(blocks),
                _ => None,
            })
            .expect("expected a user message");
        assert_eq!(
            blocks.len(),
            1,
            "expected only the image block, got {blocks:?}"
        );
        assert!(matches!(blocks[0], UserBlock::Image { .. }));
    }

    // ---- WrapUp (graceful cutoff) ----

    /// `ScriptedModel` wrapper that records every request and can hold
    /// a given turn back for a while before answering.
    struct Recording {
        inner: ScriptedModel,
        requests: Arc<StdMutex<Vec<ChatRequest>>>,
        delays: Vec<Duration>,
    }

    impl Recording {
        fn new(turns: Vec<Vec<StreamChunk>>) -> (Self, Arc<StdMutex<Vec<ChatRequest>>>) {
            let requests = Arc::new(StdMutex::new(Vec::new()));
            let model = Self {
                inner: ScriptedModel::new(turns),
                requests: requests.clone(),
                delays: Vec::new(),
            };
            (model, requests)
        }

        /// Delay the answer to the `i`-th request (zero-based).
        fn delay_turn(mut self, i: usize, d: Duration) -> Self {
            if self.delays.len() <= i {
                self.delays.resize(i + 1, Duration::ZERO);
            }
            self.delays[i] = d;
            self
        }
    }

    #[async_trait::async_trait]
    impl CompletionModel for Recording {
        type Error = ScriptedError;

        fn name(&self) -> &str {
            self.inner.name()
        }

        fn model(&self) -> &str {
            self.inner.model()
        }

        async fn chat_stream(
            &self,
            req: ChatRequest,
        ) -> Result<
            futures::stream::BoxStream<'static, Result<StreamChunk, Self::Error>>,
            Self::Error,
        > {
            let i = {
                let mut requests = self.requests.lock().unwrap();
                requests.push(req.clone());
                requests.len() - 1
            };
            if let Some(d) = self.delays.get(i).filter(|d| !d.is_zero()) {
                tokio::time::sleep(*d).await;
            }
            self.inner.chat_stream(req).await
        }
    }

    /// Child tool that takes `delay` to answer.
    struct SlowLookup {
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl ToolDyn for SlowLookup {
        fn name(&self) -> String {
            "lookup".into()
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "lookup",
                "stub",
                json!({"type": "object", "properties": {}, "required": []}),
                vec![],
            )
        }
        async fn call(&self, _: Value, _: &ToolContext) -> ToolResultContent {
            tokio::time::sleep(self.delay).await;
            ToolResultContent::text("lookup result")
        }
    }

    fn lookup_turn() -> Vec<StreamChunk> {
        vec![
            StreamChunk::ToolCallStarted {
                call_id: "toolu_1".into(),
                name: "lookup".into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_1".into(),
                name: "lookup".into(),
                args: json!({}),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage: Usage::default(),
                service_tier: None,
            },
        ]
    }

    fn child(model: Recording, tool_delay: Duration) -> Conversation<Recording> {
        Conversation::builder(model)
            .tool_dyn(Arc::new(SlowLookup { delay: tool_delay }))
            .build()
            .expect("build")
    }

    fn last_user_text(req: &ChatRequest) -> Option<String> {
        match req.messages.last()? {
            Message::User { blocks } => blocks.iter().rev().find_map(|b| match b {
                UserBlock::Text { text, .. } => Some(text.clone()),
                _ => None,
            }),
            _ => None,
        }
    }

    fn history_mentions(messages: &[Message], needle: &str) -> bool {
        messages.iter().any(|m| match m {
            Message::User { blocks } => blocks
                .iter()
                .any(|b| matches!(b, UserBlock::Text { text, .. } if text.contains(needle))),
            _ => false,
        })
    }

    /// On the last allowed iteration the child is asked to summarize
    /// with tool calls forbidden; the summary reaches the parent marked
    /// as partial and not as an error.
    #[tokio::test]
    async fn wrap_up_iteration_budget_forces_synthesis_without_tools() {
        let (model, requests) = Recording::new(vec![lookup_turn(), one_text_turn("summary")]);
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            child(model, Duration::ZERO),
            SubAgentConfig::new()
                .max_iterations(2)
                .wrap_up(WrapUp::new().instruction("WRAP UP NOW")),
        );

        let result = tool
            .call(json!({"prompt": "research"}), &ToolContext::detached())
            .await;

        assert_eq!(
            result.as_text(),
            Some("[partial: sub-agent reached its iteration budget]\nsummary")
        );
        assert!(!result.is_error);

        {
            let requests = requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            assert_eq!(requests[0].tool_choice, None);
            assert_eq!(requests[1].tool_choice, Some(ToolChoice::None));
            assert!(
                requests[1].tools.as_ref().is_some_and(|t| !t.is_empty()),
                "tool definitions stay on the wrap-up request"
            );
            assert_eq!(last_user_text(&requests[1]).as_deref(), Some("WRAP UP NOW"));
        }

        let conv = tool.conversation.lock().await;
        assert!(
            !history_mentions(conv.history_messages(), "WRAP UP NOW"),
            "the wrap-up instruction is request-only"
        );
    }

    /// Crossing `timeout * time_fraction` forces the next request to be
    /// the wrap-up turn.
    #[tokio::test(start_paused = true)]
    async fn wrap_up_soft_deadline_forces_synthesis_without_tools() {
        let (model, requests) = Recording::new(vec![lookup_turn(), one_text_turn("summary")]);
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            child(model, Duration::from_millis(60)),
            SubAgentConfig::new()
                .timeout(Duration::from_millis(100))
                .wrap_up(WrapUp::new().time_fraction(0.5)),
        );

        let result = tool
            .call(json!({"prompt": "research"}), &ToolContext::detached())
            .await;

        assert_eq!(
            result.as_text(),
            Some("[partial: sub-agent reached its time budget]\nsummary")
        );
        assert!(!result.is_error);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].tool_choice, None);
        assert_eq!(requests[1].tool_choice, Some(ToolChoice::None));
        assert_eq!(
            last_user_text(&requests[1]).as_deref(),
            Some(DEFAULT_WRAP_UP_INSTRUCTION)
        );
    }

    /// Without `wrap_up` neither the requests nor the result change —
    /// even on the run's last allowed iteration.
    #[tokio::test]
    async fn without_wrap_up_requests_and_result_are_unchanged() {
        let (model, requests) = Recording::new(vec![lookup_turn(), one_text_turn("answer")]);
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            child(model, Duration::ZERO),
            SubAgentConfig::new().max_iterations(2),
        );

        let result = tool
            .call(json!({"prompt": "research"}), &ToolContext::detached())
            .await;

        assert_eq!(result.as_text(), Some("answer"));
        assert!(!result.is_error);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests.iter().all(|r| r.tool_choice.is_none()));
        assert!(
            !history_mentions(&requests[1].messages, DEFAULT_WRAP_UP_INSTRUCTION),
            "no instruction is injected without wrap_up"
        );
    }

    /// The hard timeout stays absolute: a wrap-up turn that does not
    /// finish in time still ends as an abort with `is_error: true`.
    #[tokio::test(start_paused = true)]
    async fn wrap_up_hard_timeout_wins_when_synthesis_is_slow() {
        let (model, requests) = Recording::new(vec![lookup_turn(), one_text_turn("too late")]);
        let model = model.delay_turn(1, Duration::from_secs(1));
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            child(model, Duration::from_millis(60)),
            SubAgentConfig::new()
                .timeout(Duration::from_millis(100))
                .wrap_up(WrapUp::new().time_fraction(0.5)),
        );

        let result = tool
            .call(json!({"prompt": "research"}), &ToolContext::detached())
            .await;

        assert_eq!(
            result.as_text(),
            Some("sub-agent aborted: timeout exceeded after 100ms")
        );
        assert!(result.is_error);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2, "the wrap-up request was sent");
        assert_eq!(requests[1].tool_choice, Some(ToolChoice::None));
    }

    fn tokens(input: u32, output: u32) -> Usage {
        let mut u = Usage::default();
        u.input_tokens = input;
        u.output_tokens = output;
        u
    }

    fn text_turn_with(text: &str, usage: Usage) -> Vec<StreamChunk> {
        vec![
            StreamChunk::TextDelta { delta: text.into() },
            StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                usage,
                service_tier: None,
            },
        ]
    }

    fn call_turn_with(tool: &str, args: Value, usage: Usage) -> Vec<StreamChunk> {
        vec![
            StreamChunk::ToolCallStarted {
                call_id: "toolu_1".into(),
                name: tool.into(),
            },
            StreamChunk::ToolCallFinished {
                call_id: "toolu_1".into(),
                name: tool.into(),
                args,
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::ToolUse,
                usage,
                service_tier: None,
            },
        ]
    }

    fn io(u: Usage) -> (u32, u32) {
        (u.input_tokens, u.output_tokens)
    }

    /// The parent's run usage includes what the child spent, exactly
    /// once.
    #[tokio::test]
    async fn parent_run_usage_includes_child_usage() {
        let child = Conversation::builder(ScriptedModel::new([text_turn_with(
            "child answer",
            tokens(100, 20),
        )]))
        .build()
        .expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", child);

        let parent_model = ScriptedModel::new([
            call_turn_with("delegate", json!({"prompt": "go"}), tokens(10, 1)),
            text_turn_with("done", tokens(5, 2)),
        ]);
        let mut parent = Conversation::builder(parent_model)
            .tool_dyn(Arc::new(tool))
            .build()
            .expect("build");

        let outcome = parent.run("start").await.expect("run");
        assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
        assert_eq!(io(outcome.usage), (115, 23));
    }

    /// A child aborted by its own timeout still reports what it spent
    /// before the cutoff.
    #[tokio::test]
    async fn child_aborted_by_timeout_reports_partial_usage() {
        let mut first = lookup_turn();
        if let Some(StreamChunk::TurnFinished { usage, .. }) = first.last_mut() {
            *usage = tokens(100, 20);
        }
        let (model, _) = Recording::new(vec![first, text_turn_with("never", tokens(1, 1))]);
        let tool = SubAgentTool::with_config(
            "delegate",
            "delegate",
            child(model, Duration::from_secs(60)),
            SubAgentConfig::new().timeout(Duration::from_millis(50)),
        );

        let sink = UsageSink::new();
        let ctx = ToolContext::detached().with_usage_sink(sink.clone());
        let result = tool.call(json!({"prompt": "research"}), &ctx).await;

        assert!(result.is_error, "child should have timed out");
        assert_eq!(io(sink.total()), (100, 20));
    }

    /// When the parent aborts while the child is still running, the
    /// child's completed turns are in the parent's aborted total.
    #[tokio::test]
    async fn parent_timeout_mid_child_keeps_child_usage() {
        let mut first = lookup_turn();
        if let Some(StreamChunk::TurnFinished { usage, .. }) = first.last_mut() {
            *usage = tokens(100, 20);
        }
        let (model, _) = Recording::new(vec![first, text_turn_with("never", tokens(1, 1))]);
        let tool = SubAgentTool::new(
            "delegate",
            "delegate",
            child(model, Duration::from_secs(60)),
        );

        let parent_model = ScriptedModel::new([
            call_turn_with("delegate", json!({"prompt": "go"}), tokens(10, 1)),
            text_turn_with("never", tokens(5, 2)),
        ]);
        let mut parent = Conversation::builder(parent_model)
            .tool_dyn(Arc::new(tool))
            .build()
            .expect("build");

        let outcome = parent
            .run_with_options(
                "start",
                RunOptions::new().timeout(Duration::from_millis(50)),
            )
            .await
            .expect("run");
        assert!(matches!(
            outcome.finish_reason,
            FinishReason::Aborted(AbortReason::Timeout(_))
        ));
        assert_eq!(io(outcome.usage), (110, 21));
    }

    /// A sub-agent's own sub-agent rolls up to the outermost caller.
    #[tokio::test]
    async fn nested_sub_agent_usage_rolls_up() {
        let grandchild = Conversation::builder(ScriptedModel::new([text_turn_with(
            "leaf",
            tokens(1000, 300),
        )]))
        .build()
        .expect("build");
        let child = Conversation::builder(ScriptedModel::new([
            call_turn_with("deeper", json!({"prompt": "dig"}), tokens(100, 20)),
            text_turn_with("mid", tokens(50, 10)),
        ]))
        .tool_dyn(Arc::new(SubAgentTool::new("deeper", "deeper", grandchild)))
        .build()
        .expect("build");
        let tool = SubAgentTool::new("delegate", "delegate", child);

        let sink = UsageSink::new();
        let ctx = ToolContext::detached().with_usage_sink(sink.clone());
        let result = tool.call(json!({"prompt": "go"}), &ctx).await;

        assert_eq!(result.as_text(), Some("mid"));
        assert_eq!(io(sink.total()), (1150, 330));
    }
}
