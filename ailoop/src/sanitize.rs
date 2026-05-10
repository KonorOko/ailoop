use std::borrow::Cow;
use std::sync::Arc;

use ailoop_core::{
    AssistantBlock, ChatMiddleware, ChatRequest, Message, RunId, StepId, ToolResultContent,
    UserBlock,
};
use serde_json::Value;

/// Rewriter applied to a piece of text. Receives the current value and
/// returns either `Cow::Borrowed(input)` (no change, zero allocation) or
/// `Cow::Owned(new)` (replace). The HRTB on the function signature lets
/// callers return a borrow whose lifetime is tied to the input slice
/// without naming it explicitly.
pub type TextRewriter = Arc<dyn for<'a> Fn(&'a str) -> Cow<'a, str> + Send + Sync>;

/// Rewriter applied to a tool's `args` value. Receives the tool name so
/// the callback can dispatch on it (`match name { "fetch" => ..., _ => ()}`)
/// instead of forcing a per-tool registry.
pub type ToolArgsRewriter = Arc<dyn Fn(&str, &mut Value) + Send + Sync>;

/// Rewriter applied to a tool's result before the model sees it.
/// Receives the tool name; mutate `result` in place to redact / scrub /
/// truncate.
pub type ToolResultRewriter = Arc<dyn Fn(&str, &mut ToolResultContent) + Send + Sync>;

/// Middleware that runs caller-supplied transformations at the points
/// where text crosses the model boundary: user-message text and tool
/// args on the way out, tool results on the way back in.
///
/// No regex engine is bundled — callers provide closures and pick their
/// own substitution strategy (`str::replace`, an external `regex::Regex`,
/// a remote redaction service wrapped in `block_in_place`, etc.).
///
/// All callbacks run synchronously in registration order; each receives
/// the value produced by the previous one. Callbacks are accumulated by
/// successive builder calls — registering two `on_user_text` rewriters
/// runs both, in order.
///
/// Surfaces:
///
/// - **`on_user_text`** rewrites every `UserBlock::Text.text` in the
///   outgoing `ChatRequest::messages`. The conversation's persisted
///   history is untouched — only the wire copy that goes to the provider
///   sees the rewrite.
/// - **`on_assistant_text`** rewrites every `AssistantBlock::Text.text`
///   in the outgoing request. **Off by default**: the model already
///   emitted that text and rewriting it for replay can desync the model
///   from its own prior reasoning. Call [`enable_assistant_text`] to opt
///   in when you know what you are doing.
/// - **`on_tool_args`** rewrites the JSON `args` value via the
///   `on_before_tool_call_mut` hook, before any gating decision and
///   before the tool itself runs.
/// - **`on_tool_result`** rewrites the `ToolResultContent` via the
///   `on_after_tool_call_mut` hook, before any observer sees the result
///   and before the model sees it on the next turn.
///
/// `AssistantBlock::ToolCall.args` is intentionally not rewritten on the
/// outbound side — those are the args the model itself just produced,
/// and the tool that consumes them is already covered by `on_tool_args`.
/// `Reasoning` / `RedactedReasoning` blocks are also untouched.
///
/// [`enable_assistant_text`]: Self::enable_assistant_text
pub struct Sanitize {
    user_text: Vec<TextRewriter>,
    assistant_text: Vec<TextRewriter>,
    assistant_text_enabled: bool,
    tool_args: Vec<ToolArgsRewriter>,
    tool_result: Vec<ToolResultRewriter>,
}

impl Sanitize {
    /// Build an empty middleware. Chain
    /// [`on_user_text`](Self::on_user_text),
    /// [`on_assistant_text`](Self::on_assistant_text) +
    /// [`enable_assistant_text`](Self::enable_assistant_text),
    /// [`on_tool_args`](Self::on_tool_args), and
    /// [`on_tool_result`](Self::on_tool_result) onto the result to
    /// register rewriters. Equivalent to [`Sanitize::default`].
    pub fn new() -> Self {
        Self {
            user_text: Vec::new(),
            assistant_text: Vec::new(),
            assistant_text_enabled: false,
            tool_args: Vec::new(),
            tool_result: Vec::new(),
        }
    }

    /// Register a rewriter for `UserBlock::Text` blocks in outgoing
    /// chat requests. Multiple calls accumulate; rewriters run in
    /// registration order, each receiving the previous one's output.
    pub fn on_user_text<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&'a str) -> Cow<'a, str> + Send + Sync + 'static,
    {
        self.user_text.push(Arc::new(f));
        self
    }

    /// Register a rewriter for `AssistantBlock::Text` blocks in outgoing
    /// chat requests. **Has no effect unless [`enable_assistant_text`]
    /// is also called** — opt-in because rewriting prior assistant
    /// utterances can break the model's coherence with its own history.
    ///
    /// [`enable_assistant_text`]: Self::enable_assistant_text
    pub fn on_assistant_text<F>(mut self, f: F) -> Self
    where
        F: for<'a> Fn(&'a str) -> Cow<'a, str> + Send + Sync + 'static,
    {
        self.assistant_text.push(Arc::new(f));
        self
    }

    /// Opt in to applying `on_assistant_text` rewriters. Off by default
    /// because rewriting text the model has already emitted desyncs it
    /// from its own prior reasoning on the next turn.
    pub fn enable_assistant_text(mut self) -> Self {
        self.assistant_text_enabled = true;
        self
    }

    /// Register a rewriter for tool call arguments. The callback runs
    /// in `on_before_tool_call_mut` for every tool invocation; filter on
    /// `name` inside the closure (`match name { "fetch" => ..., _ => () }`)
    /// to scope it.
    pub fn on_tool_args<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &mut Value) + Send + Sync + 'static,
    {
        self.tool_args.push(Arc::new(f));
        self
    }

    /// Register a rewriter for tool results. The callback runs in
    /// `on_after_tool_call_mut`, before any observer sees the result and
    /// before the model sees it on the next turn. Both
    /// `ToolResultContent::Text` and `ToolResultContent::Error` reach
    /// the callback — match on the variant inside the closure if you
    /// only want to scrub one.
    pub fn on_tool_result<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &mut ToolResultContent) + Send + Sync + 'static,
    {
        self.tool_result.push(Arc::new(f));
        self
    }

    fn apply_text(rewriters: &[TextRewriter], text: &mut String) {
        for r in rewriters {
            let rewritten = r(text.as_str());
            if let Cow::Owned(s) = rewritten {
                *text = s;
            }
        }
    }
}

impl Default for Sanitize {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for Sanitize {
    async fn on_chat_request(&self, _: &RunId, _: &StepId, req: &mut ChatRequest) {
        let user_active = !self.user_text.is_empty();
        let assistant_active = self.assistant_text_enabled && !self.assistant_text.is_empty();
        if !user_active && !assistant_active {
            return;
        }
        for msg in req.messages.iter_mut() {
            match msg {
                Message::User { blocks } if user_active => {
                    for block in blocks.iter_mut() {
                        if let UserBlock::Text { text, .. } = block {
                            Self::apply_text(&self.user_text, text);
                        }
                    }
                }
                Message::Assistant { blocks } if assistant_active => {
                    for block in blocks.iter_mut() {
                        if let AssistantBlock::Text { text, .. } = block {
                            Self::apply_text(&self.assistant_text, text);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    async fn on_before_tool_call_mut(&self, _: &RunId, _: &StepId, name: &str, args: &mut Value) {
        for r in &self.tool_args {
            r(name, args);
        }
    }

    async fn on_after_tool_call_mut(
        &self,
        _: &RunId,
        _: &StepId,
        name: &str,
        _args: &Value,
        result: &mut ToolResultContent,
    ) {
        for r in &self.tool_result {
            r(name, result);
        }
    }
}
