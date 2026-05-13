//! `SubAgentTool` — wrap a [`Conversation`] so a parent agent can
//! delegate to it as a regular tool. Pure composition: nothing in the
//! engine or registry changes.
//!
//! The sub-agent's history persists across calls — each invocation sees
//! prior turns. For stateless behavior reconstruct the `SubAgentTool`
//! (or its inner `Conversation`) per call.

use ailoop_core::{CompletionModel, FinishReason, ToolDefinition, ToolResultContent};
use ailoop_tools::{ToolContext, ToolDyn};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{Conversation, RunOptions};

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
pub struct SubAgentTool<M: CompletionModel> {
    name: String,
    description: String,
    conversation: Mutex<Conversation<M>>,
}

impl<M> SubAgentTool<M>
where
    M: CompletionModel + Send + Sync + 'static,
{
    /// Wrap `conversation` as a tool exposing `name` /
    /// `description` to the parent's [`CompletionModel`]. Use
    /// [`Arc::new`](std::sync::Arc::new) when registering through
    /// [`tool_dyn`](crate::ConversationBuilder::tool_dyn).
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        conversation: Conversation<M>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            conversation: Mutex::new(conversation),
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

        let options = RunOptions::new().cancellation(ctx.cancellation().child_token());

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
