//! `SubAgentTool` — wrap a [`Conversation`] so a parent agent can
//! delegate to it as a regular tool. Pure composition: nothing in the
//! engine or registry changes.
//!
//! The sub-agent's history persists across calls — each invocation sees
//! prior turns. For stateless behavior reconstruct the `SubAgentTool`
//! (or its inner `Conversation`) per call.

use ailoop_core::{CompletionModel, FinishReason, ToolDefinition, ToolResultContent};
use ailoop_tools::ToolDyn;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::Conversation;

/// Wraps a [`Conversation`] so a parent agent can delegate to it as a
/// regular tool. Pure composition: nothing in the engine or registry
/// changes — the child runs on its own [`CompletionModel`], history,
/// and middleware chain.
///
/// The child's history persists across calls (each invocation sees
/// prior turns); rebuild the `SubAgentTool` per call if you need
/// stateless behavior. Child errors and aborts are surfaced as
/// [`ToolResultContent::Text`] (with an `"sub-agent error: …"` /
/// `"sub-agent aborted: …"` prefix) — never as
/// [`ToolResultContent::Error`] and never as a tool-registry error
/// — so the parent's loop continues and the model can react.
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

    async fn call(&self, args: Value) -> ToolResultContent {
        let prompt = args
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();

        let mut conv = self.conversation.lock().await;
        match conv.run(prompt).await {
            Ok(outcome) => {
                let text = outcome.final_text.unwrap_or_default();
                match outcome.finish_reason {
                    FinishReason::Aborted(reason) if text.is_empty() => {
                        ToolResultContent::Text(format!("sub-agent aborted: {reason}"))
                    }
                    FinishReason::Aborted(reason) => {
                        ToolResultContent::Text(format!("sub-agent aborted ({reason}): {text}"))
                    }
                    _ => ToolResultContent::Text(text),
                }
            }
            Err(e) => ToolResultContent::Text(format!("sub-agent error: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::testing::ScriptedModel;
    use ailoop_core::{
        ChatMiddleware, ChatRequest, HookAction, Message, RunConfig, RunId, StepId, StreamChunk,
        Usage,
    };
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

        let result = tool.call(json!({"prompt": "do the thing"})).await;
        match result {
            ToolResultContent::Text(t) => assert_eq!(t, "delegated answer"),
            other => panic!("expected Text, got {other:?}"),
        }
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

        let first = tool.call(json!({"prompt": "P1"})).await;
        let second = tool.call(json!({"prompt": "P2"})).await;

        assert!(matches!(first, ToolResultContent::Text(ref t) if t == "first"));
        assert!(matches!(second, ToolResultContent::Text(ref t) if t == "second"));

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
    /// with a meaningful message — never as `Err`.
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

        let result = tool.call(json!({"prompt": "anything"})).await;
        match result {
            ToolResultContent::Text(t) => {
                assert!(
                    t.contains("aborted") && t.contains("policy"),
                    "expected abort reason in text, got {t:?}"
                );
            }
            other => panic!("expected Text on abort, got {other:?}"),
        }
    }
}
