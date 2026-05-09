//! Integration tests for `Conversation::run` — the non-streaming
//! convenience built on top of `Conversation::stream`. These tests pin
//! the contract documented on `RunOutcome`: aborts surface as a finish
//! reason (not an error), `final_text` is concatenated from the last
//! assistant turn, and history is extended exactly once.

use std::sync::Arc;

use ailoop::{Conversation, Message, ToolDefinition, ToolResultContent};
use ailoop_core::testing::ScriptedModel;
use ailoop_core::{
    AssistantBlock, ChatMiddleware, FinishReason, HookAction, RunConfig, RunId, StreamChunk,
    ToolTag, Usage,
};
use ailoop_tools::registry::ToolDyn;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Happy-path: a single text turn returns a `RunOutcome` whose
/// `final_text` matches the deltas the model emitted, and history
/// reflects the assistant turn.
#[tokio::test]
async fn run_returns_final_text_for_text_only_turn() {
    let model = ScriptedModel::new([vec![
        StreamChunk::TextDelta {
            delta: "hello ".into(),
        },
        StreamChunk::TextDelta {
            delta: "world".into(),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage {
                input_tokens: 5,
                output_tokens: 7,
                ..Default::default()
            },
        },
    ]]);

    let mut chat = Conversation::builder(model).build().expect("build");

    let outcome = chat.run("hi").await.expect("run should succeed");

    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(outcome.final_text.as_deref(), Some("hello world"));
    assert_eq!(outcome.usage.input_tokens, 5);
    assert_eq!(outcome.usage.output_tokens, 7);

    let assistant_text: Option<String> = chat.history().iter().rev().find_map(|m| match m {
        Message::Assistant { blocks } => {
            let mut s = String::new();
            for b in blocks {
                if let AssistantBlock::Text(t) = b {
                    s.push_str(t);
                }
            }
            Some(s)
        }
        _ => None,
    });
    assert_eq!(assistant_text.as_deref(), Some("hello world"));
}

struct GetWeather;

#[async_trait]
impl ToolDyn for GetWeather {
    fn name(&self) -> String {
        "get_weather".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".into(),
            description: "stub".into(),
            input_schema: json!({"type":"object","properties":{},"required":[]}),
            tags: vec![ToolTag::ReadOnly],
        }
    }
    async fn call(&self, _: Value) -> ToolResultContent {
        ToolResultContent::Text("sunny".into())
    }
}

/// `final_text` walks back to the most recent assistant message and
/// joins only its `Text` blocks. A turn that started with a tool call
/// and finished with a text reply must surface the text reply.
#[tokio::test]
async fn run_final_text_reflects_last_assistant_turn_only() {
    let turn1 = vec![
        StreamChunk::ToolCallStart {
            id: "toolu_1".into(),
            name: "get_weather".into(),
        },
        StreamChunk::ToolCallEnd {
            id: "toolu_1".into(),
            name: "get_weather".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
        },
    ];
    let turn2 = vec![
        StreamChunk::TextDelta {
            delta: "it is sunny".into(),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
        },
    ];
    let model = ScriptedModel::new([turn1, turn2]);

    let mut chat = Conversation::builder(model)
        .tool(GetWeather)
        .build()
        .expect("build");

    let outcome = chat.run("what's the weather?").await.expect("run");

    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
    assert_eq!(outcome.final_text.as_deref(), Some("it is sunny"));
}

/// Aborted runs surface as `Ok(RunOutcome { finish_reason: Aborted(_), .. })`,
/// not as `Err`. `final_text` is `None` because no assistant text was
/// produced before the abort.
#[tokio::test]
async fn run_returns_ok_aborted_when_hook_terminates() {
    struct AbortingMw;

    #[async_trait]
    impl ChatMiddleware for AbortingMw {
        async fn on_run_start(
            &self,
            _run_id: &RunId,
            _messages: &[Message],
            _config: &RunConfig,
        ) -> HookAction {
            HookAction::Terminate {
                reason: "policy".into(),
            }
        }
    }

    let model = ScriptedModel::new(Vec::<Vec<StreamChunk>>::new());
    let mut chat = Conversation::builder(model)
        .middleware(Arc::new(AbortingMw))
        .build()
        .expect("build");

    let outcome = chat.run("hi").await.expect("aborted run is not Err");

    match &outcome.finish_reason {
        FinishReason::Aborted(reason) => assert_eq!(reason, "policy"),
        other => panic!("expected Aborted, got {other:?}"),
    }
    assert!(outcome.final_text.is_none());
}

/// Two consecutive runs extend history exactly once each — the second
/// run sees the first run's user input + assistant reply already in
/// place, so the total ends up at 2 user + 2 assistant messages with
/// no duplication.
#[tokio::test]
async fn run_extends_history_exactly_once_per_call() {
    let model = ScriptedModel::new([
        vec![
            StreamChunk::TextDelta {
                delta: "first".into(),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                usage: Usage::default(),
            },
        ],
        vec![
            StreamChunk::TextDelta {
                delta: "second".into(),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                usage: Usage::default(),
            },
        ],
    ]);

    let mut chat = Conversation::builder(model).build().expect("build");

    let _ = chat.run("a").await.expect("first run");
    let outcome = chat.run("b").await.expect("second run");

    assert_eq!(outcome.final_text.as_deref(), Some("second"));

    let history = chat.history();
    let user_count = history
        .iter()
        .filter(|m| matches!(m, Message::User { .. }))
        .count();
    let assistant_count = history
        .iter()
        .filter(|m| matches!(m, Message::Assistant { .. }))
        .count();
    assert_eq!(user_count, 2, "expected 2 user messages, got {history:?}");
    assert_eq!(
        assistant_count, 2,
        "expected 2 assistant messages, got {history:?}"
    );
}
