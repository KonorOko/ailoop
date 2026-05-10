//! Integration tests for `Sanitize`. Each test drives `run_chat` /
//! `Conversation::run` end-to-end so the assertions exercise the real
//! middleware chain (including the `_mut` ordering contract Sanitize
//! relies on) instead of just the trait methods in isolation.

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use ailoop::{Conversation, Message, Sanitize, ToolDefinition, ToolResultContent, advanced::run_chat};
use ailoop_core::testing::ScriptedModel;
use ailoop_core::{
    AssistantBlock, ChatMiddleware, ChatRequest, FinishReason, RunConfig, RunId, StepId,
    StreamChunk, Usage, UserBlock,
};
use ailoop_tools::{ToolRegistry, registry::ToolDyn};
use futures::StreamExt;
use serde_json::{Value, json};

/// Captures `req.messages` from `on_chat_request`. Registered after
/// `Sanitize` so the snapshot reflects the post-sanitization wire copy.
#[derive(Default)]
struct RequestRecorder {
    captures: Mutex<Vec<Vec<Message>>>,
}

#[async_trait::async_trait]
impl ChatMiddleware for RequestRecorder {
    async fn on_chat_request(&self, _: &RunId, _: &StepId, req: &mut ChatRequest) {
        self.captures.lock().unwrap().push(req.messages.clone());
    }
}

/// Echoes its `args` back as JSON so tests can confirm a `_mut`
/// rewrite reached the tool body.
struct EchoArgs;

#[async_trait::async_trait]
impl ToolDyn for EchoArgs {
    fn name(&self) -> String {
        "echo".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "echo",
            "stub",
            json!({"type":"object","properties":{},"required":[]}),
            vec![],
        )
    }
    async fn call(&self, args: Value) -> ToolResultContent {
        ToolResultContent::Text(args.to_string())
    }
}

/// Returns a fixed `Text("alice token: secret-123")` so tests can
/// assert that `on_tool_result` rewrites land on what the engine sees.
struct LeakySecret;

#[async_trait::async_trait]
impl ToolDyn for LeakySecret {
    fn name(&self) -> String {
        "leaky".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "leaky",
            "stub",
            json!({"type":"object","properties":{},"required":[]}),
            vec![],
        )
    }
    async fn call(&self, _: Value) -> ToolResultContent {
        ToolResultContent::Text("alice token: secret-123".into())
    }
}

/// `on_user_text` rewrites every `UserBlock::Text` in the outgoing
/// `ChatRequest`. Sanitize is registered first, then a recorder that
/// captures the post-sanitization message vector.
#[tokio::test]
async fn on_user_text_rewrites_outgoing_user_blocks() {
    let recorder = Arc::new(RequestRecorder::default());
    let mut conv = Conversation::builder(one_turn_model())
        .middleware(Arc::new(
            Sanitize::new().on_user_text(|s| Cow::Owned(s.replace("alice", "<REDACTED>"))),
        ))
        .middleware(recorder.clone())
        .build()
        .expect("builder should succeed");

    conv.run("hello alice").await.expect("run should succeed");

    let captures = recorder.captures.lock().unwrap();
    assert_eq!(captures.len(), 1, "exactly one ChatRequest captured");
    let blocks = match &captures[0][0] {
        Message::User { blocks } => blocks,
        other => panic!("expected first message to be User, got {other:?}"),
    };
    match &blocks[0] {
        UserBlock::Text { text, .. } => assert_eq!(text, "hello <REDACTED>"),
        other => panic!("expected UserBlock::Text, got {other:?}"),
    }
}

/// `on_tool_args` rewrites the JSON `args` value before the tool runs.
/// The `EchoArgs` tool returns the args it actually received as JSON,
/// so the next-turn `ToolResult` chunk is the witness.
#[tokio::test]
async fn on_tool_args_rewrites_args_before_tool_invocation() {
    let turn1 = vec![
        StreamChunk::ToolCallStart {
            id: "toolu_1".into(),
            name: "echo".into(),
        },
        StreamChunk::ToolCallEnd {
            id: "toolu_1".into(),
            name: "echo".into(),
            args: json!({"q": "hello"}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let turn2 = vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }];
    let model = ScriptedModel::new([turn1, turn2]);

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoArgs)).unwrap();

    let sanitize = Sanitize::new().on_tool_args(|_name, args| {
        if let Some(obj) = args.as_object_mut() {
            obj.insert("secret".into(), Value::String("<REDACTED>".into()));
        }
    });
    let mut config = RunConfig::default();
    config.middlewares = vec![Arc::new(sanitize) as Arc<dyn ChatMiddleware>];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|c| c.ok())
        .collect();

    let echoed = chunks
        .iter()
        .find_map(|c| match c {
            StreamChunk::ToolResult {
                content: ToolResultContent::Text(t),
                ..
            } => Some(t.clone()),
            _ => None,
        })
        .expect("ToolResult should be emitted");
    let parsed: Value = serde_json::from_str(&echoed).expect("tool echoed valid JSON");
    assert_eq!(parsed, json!({"q": "hello", "secret": "<REDACTED>"}));
}

/// `on_tool_result` rewrites the result before the model sees it on the
/// next turn. The witness is the `Message::User { ToolResult }` block
/// that the second `ChatRequest` carries — that is what the model gets
/// fed for the follow-up turn.
#[tokio::test]
async fn on_tool_result_rewrites_result_before_next_turn() {
    let turn1 = vec![
        StreamChunk::ToolCallStart {
            id: "toolu_1".into(),
            name: "leaky".into(),
        },
        StreamChunk::ToolCallEnd {
            id: "toolu_1".into(),
            name: "leaky".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let turn2 = vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }];

    let recorder = Arc::new(RequestRecorder::default());
    let sanitize = Sanitize::new().on_tool_result(|_name, result| {
        if let ToolResultContent::Text(t) = result {
            *t = t.replace("secret-123", "<REDACTED>");
        }
    });

    let mut conv = Conversation::builder(ScriptedModel::new([turn1, turn2]))
        .tool(LeakySecret)
        .middleware(Arc::new(sanitize))
        .middleware(recorder.clone())
        .build()
        .expect("builder should succeed");

    conv.run("hi").await.expect("run should succeed");

    let captures = recorder.captures.lock().unwrap();
    assert_eq!(captures.len(), 2, "two ChatRequests across the tool turn");
    let second_turn = &captures[1];
    let tool_result_text = second_turn
        .iter()
        .filter_map(|m| match m {
            Message::User { blocks } => Some(blocks),
            _ => None,
        })
        .flat_map(|blocks| blocks.iter())
        .find_map(|b| match b {
            UserBlock::ToolResult {
                content: ToolResultContent::Text(t),
                ..
            } => Some(t.clone()),
            _ => None,
        })
        .expect("second turn should carry the tool result");
    assert_eq!(tool_result_text, "alice token: <REDACTED>");
}

/// Multiple callbacks on the same surface accumulate and run in
/// registration order: A→B then B→C produces C, not B.
#[tokio::test]
async fn multiple_user_text_callbacks_run_in_registration_order() {
    let recorder = Arc::new(RequestRecorder::default());
    let sanitize = Sanitize::new()
        .on_user_text(|s| Cow::Owned(s.replace('A', "B")))
        .on_user_text(|s| Cow::Owned(s.replace('B', "C")));

    let mut conv = Conversation::builder(one_turn_model())
        .middleware(Arc::new(sanitize))
        .middleware(recorder.clone())
        .build()
        .expect("builder should succeed");

    conv.run("A").await.expect("run should succeed");

    let captures = recorder.captures.lock().unwrap();
    let blocks = match &captures[0][0] {
        Message::User { blocks } => blocks,
        other => panic!("expected User, got {other:?}"),
    };
    match &blocks[0] {
        UserBlock::Text { text, .. } => assert_eq!(text, "C"),
        other => panic!("expected Text, got {other:?}"),
    }
}

/// `on_assistant_text` is a no-op unless `enable_assistant_text` is
/// also called. The recorder captures the second turn's request, which
/// must carry the assistant's original "alice replied" text intact.
#[tokio::test]
async fn on_assistant_text_is_off_by_default() {
    let turn1 = vec![
        StreamChunk::TextDelta {
            delta: "alice replied".into(),
        },
        StreamChunk::ToolCallStart {
            id: "toolu_1".into(),
            name: "echo".into(),
        },
        StreamChunk::ToolCallEnd {
            id: "toolu_1".into(),
            name: "echo".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let turn2 = vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }];

    let recorder = Arc::new(RequestRecorder::default());
    // Register the rewriter but DO NOT call `enable_assistant_text`.
    let sanitize =
        Sanitize::new().on_assistant_text(|s| Cow::Owned(s.replace("alice", "<REDACTED>")));

    let mut conv = Conversation::builder(ScriptedModel::new([turn1, turn2]))
        .tool(EchoArgs)
        .middleware(Arc::new(sanitize))
        .middleware(recorder.clone())
        .build()
        .expect("builder should succeed");

    conv.run("hi").await.expect("run should succeed");

    let captures = recorder.captures.lock().unwrap();
    let second_turn = &captures[1];
    let assistant_text = second_turn
        .iter()
        .filter_map(|m| match m {
            Message::Assistant { blocks } => Some(blocks),
            _ => None,
        })
        .flat_map(|blocks| blocks.iter())
        .find_map(|b| match b {
            AssistantBlock::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .expect("second turn should carry the prior assistant text");
    assert_eq!(
        assistant_text, "alice replied",
        "assistant text must be untouched without enable_assistant_text"
    );
}

/// With `enable_assistant_text`, the same setup as the previous test
/// rewrites the assistant block on the wire.
#[tokio::test]
async fn enable_assistant_text_opts_in_to_assistant_rewrites() {
    let turn1 = vec![
        StreamChunk::TextDelta {
            delta: "alice replied".into(),
        },
        StreamChunk::ToolCallStart {
            id: "toolu_1".into(),
            name: "echo".into(),
        },
        StreamChunk::ToolCallEnd {
            id: "toolu_1".into(),
            name: "echo".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let turn2 = vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }];

    let recorder = Arc::new(RequestRecorder::default());
    let sanitize = Sanitize::new()
        .on_assistant_text(|s| Cow::Owned(s.replace("alice", "<REDACTED>")))
        .enable_assistant_text();

    let mut conv = Conversation::builder(ScriptedModel::new([turn1, turn2]))
        .tool(EchoArgs)
        .middleware(Arc::new(sanitize))
        .middleware(recorder.clone())
        .build()
        .expect("builder should succeed");

    conv.run("hi").await.expect("run should succeed");

    let captures = recorder.captures.lock().unwrap();
    let second_turn = &captures[1];
    let assistant_text = second_turn
        .iter()
        .filter_map(|m| match m {
            Message::Assistant { blocks } => Some(blocks),
            _ => None,
        })
        .flat_map(|blocks| blocks.iter())
        .find_map(|b| match b {
            AssistantBlock::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .expect("second turn should carry the prior assistant text");
    assert_eq!(assistant_text, "<REDACTED> replied");
}

/// `Reasoning` blocks are not part of any Sanitize surface — they
/// round-trip untouched even with every text rewriter active. Callers
/// who need to scrub reasoning can do it through their own
/// `on_chat_request` middleware.
#[tokio::test]
async fn reasoning_blocks_are_not_sanitized() {
    let turn1 = vec![
        StreamChunk::ReasoningDelta {
            delta: "alice thinking".into(),
        },
        StreamChunk::ReasoningEnd {
            signature: Some("sig-1".into()),
        },
        StreamChunk::ToolCallStart {
            id: "toolu_1".into(),
            name: "echo".into(),
        },
        StreamChunk::ToolCallEnd {
            id: "toolu_1".into(),
            name: "echo".into(),
            args: json!({}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let turn2 = vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }];

    let recorder = Arc::new(RequestRecorder::default());
    // Both text rewriters active and assistant text enabled — none of
    // these should touch a `Reasoning` block.
    let sanitize = Sanitize::new()
        .on_user_text(|s| Cow::Owned(s.replace("alice", "<REDACTED>")))
        .on_assistant_text(|s| Cow::Owned(s.replace("alice", "<REDACTED>")))
        .enable_assistant_text();

    let mut conv = Conversation::builder(ScriptedModel::new([turn1, turn2]))
        .tool(EchoArgs)
        .middleware(Arc::new(sanitize))
        .middleware(recorder.clone())
        .build()
        .expect("builder should succeed");

    conv.run("hi").await.expect("run should succeed");

    let captures = recorder.captures.lock().unwrap();
    let second_turn = &captures[1];
    let reasoning_text = second_turn
        .iter()
        .filter_map(|m| match m {
            Message::Assistant { blocks } => Some(blocks),
            _ => None,
        })
        .flat_map(|blocks| blocks.iter())
        .find_map(|b| match b {
            AssistantBlock::Reasoning { text, .. } => Some(text.clone()),
            _ => None,
        })
        .expect("second turn should carry the reasoning block");
    assert_eq!(reasoning_text, "alice thinking");
}

/// `on_tool_args` callbacks can scope themselves to specific tool names
/// by matching on the `name: &str` parameter. A call to a different
/// tool passes through unchanged.
#[tokio::test]
async fn tool_args_callback_can_filter_by_name() {
    /// Two tools wired side by side: `fetch` should be sanitized,
    /// `other_tool` should not.
    struct OtherTool;

    #[async_trait::async_trait]
    impl ToolDyn for OtherTool {
        fn name(&self) -> String {
            "other_tool".into()
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "other_tool",
                "stub",
                json!({"type":"object","properties":{},"required":[]}),
                vec![],
            )
        }
        async fn call(&self, args: Value) -> ToolResultContent {
            ToolResultContent::Text(args.to_string())
        }
    }

    /// Renamed echo so both tools coexist cleanly.
    struct Fetch;

    #[async_trait::async_trait]
    impl ToolDyn for Fetch {
        fn name(&self) -> String {
            "fetch".into()
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "fetch",
                "stub",
                json!({"type":"object","properties":{},"required":[]}),
                vec![],
            )
        }
        async fn call(&self, args: Value) -> ToolResultContent {
            ToolResultContent::Text(args.to_string())
        }
    }

    let turn1 = vec![
        StreamChunk::ToolCallStart {
            id: "toolu_1".into(),
            name: "fetch".into(),
        },
        StreamChunk::ToolCallEnd {
            id: "toolu_1".into(),
            name: "fetch".into(),
            args: json!({"q": "hello"}),
        },
        StreamChunk::ToolCallStart {
            id: "toolu_2".into(),
            name: "other_tool".into(),
        },
        StreamChunk::ToolCallEnd {
            id: "toolu_2".into(),
            name: "other_tool".into(),
            args: json!({"q": "hello"}),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage: Usage::default(),
            service_tier: None,
        },
    ];
    let turn2 = vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }];

    let model = ScriptedModel::new([turn1, turn2]);
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Fetch)).unwrap();
    registry.register(Arc::new(OtherTool)).unwrap();

    let sanitize = Sanitize::new().on_tool_args(|name, args| {
        if name != "fetch" {
            return;
        }
        if let Some(obj) = args.as_object_mut() {
            obj.insert("scoped".into(), Value::Bool(true));
        }
    });
    let mut config = RunConfig::default();
    config.middlewares = vec![Arc::new(sanitize) as Arc<dyn ChatMiddleware>];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|c| c.ok())
        .collect();

    let mut by_call_id: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    for c in &chunks {
        if let StreamChunk::ToolResult {
            call_id,
            content: ToolResultContent::Text(t),
            ..
        } = c
        {
            by_call_id.insert(call_id.clone(), serde_json::from_str(t).unwrap());
        }
    }

    assert_eq!(
        by_call_id["toolu_1"],
        json!({"q": "hello", "scoped": true}),
        "fetch must see the scoped sanitizer's mutation"
    );
    assert_eq!(
        by_call_id["toolu_2"],
        json!({"q": "hello"}),
        "other_tool must pass through untouched"
    );
}

fn one_turn_model() -> ScriptedModel {
    ScriptedModel::new([vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }]])
}
