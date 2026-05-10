//! Integration tests for the `_mut` middleware hooks. The unit tests
//! in `tracing_middleware.rs` and the lifecycle test in
//! `middleware_lifecycle.rs` cover the read-only hooks; this file
//! pins the contract that:
//!
//! 1. `_mut` hooks are invoked in registration order **before any**
//!    read-only hook fires (transformers run as a phase ahead of
//!    observers), and
//! 2. mutations made in a `_mut` hook are visible to every subsequent
//!    observer, to the engine's history builder, and to the stream
//!    consumer.
//!
//! These properties are the whole reason the trait splits into
//! observer + transformer methods; if the engine ever stops running
//! `_mut` first, this test catches it.

use std::sync::{Arc, Mutex};

use ailoop::{Conversation, Message, ToolDefinition, ToolResultContent, advanced::run_chat};
use ailoop_core::testing::ScriptedModel;
use ailoop_core::{ChatMiddleware, FinishReason, RunConfig, RunId, StepId, StreamChunk, Usage};
use ailoop_tools::{ToolDyn, ToolRegistry};
use futures::StreamExt;
use serde_json::{Value, json};

/// Records every `delta` it sees through `on_chunk` (after any `_mut`
/// hooks have run). The recorded value is the post-mutation delta — a
/// regression where `_mut` runs *after* observers would surface here as
/// the original (un-mutated) string.
#[derive(Default)]
struct DeltaObserver {
    seen: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl ChatMiddleware for DeltaObserver {
    async fn on_chunk(&self, chunk: &StreamChunk) {
        if let StreamChunk::TextDelta { delta } = chunk {
            self.seen.lock().unwrap().push(delta.clone());
        }
    }
}

/// Rewrites every `TextDelta.delta` to its uppercase form. Pairs with
/// `DeltaObserver` to assert the mutation is visible downstream.
struct UppercaseDeltas;

#[async_trait::async_trait]
impl ChatMiddleware for UppercaseDeltas {
    async fn on_chunk_mut(&self, chunk: &mut StreamChunk) {
        if let StreamChunk::TextDelta { delta } = chunk {
            *delta = delta.to_uppercase();
        }
    }
}

/// `on_chunk_mut` runs before `on_chunk`, so a mutator registered first
/// or last still wins: every observer sees the post-mutation value.
#[tokio::test]
async fn on_chunk_mut_mutation_visible_to_subsequent_on_chunk() {
    let model = ScriptedModel::new([vec![
        StreamChunk::TextDelta {
            delta: "hello".into(),
        },
        StreamChunk::TextDelta {
            delta: " world".into(),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        },
    ]]);

    let observer = Arc::new(DeltaObserver::default());
    let mutator: Arc<dyn ChatMiddleware> = Arc::new(UppercaseDeltas);

    let registry = ToolRegistry::new();
    let mut config = RunConfig::default();
    config.middlewares = vec![mutator, observer.clone() as Arc<dyn ChatMiddleware>];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|c| c.ok())
        .collect();

    // Observer captured the mutated values.
    let seen = observer.seen.lock().unwrap().clone();
    assert_eq!(seen, vec!["HELLO".to_string(), " WORLD".to_string()]);

    // Stream consumer also sees mutated values.
    let yielded: Vec<String> = chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::TextDelta { delta } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(yielded, vec!["HELLO".to_string(), " WORLD".to_string()]);
}

/// Three middlewares registered as `[mut1, obs, mut2]`. Even though
/// `obs` sits between the two mutators, the engine runs **every**
/// `_mut` (in registration order) before **any** `on_chunk`, so `obs`
/// must see the value after both mutators have applied.
#[tokio::test]
async fn all_chunk_muts_run_before_any_observer() {
    /// Prefixes the delta with `tag`, so the order of mutator
    /// invocations is recoverable from the final string.
    struct Prefix(&'static str);
    #[async_trait::async_trait]
    impl ChatMiddleware for Prefix {
        async fn on_chunk_mut(&self, chunk: &mut StreamChunk) {
            if let StreamChunk::TextDelta { delta } = chunk {
                *delta = format!("{}:{}", self.0, delta);
            }
        }
    }

    let model = ScriptedModel::new([vec![
        StreamChunk::TextDelta { delta: "x".into() },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        },
    ]]);

    let observer = Arc::new(DeltaObserver::default());
    let registry = ToolRegistry::new();
    let mut config = RunConfig::default();
    config.middlewares = vec![
        Arc::new(Prefix("a")) as Arc<dyn ChatMiddleware>,
        observer.clone() as Arc<dyn ChatMiddleware>,
        Arc::new(Prefix("b")) as Arc<dyn ChatMiddleware>,
    ];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let _: Vec<_> = stream.collect().await;

    let seen = observer.seen.lock().unwrap().clone();
    // Both prefixes applied in registration order before the observer ran.
    assert_eq!(seen, vec!["b:a:x".to_string()]);
}

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
        // Echo what the tool actually received so tests can assert
        // that `_mut` mutations propagate through to the tool.
        ToolResultContent::Text(args.to_string())
    }
}

/// `on_before_tool_call_mut` rewrites `args` and the change must be
/// visible to (a) any observer in `on_before_tool_call`, (b) the tool
/// itself when the engine invokes it.
#[tokio::test]
async fn on_before_tool_call_mut_mutation_visible_to_tool_and_observer() {
    /// Captures the args it sees in `on_before_tool_call`, after any
    /// `_mut` has run.
    #[derive(Default)]
    struct ArgsObserver {
        seen: Mutex<Option<Value>>,
    }

    #[async_trait::async_trait]
    impl ChatMiddleware for ArgsObserver {
        async fn on_before_tool_call(
            &self,
            _: &RunId,
            _: &StepId,
            _: &str,
            args: &Value,
        ) -> ailoop_core::ToolDecision {
            *self.seen.lock().unwrap() = Some(args.clone());
            ailoop_core::ToolDecision::Continue
        }
    }

    /// Adds a `redacted: true` field so the assertion can confirm the
    /// mutation flowed through.
    struct RedactArgs;
    #[async_trait::async_trait]
    impl ChatMiddleware for RedactArgs {
        async fn on_before_tool_call_mut(&self, _: &RunId, _: &StepId, _: &str, args: &mut Value) {
            if let Some(obj) = args.as_object_mut() {
                obj.insert("redacted".into(), Value::Bool(true));
            }
        }
    }

    let turn1 = vec![
        StreamChunk::ToolCallStarted {
            id: "toolu_1".into(),
            name: "echo".into(),
        },
        StreamChunk::ToolCallFinished {
            id: "toolu_1".into(),
            name: "echo".into(),
            args: json!({"secret": "abc"}),
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

    let observer = Arc::new(ArgsObserver::default());
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoArgs)).unwrap();

    let mut config = RunConfig::default();
    config.middlewares = vec![
        Arc::new(RedactArgs) as Arc<dyn ChatMiddleware>,
        observer.clone() as Arc<dyn ChatMiddleware>,
    ];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|c| c.ok())
        .collect();

    // (a) Observer saw the mutated args.
    let seen = observer
        .seen
        .lock()
        .unwrap()
        .clone()
        .expect("args observed");
    assert_eq!(seen, json!({"secret": "abc", "redacted": true}));

    // (b) Tool received the mutated args (echoed back into ToolResult).
    let tool_result_text = chunks
        .iter()
        .find_map(|c| match c {
            StreamChunk::ToolResult {
                content: ToolResultContent::Text(t),
                ..
            } => Some(t.clone()),
            _ => None,
        })
        .expect("ToolResult should be emitted");
    let parsed: Value = serde_json::from_str(&tool_result_text).expect("echoed args parse");
    assert_eq!(parsed, json!({"secret": "abc", "redacted": true}));
}

/// `on_after_tool_call_mut` rewrites the tool's result before any
/// observer sees it, and the rewritten result is what the engine emits
/// as `StreamChunk::ToolResult` and what lands in history (the model's
/// next turn sees the post-mutation value).
#[tokio::test]
async fn on_after_tool_call_mut_mutation_visible_to_observer_and_history() {
    /// Captures the `ToolResultContent` it sees in `on_after_tool_call`.
    #[derive(Default)]
    struct ResultObserver {
        seen: Mutex<Option<ToolResultContent>>,
    }
    #[async_trait::async_trait]
    impl ChatMiddleware for ResultObserver {
        async fn on_after_tool_call(
            &self,
            _: &RunId,
            _: &StepId,
            _: &str,
            _: &Value,
            result: &ToolResultContent,
        ) {
            *self.seen.lock().unwrap() = Some(match result {
                ToolResultContent::Text(t) => ToolResultContent::Text(t.clone()),
                ToolResultContent::Error(e) => ToolResultContent::Error(e.clone()),
                _ => ToolResultContent::Error("unknown variant".into()),
            });
        }
    }

    struct RewriteResult;
    #[async_trait::async_trait]
    impl ChatMiddleware for RewriteResult {
        async fn on_after_tool_call_mut(
            &self,
            _: &RunId,
            _: &StepId,
            _: &str,
            _: &Value,
            result: &mut ToolResultContent,
        ) {
            if let ToolResultContent::Text(t) = result {
                *t = format!("[redacted] {t}");
            }
        }
    }

    let turn1 = vec![
        StreamChunk::ToolCallStarted {
            id: "toolu_1".into(),
            name: "echo".into(),
        },
        StreamChunk::ToolCallFinished {
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
    let model = ScriptedModel::new([turn1, turn2]);

    let observer = Arc::new(ResultObserver::default());
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(EchoArgs)).unwrap();

    let mut config = RunConfig::default();
    config.middlewares = vec![
        Arc::new(RewriteResult) as Arc<dyn ChatMiddleware>,
        observer.clone() as Arc<dyn ChatMiddleware>,
    ];

    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(|c| c.ok())
        .collect();

    // Observer saw the mutated result.
    match observer.seen.lock().unwrap().clone().expect("observed") {
        ToolResultContent::Text(t) => assert!(
            t.starts_with("[redacted]"),
            "expected mutated result, got {t:?}"
        ),
        other => panic!("expected Text, got {other:?}"),
    }

    // The emitted `ToolResult` chunk reflects the mutation.
    let yielded = chunks
        .iter()
        .find_map(|c| match c {
            StreamChunk::ToolResult { content, .. } => Some(content.clone()),
            _ => None,
        })
        .expect("ToolResult should be emitted");
    match yielded {
        ToolResultContent::Text(t) => assert!(t.starts_with("[redacted]")),
        other => panic!("expected Text, got {other:?}"),
    }
}

/// End-to-end check via `Conversation::run`: a `_mut` rewriting every
/// `TextDelta` lands in `RunOutcome.final_text`. This is the surface
/// most users will interact with (CLI / batch flows), so it deserves
/// its own assertion separate from the engine-level tests above.
#[tokio::test]
async fn run_outcome_final_text_reflects_chunk_mutation() {
    let model = ScriptedModel::new([vec![
        StreamChunk::TextDelta {
            delta: "hello".into(),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        },
    ]]);

    let mut conv = Conversation::builder(model)
        .middleware(Arc::new(UppercaseDeltas))
        .build()
        .expect("builder should succeed");

    let outcome = conv.run("hi").await.expect("run should succeed");
    assert_eq!(outcome.final_text.as_deref(), Some("HELLO"));
}
