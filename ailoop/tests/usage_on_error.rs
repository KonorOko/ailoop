//! A run that ends in `Err` still reports what it spent:
//! `RunError::usage` and the `usage` passed to `on_run_error` hold the
//! run's finished turns plus the usage tools reported (sub-agents
//! included). The turn that failed is not counted.

use std::sync::{Arc, Mutex};

use ailoop::{
    ChatMiddleware, Conversation, EngineError, FinishReason, History, RunError, RunErrorInfo,
    StreamChunk, SubAgentTool, ToolContext, ToolDefinition, ToolDyn, ToolResultContent, Usage,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel, ScriptedTurn};
use async_trait::async_trait;
use serde_json::{Value, json};

fn tokens(input: u32, output: u32) -> Usage {
    let mut u = Usage::default();
    u.input_tokens = input;
    u.output_tokens = output;
    u
}

/// Tool that spends tokens of its own (as a tool calling an LLM would)
/// and reports them.
struct Reporter {
    spend: Usage,
}

#[async_trait]
impl ToolDyn for Reporter {
    fn name(&self) -> String {
        "summarize".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new("summarize", "stub", json!({"type":"object"}), vec![])
    }
    async fn call(&self, _args: Value, ctx: &ToolContext) -> ToolResultContent {
        ctx.report_usage(self.spend);
        ToolResultContent::text("summary")
    }
}

/// Records the usage every `on_run_error` receives.
#[derive(Default)]
struct Spy {
    run_errors: Mutex<Vec<Usage>>,
}

#[async_trait]
impl ChatMiddleware for Spy {
    async fn on_run_error(&self, run: &RunErrorInfo<'_>) {
        self.run_errors.lock().unwrap().push(*run.usage);
    }
}

impl Spy {
    /// The single `on_run_error` call must carry the same usage as the
    /// returned error.
    fn assert_saw(&self, err: &RunError<ScriptedError>) {
        assert_eq!(
            *self.run_errors.lock().unwrap(),
            [err.usage()],
            "on_run_error fires once with RunError::usage()"
        );
    }
}

fn call_turn(tool: &str, args: Value, usage: Usage) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::ToolCallStarted {
            call_id: "toolu_1".into(),
            name: tool.into(),
        }),
        Ok(StreamChunk::ToolCallFinished {
            call_id: "toolu_1".into(),
            name: tool.into(),
            args,
        }),
        Ok(StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage,
            service_tier: None,
        }),
    ])
}

fn text_turn(text: &str, usage: Usage) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::TextDelta { delta: text.into() }),
        Ok(StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage,
            service_tier: None,
        }),
    ])
}

/// Streams some text, then the connection drops: no `TurnFinished`.
fn broken_turn() -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::TextDelta {
            delta: "half an ans".into(),
        }),
        Err(ScriptedError("connection dropped".into())),
    ])
}

/// The request itself fails.
fn rejected_turn() -> ScriptedTurn {
    Err(ScriptedError("503".into()))
}

#[tokio::test]
async fn usage_counts_finished_turns_and_tool_reports() {
    let spy = Arc::new(Spy::default());
    let mut chat = Conversation::builder(ScriptedModel::with_turns([
        call_turn("summarize", json!({}), tokens(10, 2)),
        broken_turn(),
    ]))
    .tool_dyn(Arc::new(Reporter {
        spend: tokens(300, 40),
    }))
    .middleware(spy.clone())
    .build()
    .expect("build");

    let err = chat.run("go").await.expect_err("second turn fails");

    assert!(matches!(err.kind(), EngineError::Model(_)), "{err:?}");
    assert_eq!(err.usage(), tokens(310, 42));
    spy.assert_saw(&err);
}

#[tokio::test]
async fn usage_includes_a_sub_agent_that_ran_before_the_failure() {
    let child = Conversation::builder(ScriptedModel::new([vec![
        StreamChunk::TextDelta {
            delta: "child answer".into(),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: tokens(100, 20),
            service_tier: None,
        },
    ]]))
    .build()
    .expect("build child");

    let spy = Arc::new(Spy::default());
    let mut parent = Conversation::builder(ScriptedModel::with_turns([
        call_turn("delegate", json!({"prompt": "go"}), tokens(10, 1)),
        rejected_turn(),
    ]))
    .tool_dyn(Arc::new(SubAgentTool::new("delegate", "delegate", child)))
    .middleware(spy.clone())
    .build()
    .expect("build parent");

    let err = parent.run("start").await.expect_err("parent fails");

    assert!(matches!(err.kind(), EngineError::Model(_)), "{err:?}");
    assert_eq!(err.usage(), tokens(110, 21));
    spy.assert_saw(&err);
}

#[tokio::test]
async fn failing_first_turn_has_zero_usage() {
    for (name, turn) in [("mid-stream", broken_turn()), ("setup", rejected_turn())] {
        let spy = Arc::new(Spy::default());
        let mut chat = Conversation::builder(ScriptedModel::with_turns([
            turn,
            text_turn("never", tokens(1, 1)),
        ]))
        .middleware(spy.clone())
        .build()
        .expect("build");

        let err = chat.run("go").await.expect_err("first turn fails");

        assert_eq!(err.usage(), Usage::default(), "{name}");
        spy.assert_saw(&err);
    }
}

/// Compaction before the run starts fails before any hook fires: the
/// error has zero usage and `on_run_error` is not called.
#[tokio::test]
async fn pre_run_compaction_error_has_zero_usage_and_no_hook() {
    let spy = Arc::new(Spy::default());
    let mut chat = Conversation::builder(ScriptedModel::with_turns([text_turn(
        "never",
        tokens(1, 1),
    )]))
    .with_history(History::builder(10).preserve_n_last(5))
    .middleware(spy.clone())
    .build()
    .expect("build");

    let err = chat
        .run("x".repeat(400))
        .await
        .expect_err("compaction fails");

    assert!(matches!(err.kind(), EngineError::Context(_)), "{err:?}");
    assert_eq!(err.usage(), Usage::default());
    assert!(spy.run_errors.lock().unwrap().is_empty());
}
