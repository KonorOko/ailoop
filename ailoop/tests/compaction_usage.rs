//! What compacting costs: the `usage` on `StreamChunk::HistoryCompacted`
//! and its share of the run's total, at each point a run compacts, on
//! success and on failure, and through a sub-agent.
//!
//! Budgets use the default `CharTokenizer` (`len() / 4`). Overflow is
//! simulated with `ScriptedError("context_overflow")`.

use std::sync::{Arc, Mutex};

use ailoop::{
    ChatMiddleware, CompactionError, CompactionOutput, CompactionStrategy, CompletionModel,
    Conversation, EngineError, FinishReason, History, HistoryBuilder, Message, ProviderError,
    RunError, RunErrorInfo, RunOptions, StreamChunk, SubAgentTool, SummarizeStrategy,
    ToolDefinition, ToolResultContent, Usage,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel, ScriptedTurn};
use ailoop_tools::{ToolContext, ToolDyn};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

fn tokens(input: u64, output: u64) -> Usage {
    let mut usage = Usage::default();
    usage.input_tokens = input;
    usage.output_tokens = output;
    usage
}

/// What each summary request costs, distinct from any main-model turn.
fn summary_spend() -> Usage {
    tokens(1_000, 100)
}

fn text_turn(text: &str, usage: Usage) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::TextDelta { delta: text.into() }),
        Ok(StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage,
            service_tier: None,
            model: None,
        }),
    ])
}

fn summary_turn() -> ScriptedTurn {
    text_turn("the summary", summary_spend())
}

fn call_turn(tool: &str, args: Value, usage: Usage) -> ScriptedTurn {
    Ok(vec![
        Ok(StreamChunk::ToolCallStarted {
            call_id: "c1".into(),
            name: tool.into(),
        }),
        Ok(StreamChunk::ToolCallFinished {
            call_id: "c1".into(),
            name: tool.into(),
            args,
        }),
        Ok(StreamChunk::TurnFinished {
            reason: FinishReason::ToolUse,
            usage,
            service_tier: None,
            model: None,
        }),
    ])
}

fn overflow_turn() -> ScriptedTurn {
    Err(ScriptedError("context_overflow".into()))
}

/// A summarizer that rejects the first request, accepts the second and
/// then fails: the compaction ends in `Err` having paid for one summary.
fn failing_split() -> [ScriptedTurn; 3] {
    [
        overflow_turn(),
        summary_turn(),
        Err(ScriptedError("boom".into())),
    ]
}

fn summarizing(
    budget: usize,
    summarizer: impl IntoIterator<Item = ScriptedTurn>,
) -> HistoryBuilder {
    History::builder(budget)
        .preserve_n_last(1)
        .strategy(Box::new(SummarizeStrategy::new(Arc::new(
            ScriptedModel::with_turns(summarizer),
        ))))
}

/// Four messages of 100 tokens each.
fn seed<M>(chat: &mut Conversation<M>)
where
    M: CompletionModel,
    M::Error: ProviderError,
{
    for i in 0..2 {
        chat.push_message(Message::user(format!("{i}{}", "q".repeat(399))));
        chat.push_message(Message::assistant_text(format!("{i}{}", "a".repeat(399))));
    }
}

/// Returns a 2 400-character result (600 tokens under `CharTokenizer`).
struct BigTool;

#[async_trait]
impl ToolDyn for BigTool {
    fn name(&self) -> &str {
        "big"
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new("big", "stub", json!({"type":"object"}), vec![])
    }
    async fn call(&self, _: Value, _ctx: &ToolContext) -> ToolResultContent {
        ToolResultContent::text("x".repeat(2_400))
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

/// Drains a run that ends in `RunFinished`: the usage of each
/// `HistoryCompacted`, then the finish reason and total.
async fn run_to_end(chat: &mut Conversation<ScriptedModel>) -> (Vec<Usage>, FinishReason, Usage) {
    let mut stream = chat
        .stream_with_options("go", RunOptions::new())
        .await
        .expect("run starts");
    let mut compactions = Vec::new();
    while let Some(chunk) = stream.next().await {
        match chunk.expect("no error") {
            StreamChunk::HistoryCompacted { usage, .. } => compactions.push(usage),
            StreamChunk::RunFinished { reason, usage, .. } => {
                return (compactions, reason, usage);
            }
            _ => {}
        }
    }
    panic!("no RunFinished");
}

/// Asserts a failed compaction's spend, on the error and on the hook.
fn assert_failed_with(err: &RunError<ScriptedError>, spy: &Spy, expected: Usage) {
    assert!(matches!(err.kind(), EngineError::Context(_)), "{err:?}");
    assert_eq!(err.usage(), expected);
    assert_eq!(*spy.run_errors.lock().unwrap(), [expected]);
    match err.kind() {
        EngineError::Context(compaction) => assert_eq!(compaction.usage(), summary_spend()),
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn pre_run_summary_counts_in_the_run_and_on_the_chunk() {
    let mut chat =
        Conversation::builder(ScriptedModel::with_turns([text_turn("done", tokens(5, 1))]))
            .history(summarizing(300, [summary_turn()]))
            .build()
            .unwrap();
    seed(&mut chat);

    let (compactions, _, total) = run_to_end(&mut chat).await;

    assert_eq!(compactions, [summary_spend()]);
    assert_eq!(total, tokens(1_005, 101));
}

#[tokio::test]
async fn between_iterations_summary_counts_in_the_run_and_on_the_chunk() {
    let mut chat = Conversation::builder(ScriptedModel::with_turns([
        call_turn("big", json!({}), tokens(5, 1)),
        text_turn("done", tokens(7, 2)),
    ]))
    .tool(BigTool)
    .history(summarizing(1_000, [summary_turn()]))
    .compact_between_iterations(true)
    .build()
    .unwrap();
    seed(&mut chat);

    let (compactions, _, total) = run_to_end(&mut chat).await;

    assert_eq!(compactions, [summary_spend()]);
    assert_eq!(total, tokens(1_012, 103));
}

#[tokio::test]
async fn overflow_recovery_summary_counts_in_the_run_and_on_the_chunk() {
    let mut chat = Conversation::builder(ScriptedModel::with_turns([
        overflow_turn(),
        text_turn("fits now", tokens(5, 1)),
    ]))
    .history(summarizing(100_000, [summary_turn()]))
    .build()
    .unwrap();
    seed(&mut chat);

    let (compactions, _, total) = run_to_end(&mut chat).await;

    assert_eq!(compactions, [summary_spend()]);
    assert_eq!(total, tokens(1_005, 101));
}

/// A summary that does not shrink the history gets no chunk and ends
/// the recovery, but the provider billed it.
#[tokio::test]
async fn overflow_recovery_without_progress_still_counts_the_summary() {
    let spy = Arc::new(Spy::default());
    let mut chat = Conversation::builder(ScriptedModel::with_turns([overflow_turn()]))
        .history(summarizing(
            100_000,
            [text_turn(&"s".repeat(10_000), summary_spend())],
        ))
        .middleware(spy.clone())
        .build()
        .unwrap();
    seed(&mut chat);

    let err = chat.run("go").await.expect_err("still too large");

    assert!(
        matches!(err.kind(), EngineError::ContextOverflow(_)),
        "{err:?}"
    );
    assert_eq!(err.usage(), summary_spend());
    assert_eq!(*spy.run_errors.lock().unwrap(), [summary_spend()]);
}

#[tokio::test]
async fn abort_after_a_compaction_includes_it() {
    let mut chat = Conversation::builder(ScriptedModel::with_turns([]))
        .history(summarizing(300, [summary_turn()]))
        .max_iterations(0)
        .build()
        .unwrap();
    seed(&mut chat);

    let (_, reason, total) = run_to_end(&mut chat).await;

    assert!(matches!(reason, FinishReason::Aborted { .. }), "{reason:?}");
    assert_eq!(total, summary_spend());
}

#[tokio::test]
async fn model_error_after_a_compaction_includes_it() {
    let spy = Arc::new(Spy::default());
    let mut chat = Conversation::builder(ScriptedModel::with_turns([Err(ScriptedError(
        "boom".into(),
    ))]))
    .history(summarizing(300, [summary_turn()]))
    .middleware(spy.clone())
    .build()
    .unwrap();
    seed(&mut chat);

    let err = chat.run("go").await.expect_err("the model fails");

    assert!(matches!(err.kind(), EngineError::Model(_)), "{err:?}");
    assert_eq!(err.usage(), summary_spend());
    assert_eq!(*spy.run_errors.lock().unwrap(), [summary_spend()]);
}

/// One pair over budget takes three requests: both messages rejected,
/// then one message each, the second with the first's summary.
#[tokio::test]
async fn split_compaction_sums_every_request_in_the_run() {
    let mut chat =
        Conversation::builder(ScriptedModel::with_turns([text_turn("done", tokens(5, 1))]))
            .history(summarizing(
                150,
                [overflow_turn(), summary_turn(), summary_turn()],
            ))
            .build()
            .unwrap();
    chat.push_message(Message::user("q".repeat(400)));
    chat.push_message(Message::assistant_text("a".repeat(400)));

    let (compactions, _, total) = run_to_end(&mut chat).await;

    assert_eq!(compactions, [tokens(2_000, 200)]);
    assert_eq!(total, tokens(2_005, 201));
}

#[tokio::test]
async fn truncate_compaction_reports_zero() {
    let mut chat =
        Conversation::builder(ScriptedModel::with_turns([text_turn("done", tokens(5, 1))]))
            .history(History::builder(300).preserve_n_last(1))
            .build()
            .unwrap();
    seed(&mut chat);

    let (compactions, _, total) = run_to_end(&mut chat).await;

    assert_eq!(compactions, [Usage::default()]);
    assert_eq!(total, tokens(5, 1));
}

#[tokio::test]
async fn failed_pre_run_split_reports_what_it_spent() {
    let spy = Arc::new(Spy::default());
    let mut chat = Conversation::builder(ScriptedModel::with_turns([]))
        .history(summarizing(300, failing_split()))
        .middleware(spy.clone())
        .build()
        .unwrap();
    seed(&mut chat);

    let err = chat.run("go").await.expect_err("compaction fails");

    assert_failed_with(&err, &spy, summary_spend());
}

#[tokio::test]
async fn failed_split_between_iterations_adds_to_the_turns_so_far() {
    let spy = Arc::new(Spy::default());
    let mut chat = Conversation::builder(ScriptedModel::with_turns([call_turn(
        "big",
        json!({}),
        tokens(5, 1),
    )]))
    .tool(BigTool)
    .history(summarizing(1_000, failing_split()))
    .compact_between_iterations(true)
    .middleware(spy.clone())
    .build()
    .unwrap();
    seed(&mut chat);

    let err = chat.run("go").await.expect_err("compaction fails");

    assert_failed_with(&err, &spy, tokens(1_005, 101));
}

#[tokio::test]
async fn failed_split_during_overflow_recovery_reports_what_it_spent() {
    let spy = Arc::new(Spy::default());
    let mut chat = Conversation::builder(ScriptedModel::with_turns([overflow_turn()]))
        .history(summarizing(100_000, failing_split()))
        .middleware(spy.clone())
        .build()
        .unwrap();
    seed(&mut chat);

    let err = chat.run("go").await.expect_err("compaction fails");

    assert_failed_with(&err, &spy, summary_spend());
}

/// A child over budget summarizes before its first turn. The exact
/// total also rules out counting the summary twice.
#[tokio::test]
async fn sub_agent_parent_receives_the_child_compaction() {
    let mut child = Conversation::builder(ScriptedModel::with_turns([text_turn(
        "child answer",
        tokens(200, 20),
    )]))
    .history(summarizing(300, [summary_turn()]))
    .build()
    .unwrap();
    seed(&mut child);

    let mut parent = Conversation::builder(ScriptedModel::with_turns([
        call_turn("delegate", json!({"prompt": "go"}), tokens(10, 1)),
        text_turn("done", tokens(5, 1)),
    ]))
    .tool_dyn(Arc::new(SubAgentTool::new("delegate", "delegate", child)))
    .build()
    .unwrap();

    let (compactions, _, total) = run_to_end(&mut parent).await;

    assert!(
        compactions.is_empty(),
        "the child's chunk stays in the child"
    );
    assert_eq!(total, tokens(1_215, 122));
}

#[tokio::test]
async fn sub_agent_parent_receives_a_failed_child_compaction() {
    let mut child = Conversation::builder(ScriptedModel::with_turns([]))
        .history(summarizing(300, failing_split()))
        .build()
        .unwrap();
    seed(&mut child);

    let mut parent = Conversation::builder(ScriptedModel::with_turns([
        call_turn("delegate", json!({"prompt": "go"}), tokens(10, 1)),
        text_turn("done", tokens(5, 1)),
    ]))
    .tool_dyn(Arc::new(SubAgentTool::new("delegate", "delegate", child)))
    .build()
    .unwrap();

    let (_, _, total) = run_to_end(&mut parent).await;

    assert_eq!(total, tokens(1_015, 102));
}

/// A strategy from outside the workspace's crates, written as before
/// compaction reported usage.
struct ThirdParty;

#[async_trait]
impl CompactionStrategy for ThirdParty {
    fn name(&self) -> &'static str {
        "third-party"
    }

    async fn compact(
        &self,
        messages: &[Message],
        pinned: &[bool],
        preserve_n_last: usize,
    ) -> Result<CompactionOutput, CompactionError> {
        let start = messages.len().saturating_sub(preserve_n_last);
        Ok(CompactionOutput::new(
            messages[start..].to_vec(),
            pinned[start..].to_vec(),
        ))
    }
}

#[tokio::test]
async fn third_party_strategy_using_new_reports_zero() {
    let mut chat =
        Conversation::builder(ScriptedModel::with_turns([text_turn("done", tokens(5, 1))]))
            .history(
                History::builder(300)
                    .preserve_n_last(1)
                    .strategy(Box::new(ThirdParty)),
            )
            .build()
            .unwrap();
    seed(&mut chat);

    let (compactions, _, total) = run_to_end(&mut chat).await;

    assert_eq!(compactions, [Usage::default()]);
    assert_eq!(total, tokens(5, 1));
}

/// A strategy with its own model reports through the builders.
struct Billed {
    fail: bool,
}

#[async_trait]
impl CompactionStrategy for Billed {
    fn name(&self) -> &'static str {
        "billed"
    }

    async fn compact(
        &self,
        messages: &[Message],
        pinned: &[bool],
        preserve_n_last: usize,
    ) -> Result<CompactionOutput, CompactionError> {
        if self.fail {
            return Err(
                CompactionError::summarization_failed("refused", false, None)
                    .with_usage(summary_spend()),
            );
        }
        ThirdParty
            .compact(messages, pinned, preserve_n_last)
            .await
            .map(|output| output.with_usage(summary_spend()))
    }
}

#[tokio::test]
async fn third_party_strategy_reports_through_the_builders() {
    let mut chat =
        Conversation::builder(ScriptedModel::with_turns([text_turn("done", tokens(5, 1))]))
            .history(
                History::builder(300)
                    .preserve_n_last(1)
                    .strategy(Box::new(Billed { fail: false })),
            )
            .build()
            .unwrap();
    seed(&mut chat);

    let (compactions, _, total) = run_to_end(&mut chat).await;
    assert_eq!(compactions, [summary_spend()]);
    assert_eq!(total, tokens(1_005, 101));

    let spy = Arc::new(Spy::default());
    let mut chat = Conversation::builder(ScriptedModel::with_turns([]))
        .history(
            History::builder(300)
                .preserve_n_last(1)
                .strategy(Box::new(Billed { fail: true })),
        )
        .middleware(spy.clone())
        .build()
        .unwrap();
    seed(&mut chat);

    let err = chat.run("go").await.expect_err("compaction fails");
    assert_failed_with(&err, &spy, summary_spend());
}
