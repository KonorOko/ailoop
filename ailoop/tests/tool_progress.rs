//! `ToolContext::progress`: updates a running tool reports reach the
//! stream as `ToolProgress` chunks with the call's id, before its
//! result, without entering the history; stale or excess updates are
//! dropped and reporting never blocks the tool.

use std::sync::Arc;
use std::time::Duration;

use ailoop::{
    FinishReason, Message, RunConfig, StreamChunk, ToolDefinition, ToolProgress, ToolResultContent,
    Usage, advanced::run_chat,
};
use ailoop_core::testing::ScriptedModel;
use ailoop_tools::{ToolContext, ToolDyn, ToolRegistry};
use futures::StreamExt;
use serde_json::{Value, json};

/// What a `Reporter` does when called.
#[derive(Clone, Copy)]
enum Behaviour {
    /// Reports `n` steps of progress, pausing between them.
    Steps(usize),
    /// Reports `n` updates at once, without yielding.
    Burst(usize),
    /// Returns at once, leaving a task that reports after `delay`.
    LateReport(Duration),
    /// Sleeps for `delay` without reporting.
    Quiet(Duration),
}

struct Reporter {
    name: &'static str,
    behaviour: Behaviour,
}

#[async_trait::async_trait]
impl ToolDyn for Reporter {
    fn name(&self) -> &str {
        self.name
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition::new(self.name, "reports", json!({"type": "object"}), vec![])
    }
    fn concurrency_safe(&self) -> bool {
        true
    }
    async fn call(&self, _: Value, ctx: &ToolContext) -> ToolResultContent {
        match self.behaviour {
            Behaviour::Steps(n) => {
                for i in 1..=n {
                    ctx.progress(
                        ToolProgress::new()
                            .with_progress(i as f64)
                            .with_total(n as f64)
                            .with_message(format!("{} {i}", ctx.call_id())),
                    );
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
            Behaviour::Burst(n) => {
                for i in 0..n {
                    ctx.progress(ToolProgress::new().with_progress(i as f64));
                }
            }
            Behaviour::LateReport(delay) => {
                let sink = ctx.progress_sink().clone();
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    sink.report(ToolProgress::message("too late"));
                });
            }
            Behaviour::Quiet(delay) => tokio::time::sleep(delay).await,
        }
        ToolResultContent::text("done")
    }
}

fn calls_turn(calls: &[(&str, &str)]) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();
    for (id, name) in calls {
        chunks.push(StreamChunk::ToolCallStarted {
            call_id: (*id).into(),
            name: (*name).into(),
        });
        chunks.push(StreamChunk::ToolCallFinished {
            call_id: (*id).into(),
            name: (*name).into(),
            args: json!({}),
        });
    }
    chunks.push(StreamChunk::TurnFinished {
        reason: FinishReason::ToolUse,
        usage: Usage::default(),
        service_tier: None,
    });
    chunks
}

fn end_turn() -> Vec<StreamChunk> {
    vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }]
}

async fn run(
    turns: Vec<Vec<StreamChunk>>,
    tools: Vec<Reporter>,
    concurrency: usize,
) -> Vec<StreamChunk> {
    let mut registry = ToolRegistry::new();
    for tool in tools {
        registry.register(Arc::new(tool)).unwrap();
    }
    let model = ScriptedModel::new(turns);
    let mut config = RunConfig::default();
    config.tool_concurrency = concurrency;
    run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run starts")
        .map(|c| c.expect("no error"))
        .collect()
        .await
}

/// `(call_id, message)` of every progress chunk, and the position of
/// each chunk kind, in stream order.
fn events(chunks: &[StreamChunk]) -> Vec<String> {
    chunks
        .iter()
        .filter_map(|c| match c {
            StreamChunk::ToolProgress {
                call_id, progress, ..
            } => Some(format!(
                "progress:{call_id}:{}",
                progress.message.clone().unwrap_or_default()
            )),
            StreamChunk::ToolResult { call_id, .. } => Some(format!("result:{call_id}")),
            _ => None,
        })
        .collect()
}

/// Updates arrive in order, with the call's id, before its result, and
/// stay out of the history.
#[tokio::test]
async fn progress_precedes_the_result_and_skips_history() {
    let chunks = run(
        vec![calls_turn(&[("t1", "steps")]), end_turn()],
        vec![Reporter {
            name: "steps",
            behaviour: Behaviour::Steps(3),
        }],
        1,
    )
    .await;

    assert_eq!(
        events(&chunks),
        [
            "progress:t1:t1 1",
            "progress:t1:t1 2",
            "progress:t1:t1 3",
            "result:t1"
        ]
    );
    let first = chunks.iter().find_map(|c| match c {
        StreamChunk::ToolProgress { progress, .. } => Some(progress.clone()),
        _ => None,
    });
    assert_eq!(
        first,
        Some(
            ToolProgress::new()
                .with_progress(1.0)
                .with_total(3.0)
                .with_message("t1 1")
        )
    );
    let Some(StreamChunk::RunFinished { new_messages, .. }) = chunks.last() else {
        panic!("no RunFinished");
    };
    let text = format!("{new_messages:?}");
    assert!(!text.contains("t1 1"), "progress must not reach history");
}

/// In a parallel batch each update carries its own call's id, and all
/// of them precede the batch's results.
#[tokio::test]
async fn batch_updates_carry_their_own_call_id() {
    let chunks = run(
        vec![calls_turn(&[("t1", "steps"), ("t2", "steps")]), end_turn()],
        vec![Reporter {
            name: "steps",
            behaviour: Behaviour::Steps(2),
        }],
        2,
    )
    .await;

    let events = events(&chunks);
    let first_result = events.iter().position(|e| e.starts_with("result")).unwrap();
    let progress: Vec<&String> = events[..first_result].iter().collect();
    assert_eq!(progress.len(), 4);
    for id in ["t1", "t2"] {
        assert!(events.contains(&format!("progress:{id}:{id} 1")));
        assert!(events.contains(&format!("progress:{id}:{id} 2")));
    }
    assert_eq!(&events[first_result..], ["result:t1", "result:t2"]);
}

/// An update sent after its call has its result is dropped, even when
/// it arrives while a later call runs.
#[tokio::test]
async fn late_updates_are_dropped() {
    let chunks = run(
        vec![
            calls_turn(&[("t1", "late")]),
            calls_turn(&[("t2", "quiet")]),
            end_turn(),
        ],
        vec![
            Reporter {
                name: "late",
                behaviour: Behaviour::LateReport(Duration::from_millis(10)),
            },
            Reporter {
                name: "quiet",
                behaviour: Behaviour::Quiet(Duration::from_millis(60)),
            },
        ],
        1,
    )
    .await;
    assert_eq!(events(&chunks), ["result:t1", "result:t2"]);
}

/// A tool that floods updates is never blocked; past the buffer they
/// are dropped.
#[tokio::test]
async fn a_burst_never_blocks_the_tool() {
    let chunks = tokio::time::timeout(
        Duration::from_secs(5),
        run(
            vec![calls_turn(&[("t1", "burst")]), end_turn()],
            vec![Reporter {
                name: "burst",
                behaviour: Behaviour::Burst(1_000),
            }],
            1,
        ),
    )
    .await
    .expect("the tool must not block");
    let progress = chunks
        .iter()
        .filter(|c| matches!(c, StreamChunk::ToolProgress { .. }))
        .count();
    assert!(progress > 0 && progress < 1_000, "got {progress}");
}

/// Reporting on a detached context is a no-op.
#[test]
fn detached_context_ignores_progress() {
    ToolContext::detached().progress(ToolProgress::message("nobody listens"));
}
