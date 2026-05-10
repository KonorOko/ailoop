//! Engine-level integration tests for `AntiLoop`. The unit-style tests
//! in `anti_loop.rs` itself would call hooks directly, which cannot
//! catch a regression where the engine stops invoking the right hook on
//! the right chunk; these tests drive `run_chat` end-to-end against a
//! `ScriptedModel` so the full middleware contract is exercised.

use std::sync::Arc;

use ailoop::{AntiLoop, Message, ToolDefinition, ToolResultContent, run_chat};
use ailoop_core::testing::ScriptedModel;
use ailoop_core::{FinishReason, RunConfig, StreamChunk, Usage};
use ailoop_tools::{ToolRegistry, registry::ToolDyn};
use futures::StreamExt;
use serde_json::{Value, json};

struct GetWeather;

#[async_trait::async_trait]
impl ToolDyn for GetWeather {
    fn name(&self) -> String {
        "get_weather".into()
    }
    fn tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".into(),
            description: "stub".into(),
            input_schema: json!({"type":"object","properties":{},"required":[]}),
            tags: vec![],
            cache_control: None,
        }
    }
    async fn call(&self, _: Value) -> ToolResultContent {
        ToolResultContent::Text("sunny".into())
    }
}

/// One turn that emits a tool call with the given id/args plus a text
/// preamble. The engine treats `FinishReason::ToolUse` as "keep
/// iterating", so chaining N of these in a `ScriptedModel` produces an
/// N-iteration run.
fn tool_turn(id: &str, args: Value, text: Option<&str>) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();
    if let Some(t) = text {
        chunks.push(StreamChunk::TextDelta { delta: t.into() });
    }
    chunks.push(StreamChunk::ToolCallStart {
        id: id.into(),
        name: "get_weather".into(),
    });
    chunks.push(StreamChunk::ToolCallEnd {
        id: id.into(),
        name: "get_weather".into(),
        args,
    });
    chunks.push(StreamChunk::TurnFinished {
        reason: FinishReason::ToolUse,
        usage: Usage::default(),
        service_tier: None,
    });
    chunks
}

fn registry_with_weather() -> ToolRegistry {
    let mut r = ToolRegistry::new();
    r.register(Arc::new(GetWeather)).unwrap();
    r
}

async fn collect_finish_reason(model: ScriptedModel, mw: Arc<AntiLoop>) -> FinishReason {
    let registry = registry_with_weather();
    let config = RunConfig {
        middlewares: vec![mw],
        ..RunConfig::default()
    };
    let stream = run_chat(&model, vec![Message::user("hi")], &registry, config)
        .await
        .expect("run_chat should start");
    let chunks: Vec<_> = stream.collect().await;
    chunks
        .into_iter()
        .find_map(|c| match c {
            Ok(StreamChunk::RunFinished { reason, .. }) => Some(reason),
            _ => None,
        })
        .expect("run should emit RunFinished")
}

#[tokio::test]
async fn tool_call_loop_aborts_on_third_identical_call() {
    let turns = (0..3).map(|i| tool_turn(&format!("toolu_{i}"), json!({}), None));
    let model = ScriptedModel::new(turns);
    let mw = Arc::new(AntiLoop::new());

    let reason = collect_finish_reason(model, mw).await;
    match reason {
        FinishReason::Aborted(r) => {
            assert!(
                r.starts_with("anti-loop: tool 'get_weather' called"),
                "got: {r}"
            );
            assert!(r.contains("3 times"), "got: {r}");
        }
        other => panic!("expected Aborted, got {other:?}"),
    }
}

#[tokio::test]
async fn tool_call_loop_does_not_fire_with_distinct_args() {
    let turns = vec![
        tool_turn("toolu_a", json!({"q": "a"}), None),
        tool_turn("toolu_b", json!({"q": "b"}), None),
        tool_turn("toolu_c", json!({"q": "c"}), None),
        // Final turn ends the run cleanly so the iteration count stays bounded.
        vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        }],
    ];
    let model = ScriptedModel::new(turns);
    let mw = Arc::new(AntiLoop::new());

    let reason = collect_finish_reason(model, mw).await;
    assert!(
        matches!(reason, FinishReason::EndTurn),
        "different args must not trip the detector, got {reason:?}",
    );
}

#[tokio::test]
async fn tool_call_loop_does_not_fire_below_threshold() {
    let turns = vec![
        tool_turn("toolu_a", json!({}), None),
        tool_turn("toolu_b", json!({}), None),
        vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        }],
    ];
    let model = ScriptedModel::new(turns);
    let mw = Arc::new(AntiLoop::new());

    let reason = collect_finish_reason(model, mw).await;
    assert!(
        matches!(reason, FinishReason::EndTurn),
        "two identical calls must not trip default threshold of 3, got {reason:?}",
    );
}

#[tokio::test]
async fn text_loop_aborts_after_three_identical_turns() {
    // Each turn carries identical assistant text plus a tool call (with
    // distinct args / ids so the tool-call detector cannot also fire).
    let turns = vec![
        tool_turn("toolu_a", json!({"q": "a"}), Some("checking now")),
        tool_turn("toolu_b", json!({"q": "b"}), Some("checking now")),
        tool_turn("toolu_c", json!({"q": "c"}), Some("checking now")),
    ];
    let model = ScriptedModel::new(turns);
    let mw = Arc::new(AntiLoop::new());

    let reason = collect_finish_reason(model, mw).await;
    match reason {
        FinishReason::Aborted(r) => {
            assert!(
                r.starts_with("anti-loop: assistant text repeated identically across"),
                "got: {r}"
            );
            assert!(r.contains("3 turns"), "got: {r}");
        }
        other => panic!("expected Aborted, got {other:?}"),
    }
}

#[tokio::test]
async fn state_clears_between_runs_so_second_run_re_arms() {
    let mw = Arc::new(AntiLoop::new());

    // First run: 3 identical tool calls -> aborts.
    let model1 =
        ScriptedModel::new((0..3).map(|i| tool_turn(&format!("toolu_{i}"), json!({}), None)));
    let reason1 = collect_finish_reason(model1, mw.clone()).await;
    assert!(
        matches!(reason1, FinishReason::Aborted(_)),
        "first run should abort on the loop, got {reason1:?}",
    );

    // Second run on a fresh model: same 3 identical calls. Per-run state
    // must have been wiped (on_run_finished) so the new run abort re-fires
    // at the third call rather than slipping through (or, conversely,
    // mis-firing earlier because the previous streak leaked over).
    let model2 =
        ScriptedModel::new((0..3).map(|i| tool_turn(&format!("toolu2_{i}"), json!({}), None)));
    let reason2 = collect_finish_reason(model2, mw).await;
    match reason2 {
        FinishReason::Aborted(r) => assert!(r.contains("3 times"), "got: {r}"),
        other => panic!("expected Aborted on second run, got {other:?}"),
    }
}

#[tokio::test]
async fn threshold_zero_disables_tool_call_detector() {
    let turns: Vec<Vec<StreamChunk>> = (0..4)
        .map(|i| tool_turn(&format!("toolu_{i}"), json!({}), None))
        .chain(std::iter::once(vec![StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        }]))
        .collect();
    let model = ScriptedModel::new(turns);
    let mw = Arc::new(AntiLoop::new().with_max_repeated_tool_calls(0));

    let reason = collect_finish_reason(model, mw).await;
    assert!(
        matches!(reason, FinishReason::EndTurn),
        "threshold 0 must disable the detector even for many identical calls, got {reason:?}",
    );
}

#[tokio::test]
async fn custom_text_predicate_starts_with_fires() {
    // Custom predicate: "current text starts with the previous turn's text".
    // Three turns where each turn's text is a strict superset of the prior
    // turn's text triggers the streak.
    let turns = vec![
        tool_turn("toolu_a", json!({"q": "a"}), Some("step")),
        tool_turn("toolu_b", json!({"q": "b"}), Some("step done")),
        tool_turn("toolu_c", json!({"q": "c"}), Some("step done extra")),
    ];
    let model = ScriptedModel::new(turns);
    let mw = Arc::new(
        AntiLoop::new()
            .with_text_predicate(|prev, current| current.starts_with(prev))
            // Disable the tool-call detector so only the text path can fire.
            .with_max_repeated_tool_calls(0),
    );

    let reason = collect_finish_reason(model, mw).await;
    match reason {
        FinishReason::Aborted(r) => assert!(
            r.starts_with("anti-loop: assistant text repeated identically across"),
            "got: {r}"
        ),
        other => panic!("expected Aborted from custom predicate, got {other:?}"),
    }
}

/// Exercises the public surface end-to-end: register `AntiLoop` via the
/// `ConversationBuilder` middleware entry point and ensure the wiring
/// keeps `ToolDecision::Continue` as the default for non-looping calls.
#[tokio::test]
async fn anti_loop_wires_through_conversation_builder() {
    use ailoop::Conversation;

    let model = ScriptedModel::new([vec![StreamChunk::TurnFinished {
        reason: FinishReason::EndTurn,
        usage: Usage::default(),
        service_tier: None,
    }]]);

    let mut conv = Conversation::builder(model)
        .middleware(Arc::new(AntiLoop::new()))
        .build()
        .expect("builder should succeed");
    let outcome = conv.run("hi").await.expect("run should complete");
    assert!(matches!(outcome.finish_reason, FinishReason::EndTurn));
}
