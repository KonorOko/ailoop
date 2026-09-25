use ailoop::{
    CacheControl, ContinueDecision, FinishReason, RunConfig, RunErrorInfo, RunFinishedInfo, RunId,
    RunStartInfo, StepId, StepInfo, SystemBlock, SystemPrompt, ToolResultBlock, TurnEndInfo, Usage,
};

#[test]
fn reexports_compile() {
    let _ = CacheControl::Ephemeral;
    let _ = SystemBlock::new("x");
    let _: SystemPrompt = "hi".into();
    let _ = ToolResultBlock::text("ok");
    let _ = ContinueDecision::continue_with("keep going");
    let _ = ContinueDecision::default();
}

/// The hook context structs are built from the facade alone, the way a
/// user unit-tests a middleware.
#[test]
fn hook_contexts_are_constructible_from_the_facade() {
    let run_id = RunId::new();
    let step_id = StepId::new();
    let config = RunConfig::default();
    let usage = Usage::default();
    let reason = FinishReason::EndTurn;
    let err = std::io::Error::other("boom");

    let start = RunStartInfo::new(&run_id, &[], &config);
    assert_eq!(start.config.max_iterations, config.max_iterations);
    let step = StepInfo::new(run_id.clone(), step_id.clone());
    assert_eq!(step.step_id, step_id);
    let finished = RunFinishedInfo::new(&run_id, &reason, &usage, &[]);
    assert!(finished.new_messages.is_empty());
    let turn = TurnEndInfo::new(&run_id, &step_id, &reason, &[]);
    assert_eq!(turn.run_id, &run_id);
    let error = RunErrorInfo::new(&run_id, &err, &usage, &[]);
    assert_eq!(error.error.to_string(), "boom");
}

/// The compaction types a custom `CompactionStrategy` returns, and the
/// stats `History` reports, are nameable from the facade.
#[tokio::test]
async fn compaction_types_are_reexported() {
    let out = ailoop::CompactionOutput::new(vec![ailoop::Message::user("hi")], vec![false]);
    assert_eq!(out.messages.len(), out.pinned.len());
    assert!(!ailoop::DEFAULT_SUMMARIZER_PROMPT.is_empty());

    let mut history = ailoop::History::builder(1_000).preserve_n_last(1).build();
    history.add_message(ailoop::Message::user("one"));
    history.add_message(ailoop::Message::assistant_text("two"));
    history.add_message(ailoop::Message::user("three"));
    let stats: ailoop::CompactionStats = history.force_compact().await.unwrap();
    assert_eq!(stats.strategy, "truncate");
    assert!(stats.after <= stats.before);
}

/// A user-defined strategy, written with the facade's `async_trait`
/// re-export and returning the facade's `CompactionOutput`.
struct KeepLast;

#[ailoop::async_trait]
impl ailoop::CompactionStrategy for KeepLast {
    fn name(&self) -> &'static str {
        "keep_last"
    }

    async fn compact(
        &self,
        messages: &[ailoop::Message],
        pinned: &[bool],
        _preserve_n_last: usize,
    ) -> Result<ailoop::CompactionOutput, ailoop::CompactionError> {
        let n = messages.len() - 1;
        Ok(ailoop::CompactionOutput::new(
            messages[n..].to_vec(),
            pinned[n..].to_vec(),
        ))
    }
}

#[tokio::test]
async fn async_trait_reexport_implements_compaction_strategy() {
    let mut history = ailoop::History::builder(1_000)
        .strategy(Box::new(KeepLast))
        .build();
    history.add_message(ailoop::Message::user("one"));
    history.add_message(ailoop::Message::user("two"));
    let stats = history.force_compact().await.unwrap();
    assert_eq!(stats.strategy, "keep_last");
    assert_eq!(history.messages().len(), 1);
}

struct AlwaysContinue;

#[ailoop::async_trait]
impl ailoop::ChatMiddleware for AlwaysContinue {
    async fn on_turn_end(&self, _turn: &TurnEndInfo<'_>) -> ContinueDecision {
        ContinueDecision::continue_with("again")
    }
}

#[tokio::test]
async fn async_trait_reexport_implements_chat_middleware() {
    use ailoop::ChatMiddleware;

    let run_id = RunId::new();
    let step_id = StepId::new();
    let reason = FinishReason::EndTurn;
    let turn = TurnEndInfo::new(&run_id, &step_id, &reason, &[]);
    let decision = AlwaysContinue.on_turn_end(&turn).await;
    assert!(matches!(decision, ContinueDecision::Continue { .. }));
}

/// With the `testing` feature, a whole run can be scripted from the
/// facade alone.
#[cfg(feature = "testing")]
#[tokio::test]
async fn scripted_model_is_reachable_through_the_testing_feature() {
    use ailoop::testing::ScriptedModel;
    use ailoop::{Conversation, StreamChunk};

    let model = ScriptedModel::new([vec![
        StreamChunk::TextDelta {
            delta: "hello".into(),
        },
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::new(3, 1),
            service_tier: None,
        },
    ]]);
    let mut chat = Conversation::builder(model).build().unwrap();
    let outcome = chat.run("hi").await.unwrap();
    assert_eq!(outcome.final_text.as_deref(), Some("hello"));
    assert_eq!(outcome.usage.input_tokens, 3);
}
