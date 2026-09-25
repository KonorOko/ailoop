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
