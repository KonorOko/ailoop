use ailoop::{CacheControl, ContinueDecision, SystemBlock, SystemPrompt, ToolResultBlock};

#[test]
fn reexports_compile() {
    let _ = CacheControl::Ephemeral;
    let _ = SystemBlock::new("x");
    let _: SystemPrompt = "hi".into();
    let _ = ToolResultBlock::text("ok");
    let _ = ContinueDecision::continue_with("keep going");
    let _ = ContinueDecision::default();
}
