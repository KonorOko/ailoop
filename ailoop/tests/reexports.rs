use ailoop::{CacheControl, SystemBlock, SystemPrompt, ToolResultBlock};

#[test]
fn reexports_compile() {
    let _ = CacheControl::Ephemeral;
    let _ = SystemBlock::new("x");
    let _: SystemPrompt = "hi".into();
    let _ = ToolResultBlock::text("ok");
}
