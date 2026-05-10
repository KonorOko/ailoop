use ailoop::{Conversation, RetryingModel, ailoop_tool};
use ailoop_anthropic::AnthropicClient;

#[ailoop_tool(description = "Sum two numbers")]
async fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `RetryingModel` retries setup-time failures (5xx, 429, transport
    // hiccups) with exponential backoff that honours `Retry-After`.
    // It is opt-in — wrap whichever provider model you use.
    let model = RetryingModel::new(AnthropicClient::from_env()?.model("claude-sonnet-4-6"));
    let mut chat = Conversation::builder(model)
        .system_prompt("You are a helpful math assistant.")
        .tool(Add)
        .build()?;

    // The model picks up that it needs the `add` tool to answer this,
    // calls it, and folds the result into its final reply. Everything
    // — tool dispatch, history bookkeeping, multi-turn loop — happens
    // inside `chat.run`.
    let outcome = chat.run("What is 17 + 28?").await?;
    println!("{}", outcome.final_text.unwrap_or_default());

    Ok(())
}
