use ailoop::{Conversation, RetryingModel};
use ailoop_anthropic::AnthropicClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `RetryingModel` retries setup-time failures (5xx, 429, transport
    // hiccups) with exponential backoff that honours `Retry-After`.
    // It is opt-in — wrap whichever provider model you use.
    let model = RetryingModel::new(AnthropicClient::from_env()?.model("claude-sonnet-4-6"));
    let mut chat = Conversation::builder(model).build()?;

    let outcome = chat
        .run("Explain Rust ownership in one short sentence.")
        .await?;
    println!("{}", outcome.final_text.unwrap_or_default());

    Ok(())
}
