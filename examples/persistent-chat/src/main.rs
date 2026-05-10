use ailoop::{
    Conversation, ConversationBuilder, HistoryStore, JsonFileHistoryStore, RetryingModel,
};
use ailoop_anthropic::AnthropicClient;

/// Reads the prompt from `argv[1]` (so each run is one turn), loads
/// any prior conversation from disk, runs the turn, and saves the new
/// snapshot back. Run twice in a row and the second invocation sees
/// the first turn's history.
///
/// ```text
/// $ cargo run -p persistent-chat -- "Hi, my name is Ada."
/// $ cargo run -p persistent-chat -- "What did I just tell you my name was?"
/// ```
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args()
        .nth(1)
        .ok_or("usage: persistent-chat <prompt>")?;

    let store = JsonFileHistoryStore::new("chat-history.json");
    let prior = store.load().await?;

    let model = RetryingModel::new(AnthropicClient::from_env()?.model("claude-sonnet-4-6"));

    // First run: no snapshot on disk, start a fresh conversation.
    // Subsequent runs: rebuild from the snapshot so the model sees
    // every prior turn.
    let mut chat: Conversation<_> = match prior {
        Some(snapshot) => ConversationBuilder::from_snapshot(model, snapshot).build()?,
        None => Conversation::builder(model).build()?,
    };

    let outcome = chat.run(prompt).await?;
    println!("{}", outcome.final_text.unwrap_or_default());

    // Persist the post-turn state. Next invocation will pick this up.
    store.save(&chat.snapshot()).await?;

    Ok(())
}
