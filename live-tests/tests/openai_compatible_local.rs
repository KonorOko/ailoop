//! `ailoop-openai` against a real local server (Ollama or LM Studio).
//! Free: nothing here talks to a paid API unless you point it at one.
//!
//! - `AILOOP_OPENAI_BASE_URL`: defaults to Ollama, `http://localhost:11434/v1`.
//!   LM Studio is `http://localhost:1234/v1`.
//! - `AILOOP_OPENAI_MODEL`: defaults to the first model the server lists.
//! - `AILOOP_OPENAI_API_KEY`: optional; local servers need none.
//!
//! ```text
//! cargo test -p ailoop-live-tests --test openai_compatible_local -- --ignored --nocapture
//! ```

use ailoop::{Conversation, ailoop_tool};
use ailoop_openai::{OpenAIClient, OpenAIError};

fn client() -> OpenAIClient {
    let base = std::env::var("AILOOP_OPENAI_BASE_URL")
        .unwrap_or_else(|_| OpenAIClient::OLLAMA_BASE_URL.into());
    OpenAIClient::new(base).maybe_api_key(std::env::var("AILOOP_OPENAI_API_KEY").ok())
}

async fn model_name(client: &OpenAIClient) -> Result<String, OpenAIError> {
    if let Ok(name) = std::env::var("AILOOP_OPENAI_MODEL") {
        return Ok(name);
    }
    let models = client.list_models().await?;
    Ok(models
        .first()
        .expect("the server lists no models; set AILOOP_OPENAI_MODEL")
        .id
        .clone())
}

#[tokio::test]
#[ignore = "needs a local OpenAI-compatible server"]
async fn local_lists_models_and_capabilities() {
    let client = client();
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(e) => panic!("{:?}: {e}", e.category()),
    };
    for m in &models {
        let caps = client.model_capabilities(&m.id).await.unwrap();
        println!("{} images={:?} tools={:?}", m.id, caps.images, caps.tools);
    }
    assert!(!models.is_empty());
}

#[tokio::test]
#[ignore = "needs a local OpenAI-compatible server"]
async fn local_streams_a_reply() {
    let client = client();
    let name = model_name(&client).await.unwrap();
    println!("model: {name}");
    let mut chat = Conversation::builder(client.model(name)).build().unwrap();
    let outcome = chat
        .run("Answer with one word: what colour is the sky on a clear day?")
        .await
        .unwrap();
    println!("reply: {:?}", outcome.final_text);
    println!("usage: {:?}", outcome.usage);
    assert!(outcome.final_text.is_some_and(|t| !t.trim().is_empty()));
}

#[ailoop_tool(description = "Return the number of notes in the user's notebook")]
async fn count_notes() -> u32 {
    42
}

#[tokio::test]
#[ignore = "needs a local OpenAI-compatible server and a model with tools"]
async fn local_calls_a_tool() {
    let client = client();
    let name = model_name(&client).await.unwrap();
    let mut chat = Conversation::builder(client.model(name))
        .tool(CountNotes)
        .build()
        .unwrap();
    let outcome = chat
        .run("How many notes do I have? Use the tool, then answer with the number.")
        .await
        .unwrap();
    println!("reply: {:?}", outcome.final_text);
    assert!(outcome.final_text.unwrap_or_default().contains("42"));
}
