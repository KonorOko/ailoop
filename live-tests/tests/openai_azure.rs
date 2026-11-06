//! `ailoop-openai` against a real Azure OpenAI resource, through
//! [`OpenAIClient::azure_from_env`]. Costs a few tokens per run.
//!
//! - `AZURE_OPENAI_ENDPOINT`: the resource endpoint,
//!   `https://my-resource.openai.azure.com`.
//! - `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_TOKEN` (exactly one).
//! - `AILOOP_AZURE_DEPLOYMENT`: the deployment to call.
//!
//! ```text
//! cargo test -p ailoop-live-tests --test openai_azure -- --ignored --nocapture
//! ```

use ailoop::{Conversation, ProviderError, ailoop_tool};
use ailoop_core::ProviderErrorKind;
use ailoop_openai::OpenAIClient;

fn deployment() -> String {
    std::env::var("AILOOP_AZURE_DEPLOYMENT").expect("set AILOOP_AZURE_DEPLOYMENT")
}

fn client() -> OpenAIClient {
    match OpenAIClient::azure_from_env() {
        Ok(client) => client,
        Err(e) => panic!("{e}"),
    }
}

#[tokio::test]
#[ignore = "needs an Azure OpenAI resource"]
async fn azure_streams_a_reply() {
    let mut chat = Conversation::builder(client().model(deployment()))
        .build()
        .unwrap();
    let outcome = match chat
        .run("Answer with one word: what colour is the sky on a clear day?")
        .await
    {
        Ok(outcome) => outcome,
        Err(e) => panic!("{e}"),
    };
    println!("reply: {:?}", outcome.final_text);
    println!("usage: {:?}", outcome.usage);
    assert!(outcome.final_text.is_some_and(|t| !t.trim().is_empty()));
    assert!(outcome.usage.input_tokens > 0);
}

#[ailoop_tool(description = "Return the number of notes in the user's notebook")]
async fn count_notes() -> u32 {
    42
}

#[tokio::test]
#[ignore = "needs an Azure OpenAI resource"]
async fn azure_calls_a_tool() {
    let mut chat = Conversation::builder(client().model(deployment()))
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

#[tokio::test]
#[ignore = "needs an Azure OpenAI resource"]
async fn azure_unknown_deployment_is_model_not_found() {
    use ailoop::CompletionModel;
    let model = client().model("ailoop-no-such-deployment");
    let err = match model
        .chat_stream(ailoop::ChatRequest::new(
            vec![ailoop::Message::user("hi")],
            16,
        ))
        .await
    {
        Ok(_) => panic!("the deployment should not exist"),
        Err(e) => e,
    };
    println!("{:?}: {err}", err.kind());
    assert_eq!(err.kind(), ProviderErrorKind::ModelNotFound);
}
