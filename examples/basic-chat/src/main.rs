use ailoop::{Conversation, StreamChunk};
use ailoop_anthropic::AnthropicClient;
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = AnthropicClient::from_env()?.model("claude-sonnet-4-6");
    let mut chat = Conversation::builder(model).build()?;

    println!("Type a message and press Enter. Type 'exit' or send EOF to quit.");

    loop {
        print!("\nYou: ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() || input == "exit" {
            break;
        }

        print!("Assistant: ");
        io::stdout().flush().ok();

        let mut stream = chat.stream(input).await?;
        while let Some(chunk) = stream.next().await {
            if let StreamChunk::TextDelta { delta } = chunk? {
                print!("{delta}");
                io::stdout().flush().ok();
            }
        }
        println!();
    }

    Ok(())
}
