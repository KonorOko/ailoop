use ailoop::{Conversation, StreamChunk, ailoop_tool};
use ailoop_anthropic::AnthropicClient;
use futures::StreamExt;
use std::io::{self, Write};

#[ailoop_tool(description = "Sum two numbers")]
async fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = AnthropicClient::from_env()?.model("claude-sonnet-4-6");
    let mut chat = Conversation::builder(model)
        .system_prompt("You are a helpful math assistant.")
        .tool(Add)
        .build()?;

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

        let mut stream = chat.stream(input).await?;
        while let Some(chunk) = stream.next().await {
            match chunk? {
                StreamChunk::TextDelta { delta } => {
                    print!("{delta}");
                    io::stdout().flush().ok();
                }
                StreamChunk::ToolCallStart { name, .. } => {
                    print!("\n[calling {name}]");
                    io::stdout().flush().ok();
                }
                StreamChunk::ToolResult { content, .. } => {
                    println!("\n[result: {content:?}]");
                }
                _ => {}
            }
        }
        println!();
    }

    Ok(())
}
