use ailoop::{Conversation, StreamChunk, Tool, ToolDefinition};
use ailoop_anthropic::AnthropicClient;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::json;
use std::convert::Infallible;
use std::io::{self, Write};

struct Add;

#[derive(Deserialize)]
struct AddArgs {
    a: f64,
    b: f64,
}

impl Tool for Add {
    const NAME: &'static str = "add";

    type Args = AddArgs;
    type Output = String;
    type Error = Infallible;

    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            Self::NAME,
            "Add two numbers and return the sum.",
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number", "description": "First addend." },
                    "b": { "type": "number", "description": "Second addend." }
                },
                "required": ["a", "b"]
            }),
            vec![],
        )
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        async move { Ok(format!("{}", args.a + args.b)) }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = AnthropicClient::from_env()?.model("claude-sonnet-4-6");
    let mut chat = Conversation::builder(model)
        .system_prompt(
            "You are a helpful math assistant. \
             Use the `add` tool whenever the user asks you to sum two numbers.",
        )
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
