//! Interactive chat example wired to the `mcp-server-time` MCP server.
//!
//! Requires:
//! - `ANTHROPIC_API_KEY` in the environment
//! - `uvx` on `PATH` (`pip install uv`)
//!
//! Run with: `cargo run -p mcp-time`

use ailoop::{Conversation, RetryingModel, StreamChunk};
use ailoop_anthropic::AnthropicClient;
use ailoop_mcp::McpConnection;
use futures::StreamExt;
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mcp = McpConnection::builder("time")
        .command("uvx")
        .args(["mcp-server-time"])
        .connect()
        .await?;

    println!(
        "Connected to MCP server: {}",
        mcp.server_name().unwrap_or_else(|| "<unknown>".into())
    );

    let model = RetryingModel::new(AnthropicClient::from_env()?.model("claude-sonnet-4-6"));
    let mut builder = Conversation::builder(model)
        .system_prompt("You are a helpful assistant. Use the available tools to answer time- and timezone-related questions.");

    for tool in mcp.list_tools().await? {
        builder = builder.tool_dyn(tool);
    }

    let mut chat = builder.build()?;

    println!("Active tools: {:?}", chat.active_tool_names());
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
