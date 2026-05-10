//! Minimal example wired to the `mcp-server-time` MCP server.
//!
//! Requires:
//! - `ANTHROPIC_API_KEY` in the environment
//! - `uvx` on `PATH` (`pip install uv`)
//!
//! Run with: `cargo run -p mcp-time`

use ailoop::{Conversation, RetryingModel};
use ailoop_anthropic::AnthropicClient;
use ailoop_mcp::McpConnection;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mcp = McpConnection::builder("time")
        .command("uvx")
        .args(["mcp-server-time"])
        .connect()
        .await?;

    let model = RetryingModel::new(AnthropicClient::from_env()?.model("claude-sonnet-4-6"));
    let mut builder = Conversation::builder(model).system_prompt(
        "You are a helpful assistant. Use the available tools to answer time- and \
         timezone-related questions.",
    );

    // Every tool the MCP server exposes is registered as `Arc<dyn ToolDyn>`,
    // discoverable by name in the running conversation.
    for tool in mcp.list_tools().await? {
        builder = builder.tool_dyn(tool);
    }

    let mut chat = builder.build()?;

    let outcome = chat.run("What time is it in Tokyo right now?").await?;
    println!("{}", outcome.final_text.unwrap_or_default());

    Ok(())
}
