//! Integration tests against `mcp-server-time` (Python).
//!
//! These tests spawn a real MCP server child process via `uvx`, so they
//! are gated behind `#[ignore]`. Run them locally with:
//!
//! ```text
//! cargo test -p ailoop-mcp -- --ignored
//! ```
//!
//! Requires `uvx` on `PATH` (install via `pip install uv`).

use ailoop_core::ToolTag;
use ailoop_mcp::McpConnection;
use serde_json::json;

async fn connect() -> McpConnection {
    McpConnection::builder("time")
        .command("uvx")
        .args(["mcp-server-time"])
        .connect()
        .await
        .expect("connect to mcp-server-time")
}

#[tokio::test]
#[ignore = "requires `uvx` and network access to install mcp-server-time"]
async fn connects_to_mcp_server_time() {
    let mcp = connect().await;

    assert_eq!(mcp.server_label(), "time");
    let name = mcp
        .server_name()
        .expect("handshake should populate server name");
    assert!(!name.is_empty(), "server name should be non-empty");
}

#[tokio::test]
#[ignore = "requires `uvx` and network access to install mcp-server-time"]
async fn list_tools_returns_prefixed_names_with_default_tags() {
    let mcp = connect().await;
    let tools = mcp.list_tools().await.expect("list_tools");

    assert!(!tools.is_empty(), "mcp-server-time should expose tools");

    for tool in &tools {
        let def = tool.tool_definition();
        assert!(
            def.name.starts_with("mcp__time__"),
            "expected name to start with prefix, got {}",
            def.name
        );
        assert!(
            def.tags.contains(&ToolTag::Network),
            "every MCP tool should be tagged Network"
        );
        assert!(
            def.tags.contains(&ToolTag::Custom("mcp".into())),
            "every MCP tool should be tagged Custom(\"mcp\")"
        );
        assert!(
            def.input_schema.is_object(),
            "input_schema should be a JSON object"
        );
    }
}

#[tokio::test]
#[ignore = "requires `uvx` and network access to install mcp-server-time"]
async fn calls_get_current_time_tool() {
    let mcp = connect().await;
    let tools = mcp.list_tools().await.expect("list_tools");

    let tool = tools
        .iter()
        .find(|t| t.tool_definition().name.contains("current_time"))
        .expect("mcp-server-time should expose a get_current_time-style tool");

    let result = tool.call(json!({"timezone": "UTC"})).await;
    assert!(!result.is_error, "unexpected error reply: {result:?}");
    let text = result.as_text().expect("expected a text payload");
    assert!(!text.is_empty(), "expected a non-empty text payload");
}

#[tokio::test]
#[ignore = "requires `uvx` and network access to install mcp-server-time"]
async fn invalid_args_surface_as_tool_error_not_engine_error() {
    let mcp = connect().await;
    let tools = mcp.list_tools().await.expect("list_tools");
    let tool = tools
        .iter()
        .find(|t| t.tool_definition().name.contains("current_time"))
        .expect("get_current_time tool");

    // Missing required `timezone` argument.
    let result = tool.call(json!({})).await;
    assert!(
        result.is_error,
        "expected is_error=true, got {:?}",
        result
    );
}
