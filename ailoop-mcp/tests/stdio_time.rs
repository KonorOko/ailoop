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

use ailoop_mcp::McpConnection;

#[tokio::test]
#[ignore = "requires `uvx` and network access to install mcp-server-time"]
async fn connects_to_mcp_server_time() {
    let mcp = McpConnection::builder("time")
        .command("uvx")
        .args(["mcp-server-time"])
        .connect()
        .await
        .expect("connect to mcp-server-time");

    assert_eq!(mcp.server_label(), "time");
    let name = mcp
        .server_name()
        .expect("handshake should populate server name");
    assert!(!name.is_empty(), "server name should be non-empty");
}
