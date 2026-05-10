//! MCP (Model Context Protocol) adapter for ailoop.
//!
//! Thin wrapper around the official [`rmcp`] Rust SDK that exposes
//! tools discovered over the MCP `tools/*` surface as
//! [`ailoop_tools::ToolDyn`] instances. Register the result with
//! [`ConversationBuilder::tool_dyn`](https://docs.rs/ailoop) to put
//! the model in front of any MCP server.
//!
//! ## Scope (1.0)
//!
//! - Stdio transport via `tokio::process::Command`.
//! - `tools/list` discovery + `tools/call` invocation.
//! - Engine-facing names follow the Claude Desktop convention
//!   `mcp__<server_label>__<tool_name>`. Characters outside
//!   `[A-Za-z0-9_-]` are sanitized to `_`; over-long composed names
//!   truncate-with-deterministic-hash so distinct long names do not
//!   collide.
//! - Default tagging `[ToolTag::Network, ToolTag::Custom("mcp")]` so
//!   `with_capabilities` and `with_approval_for_tags` can scope MCP
//!   tools without per-tool boilerplate.
//!
//! Resources, prompts, sampling (server → client), SSE / Streamable
//! HTTP transports, `tools/list_changed` notifications, and
//! cooperative `notifications/cancelled` are out of scope for 1.0 and
//! land as follow-up iterations on `ailoop-mcp` driven by real use
//! cases.
//!
//! ## Mini-index
//!
//! - [`McpConnection`] / [`McpConnectionBuilder`] — spawn an MCP
//!   server child process and complete the `initialize` handshake.
//! - [`McpTool`] — `ToolDyn` wrapper around one discovered tool.
//! - [`McpError`] — failure surface of connection setup and
//!   discovery. Per-tool runtime failures map to a
//!   [`ToolResultContent`] with `is_error: true` inside the wrapper
//!   instead.
//!
//! [`rmcp`]: https://docs.rs/rmcp
//! [`ToolResultContent`]: ailoop_core::ToolResultContent

#![deny(missing_docs)]

mod connection;
mod errors;
mod naming;
mod tool;

pub use connection::{McpConnection, McpConnectionBuilder};
pub use errors::McpError;
pub use tool::McpTool;
