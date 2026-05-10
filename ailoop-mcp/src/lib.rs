//! MCP (Model Context Protocol) adapter for ailoop.
//!
//! Wraps the official `rmcp` Rust SDK so MCP tools surface as
//! [`ailoop_tools::ToolDyn`] implementations registrable on a
//! [`ConversationBuilder`] via `tool_dyn`.
//!
//! [`ConversationBuilder`]: https://docs.rs/ailoop

mod connection;
mod errors;
mod naming;
mod tool;

pub use connection::{McpConnection, McpConnectionBuilder};
pub use errors::McpError;
pub use tool::McpTool;
