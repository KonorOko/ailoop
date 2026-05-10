use thiserror::Error;

/// Errors surfaced by the MCP adapter at connection / discovery time.
///
/// Per-tool runtime errors (transport drops, server-side `isError: true`,
/// schema mismatches) are *not* in here — they are mapped to
/// [`ailoop_core::ToolResultContent::Error`] inside the tool wrapper so
/// the model sees the failure as a tool reply, mirroring the
/// `SubAgentTool` policy.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum McpError {
    /// Setting up the underlying transport failed before the
    /// `initialize` handshake could run — typically a missing
    /// command, executable not on `PATH`, or
    /// [`McpConnectionBuilder::connect`](crate::McpConnectionBuilder::connect)
    /// called without a [`command`](crate::McpConnectionBuilder::command)
    /// having been set.
    #[error("MCP transport creation failed: {0}")]
    TransportCreation(String),

    /// `rmcp` service-layer failure during handshake or discovery
    /// (`tools/list`). Carries the underlying error rendered as a
    /// string to keep [`McpError`] non-generic and object-safe; the
    /// `rmcp` types are not part of this crate's public API.
    #[error("MCP service error: {0}")]
    Service(String),
}
