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
    #[error("MCP transport creation failed: {0}")]
    TransportCreation(String),

    #[error("MCP service error: {0}")]
    Service(String),
}
