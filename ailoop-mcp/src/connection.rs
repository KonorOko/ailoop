use std::sync::Arc;

use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::TokioChildProcess;
use tokio::process::Command;

use crate::errors::McpError;

/// Live connection to an MCP server over stdio.
///
/// Wraps the `rmcp` peer behind an [`Arc`] so the connection survives
/// for as long as any tool wrapper derived from it is alive. Dropping
/// the last reference closes the underlying child process.
pub struct McpConnection {
    server_label: String,
    inner: Arc<RunningService<RoleClient, ()>>,
}

impl McpConnection {
    /// Start configuring a new connection. The `server_label` is the
    /// human-friendly identifier used as the namespace prefix when
    /// discovered tools are exposed to the engine
    /// (`mcp__<server_label>__<tool_name>`); pick something short and
    /// unique within the conversation (e.g. `"time"`, `"fetch"`).
    pub fn builder(server_label: impl Into<String>) -> McpConnectionBuilder {
        McpConnectionBuilder {
            server_label: server_label.into(),
            command: None,
            args: Vec::new(),
            envs: Vec::new(),
        }
    }

    /// Label supplied at connection time. The naming convention for
    /// discovered tools is `mcp__<server_label>__<tool_name>`.
    pub fn server_label(&self) -> &str {
        &self.server_label
    }

    /// Server-reported metadata returned during the `initialize`
    /// handshake (server name, version, declared capabilities).
    /// Returns `None` if the peer never completed initialization.
    pub fn server_name(&self) -> Option<String> {
        self.inner
            .peer_info()
            .map(|info| info.server_info.name.clone())
    }
}

/// Fluent builder for [`McpConnection`]. Spawn-time configuration only —
/// every option here goes into the [`tokio::process::Command`] that
/// launches the MCP server.
pub struct McpConnectionBuilder {
    server_label: String,
    command: Option<String>,
    args: Vec<String>,
    envs: Vec<(String, String)>,
}

impl McpConnectionBuilder {
    /// Executable to launch (e.g. `"uvx"`, `"npx"`, `"python"`).
    pub fn command(mut self, cmd: impl Into<String>) -> Self {
        self.command = Some(cmd.into());
        self
    }

    /// Arguments passed verbatim to the executable. Successive calls
    /// replace the list.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Add a single environment variable on top of the parent's
    /// environment. Calls accumulate.
    pub fn env(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.envs.push((key.into(), val.into()));
        self
    }

    /// Spawn the MCP server child process and complete the
    /// `initialize` handshake. Returns a connected [`McpConnection`]
    /// ready for tool discovery.
    pub async fn connect(self) -> Result<McpConnection, McpError> {
        let exe = self
            .command
            .ok_or_else(|| McpError::TransportCreation("missing command".into()))?;

        let mut command = Command::new(&exe);
        for arg in &self.args {
            command.arg(arg);
        }
        for (k, v) in &self.envs {
            command.env(k, v);
        }

        let transport = TokioChildProcess::new(command)
            .map_err(|e| McpError::TransportCreation(e.to_string()))?;

        let inner = ()
            .serve(transport)
            .await
            .map_err(|e| McpError::Service(e.to_string()))?;

        Ok(McpConnection {
            server_label: self.server_label,
            inner: Arc::new(inner),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_stores_label() {
        let b = McpConnection::builder("time");
        assert_eq!(b.server_label, "time");
    }

    #[tokio::test]
    async fn connect_without_command_returns_error() {
        let result = McpConnection::builder("x").connect().await;
        match result {
            Ok(_) => panic!("connect with no command should fail"),
            Err(McpError::TransportCreation(msg)) => assert!(msg.contains("missing command")),
            Err(other) => panic!("expected TransportCreation, got {other:?}"),
        }
    }
}
