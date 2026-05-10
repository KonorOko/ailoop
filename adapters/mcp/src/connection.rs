use std::sync::Arc;

use ailoop_core::{ToolDefinition, ToolTag};
use ailoop_tools::ToolDyn;
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::TokioChildProcess;
use tokio::process::Command;

use crate::errors::McpError;
use crate::naming;
use crate::tool::McpTool;

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

    /// Discover the server's tools and wrap each one as
    /// `Arc<dyn ToolDyn>`. Each wrapper holds an `Arc` clone of the
    /// underlying rmcp client, so the connection survives as long as
    /// any returned tool is alive.
    ///
    /// Tools are exposed to the engine as
    /// `mcp__<server_label>__<tool_name>` (Claude-Desktop convention).
    /// Characters outside `[A-Za-z0-9_-]` are replaced with `_`; if the
    /// composed name still exceeds 64 chars it is truncated and a short
    /// deterministic hash suffix is appended so distinct long names do
    /// not collide.
    ///
    /// Every tool is tagged with `[Network, Custom("mcp")]`. Use the
    /// `Custom("mcp")` tag with `with_approval_for_tags` if you want
    /// to gate every MCP call through a single approval callback.
    pub async fn list_tools(&self) -> Result<Vec<Arc<dyn ToolDyn>>, McpError> {
        let tools = self
            .inner
            .list_all_tools()
            .await
            .map_err(|e| McpError::Service(e.to_string()))?;

        let default_tags = vec![ToolTag::Network, ToolTag::Custom("mcp".into())];

        let wrappers = tools
            .into_iter()
            .map(|t| {
                let name_at_server = t.name.to_string();
                let name_for_engine = naming::compose(&self.server_label, &name_at_server);
                let definition = ToolDefinition::new(
                    &name_for_engine,
                    &t.description.map(|d| d.to_string()).unwrap_or_default(),
                    serde_json::Value::Object((*t.input_schema).clone()),
                    default_tags.clone(),
                );
                Arc::new(McpTool {
                    client: self.inner.clone(),
                    name_for_engine,
                    name_at_server,
                    definition,
                }) as Arc<dyn ToolDyn>
            })
            .collect();

        Ok(wrappers)
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

        let inner = ().serve(transport).await.map_err(|e| McpError::Service(e.to_string()))?;

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
