use std::sync::Arc;

use ailoop_core::{ToolDefinition, ToolResultContent};
use ailoop_tools::{ToolContext, ToolDyn};
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, ClientRequest, Content, Meta, NumberOrString,
    ProgressToken, RawContent, ServerResult,
};
use rmcp::service::{PeerRequestOptions, RoleClient, RunningService, ServiceError};

use crate::progress::ProgressRouter;

/// `ToolDyn` wrapper around a single tool exposed by an MCP server.
///
/// One `McpTool` instance per tool discovered on the server. All
/// instances built from the same [`McpConnection`] share the same
/// underlying `Arc<RunningService>`, so the connection survives as
/// long as any tool is registered.
///
/// Errors from the wire (transport drops, server-side `isError: true`,
/// schema mismatches) are mapped to a [`ToolResultContent`] with
/// `is_error: true` so the model sees them as a tool reply — never as
/// an `Err` to the engine. This mirrors the [`SubAgentTool`]
/// convention.
///
/// [`ToolResultContent`]: ailoop_core::ToolResultContent
///
/// Each call asks the server for progress (a `progressToken` in the
/// request's `_meta`); the server's `notifications/progress` for it are
/// forwarded through [`ToolContext::progress`], so they reach the run's
/// stream as `ToolProgress` chunks. Servers that do not report progress
/// simply send none.
///
/// [`McpConnection`]: crate::McpConnection
/// [`SubAgentTool`]: https://docs.rs/ailoop
pub struct McpTool {
    pub(crate) client: Arc<RunningService<RoleClient, ProgressRouter>>,
    pub(crate) progress: ProgressRouter,
    /// Engine-facing name (`mcp__<server>__<tool>`).
    pub(crate) name_for_engine: String,
    /// Original name as exposed by the server — what we send back on
    /// `tools/call`.
    pub(crate) name_at_server: String,
    pub(crate) definition: ToolDefinition,
}

#[async_trait::async_trait]
impl ToolDyn for McpTool {
    fn name(&self) -> &str {
        &self.name_for_engine
    }

    fn tool_definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn call(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolResultContent {
        let mut req = CallToolRequestParams::new(self.name_at_server.clone());
        if let serde_json::Value::Object(map) = args {
            req = req.with_arguments(map);
        }
        // Non-object arguments (e.g. `null` from a model that decided
        // the tool takes no inputs) are sent without `arguments`, which
        // matches the MCP semantics for "no args".

        // Unique across runs sharing this connection: call ids come from
        // the provider and only need to be unique within a run.
        let token = format!("{}:{}", ctx.run_id(), ctx.call_id());
        let _routed = self
            .progress
            .register(token.clone(), ctx.progress_sink().clone());
        // rmcp stamps every request with a progress token of its own,
        // which overrides one set in the params' `_meta`; `options.meta`
        // is applied after it, so ours is the one the server sees.
        let mut options = PeerRequestOptions::no_options();
        options.meta = Some(Meta::with_progress_token(ProgressToken(
            NumberOrString::String(token.into()),
        )));
        let request = ClientRequest::CallToolRequest(CallToolRequest::new(req));
        let result = async {
            let handle = self
                .client
                .send_request_with_option(request, options)
                .await?;
            match handle.await_response().await? {
                ServerResult::CallToolResult(result) => Ok(result),
                _ => Err(ServiceError::UnexpectedResponse),
            }
        }
        .await;

        match result {
            Err(e) => ToolResultContent::error(format!("MCP transport error: {e}")),
            Ok(result) => {
                let text = stringify_content(&result.content);
                if result.is_error.unwrap_or(false) {
                    ToolResultContent::error(text)
                } else {
                    ToolResultContent::text(text)
                }
            }
        }
    }
}

/// Concatenate every text block in `content` and replace non-text
/// blocks with short placeholders so the model still sees the
/// structure even when binary data is involved.
pub(crate) fn stringify_content(content: &[Content]) -> String {
    let mut buf = String::new();
    for (i, c) in content.iter().enumerate() {
        if i > 0 {
            buf.push('\n');
        }
        match &c.raw {
            RawContent::Text(t) => buf.push_str(&t.text),
            RawContent::Image(img) => buf.push_str(&format!(
                "[image: {} base64 chars, {}]",
                img.data.len(),
                img.mime_type
            )),
            RawContent::Audio(a) => buf.push_str(&format!(
                "[audio: {} base64 chars, {}]",
                a.data.len(),
                a.mime_type
            )),
            RawContent::Resource(_) => buf.push_str("[embedded resource]"),
            RawContent::ResourceLink(r) => buf.push_str(&format!("[resource: {}]", r.uri)),
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{RawImageContent, RawTextContent};

    fn text(s: &str) -> Content {
        Content {
            raw: RawContent::Text(RawTextContent {
                text: s.into(),
                meta: None,
            }),
            annotations: None,
        }
    }

    fn image(data: &str, mime: &str) -> Content {
        Content {
            raw: RawContent::Image(RawImageContent {
                data: data.into(),
                mime_type: mime.into(),
                meta: None,
            }),
            annotations: None,
        }
    }

    #[test]
    fn stringify_concatenates_text_blocks_with_newlines() {
        let blocks = vec![text("hello"), text("world")];
        assert_eq!(stringify_content(&blocks), "hello\nworld");
    }

    #[test]
    fn stringify_replaces_non_text_with_placeholders() {
        let blocks = vec![text("here is an image:"), image("AAAA", "image/png")];
        let s = stringify_content(&blocks);
        assert!(s.starts_with("here is an image:\n["));
        assert!(s.contains("image/png"));
    }

    #[test]
    fn stringify_handles_empty_content() {
        assert_eq!(stringify_content(&[]), "");
    }

    /// Reports progress on the token the request carries, then holds
    /// the call open until the test has seen the update.
    struct ReportingServer {
        release: Arc<tokio::sync::Notify>,
    }

    impl rmcp::ServerHandler for ReportingServer {
        async fn call_tool(
            &self,
            _request: CallToolRequestParams,
            context: rmcp::service::RequestContext<rmcp::RoleServer>,
        ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
            let token = context.meta.get_progress_token().expect("a progress token");
            let mut update = rmcp::model::ProgressNotificationParam::new(token, 1.0);
            update.total = Some(2.0);
            context.peer.notify_progress(update).await.expect("notify");
            self.release.notified().await;
            Ok(rmcp::model::CallToolResult::success(vec![Content::text(
                "done",
            )]))
        }
    }

    #[tokio::test]
    async fn server_progress_reaches_the_call() {
        use rmcp::ServiceExt;

        let (client_io, server_io) = tokio::io::duplex(4096);
        let release = Arc::new(tokio::sync::Notify::new());
        let server = ReportingServer {
            release: release.clone(),
        };
        tokio::spawn(async move {
            let running = server.serve(server_io).await.expect("server handshake");
            let _ = running.waiting().await;
        });
        let progress = ProgressRouter::default();
        let client = progress
            .clone()
            .serve(client_io)
            .await
            .expect("client handshake");
        let tool = McpTool {
            client: Arc::new(client),
            progress,
            name_for_engine: "mcp__test__work".into(),
            name_at_server: "work".into(),
            definition: ToolDefinition::new("mcp__test__work", "", serde_json::json!({}), vec![]),
        };

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let ctx = ToolContext::detached()
            .with_progress_sink(ailoop_tools::ProgressSink::new("call_1", tx));
        let call = tokio::spawn(async move { tool.call(serde_json::json!({}), &ctx).await });

        let (call_id, update) = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("the server's progress should reach the call")
            .expect("sink open");
        assert_eq!(call_id, "call_1");
        assert_eq!(
            update,
            ailoop_core::ToolProgress::new()
                .with_progress(1.0)
                .with_total(2.0)
        );

        release.notify_one();
        let result = call.await.expect("call task");
        assert!(!result.is_error);
    }
}
