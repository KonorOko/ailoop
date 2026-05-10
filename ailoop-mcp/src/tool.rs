use std::sync::Arc;

use ailoop_core::{ToolDefinition, ToolResultContent};
use ailoop_tools::ToolDyn;
use rmcp::model::{CallToolRequestParams, Content, RawContent};
use rmcp::service::{RoleClient, RunningService};

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
/// [`McpConnection`]: crate::McpConnection
/// [`SubAgentTool`]: https://docs.rs/ailoop
pub struct McpTool {
    pub(crate) client: Arc<RunningService<RoleClient, ()>>,
    /// Engine-facing name (`mcp__<server>__<tool>`).
    pub(crate) name_for_engine: String,
    /// Original name as exposed by the server — what we send back on
    /// `tools/call`.
    pub(crate) name_at_server: String,
    pub(crate) definition: ToolDefinition,
}

#[async_trait::async_trait]
impl ToolDyn for McpTool {
    fn name(&self) -> String {
        self.name_for_engine.clone()
    }

    fn tool_definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn call(&self, args: serde_json::Value) -> ToolResultContent {
        let mut req = CallToolRequestParams::new(self.name_at_server.clone());
        if let serde_json::Value::Object(map) = args {
            req = req.with_arguments(map);
        }
        // Non-object arguments (e.g. `null` from a model that decided
        // the tool takes no inputs) are sent without `arguments`, which
        // matches the MCP semantics for "no args".

        match self.client.call_tool(req).await {
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
}
