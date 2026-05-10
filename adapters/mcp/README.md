# ailoop-mcp

[Model Context Protocol](https://modelcontextprotocol.io) adapter
for [`ailoop`](https://crates.io/crates/ailoop).

Wraps the official `rmcp` SDK so tools discovered from any MCP
server register through `ConversationBuilder::tool_dyn` like any
other ailoop tool.

Current MVP scope (1.0):

- **Transport:** stdio (child process)
- **Surface:** `tools/*` — `list_tools`, `call_tool`
- **Error mapping:** typed `McpError` for connection and protocol
  failures, with in-band `ToolResultContent::Error` for
  model-visible tool failures

```toml
[dependencies]
ailoop = "1.0.0-rc.1"
ailoop-anthropic = "1.0.0-rc.1"
ailoop-mcp = "1.0.0-rc.1"
```

Advanced MCP features (resources, prompts, sampling, SSE / HTTP
transports) are tracked for follow-up releases driven by real use
cases.

See the
[`mcp-time` example](https://github.com/KonorOko/ailoop/tree/main/examples/mcp-time)
for a working setup, and the
[workspace README](https://github.com/KonorOko/ailoop) for the big
picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
