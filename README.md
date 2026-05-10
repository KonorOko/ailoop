# Ailoop

Ergonomic, provider-agnostic AI agent engine for Rust.

Ailoop is a Rust toolkit for building streaming chat agents with tool use,
system prompts, and conversation history. The engine is a thin layer over an
open trait-based provider abstraction — provider adapters keep your application
decoupled from any single vendor.

> **Status: early development.** APIs will change without notice. Not
> recommended for production use.

## Features

- Token-level streaming with a unified `StreamChunk` event model
- Type-safe tool use backed by `serde` (`Tool` trait + `ToolRegistry`)
- Conversation history with automatic context management
- Provider-agnostic: implement two traits to support a new provider
- Middleware hooks for observation, cancellation, and request transformation
- MCP-compatible: register tools discovered from any [Model Context
  Protocol][mcp-spec] server via the `ailoop-mcp` adapter

[mcp-spec]: https://modelcontextprotocol.io

## Workspace layout

| Crate                 | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| `ailoop`              | High-level API (`Conversation`, `advanced::run_chat`) |
| `ailoop-core`         | Message, stream, and provider trait definitions    |
| `ailoop-anthropic`    | Anthropic Messages API adapter                     |
| `ailoop-azure-openai` | Azure OpenAI v1 API adapter (Chat Completions)     |
| `ailoop-context`      | Conversation history and context management        |
| `ailoop-mcp`          | MCP (Model Context Protocol) adapter (stdio MVP)   |
| `ailoop-prompts`      | Composable system prompt utilities                 |
| `ailoop-tools`        | Tool registry and tool calling primitives          |
| `ailoop-derive`       | Derive macros (proc-macro)                         |

## Quick start

```toml
[dependencies]
ailoop = "0.1"
ailoop-anthropic = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust
use ailoop::{Conversation, RetryingModel};
use ailoop_anthropic::AnthropicClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = RetryingModel::new(
        AnthropicClient::from_env()?.model("claude-sonnet-4-6"),
    );
    let mut chat = Conversation::builder(model).build()?;

    let outcome = chat.run("Explain Rust ownership in one short sentence.").await?;
    println!("{}", outcome.final_text.unwrap_or_default());
    Ok(())
}
```

`Conversation::run` drives the agent loop end-to-end: it sends the request,
streams the response, runs any tools the model calls, and returns a
`RunOutcome` carrying the final text, token usage, and the new messages it
appended to history. For token-level streaming, use `Conversation::stream`
instead.

## Examples

The repository ships with three runnable examples. All three require an
`ANTHROPIC_API_KEY` in your environment (a `.env` file at the repo root also
works).

| Example                                                | Demonstrates                                              |
| ------------------------------------------------------ | --------------------------------------------------------- |
| [`basic-chat`](examples/basic-chat/src/main.rs)        | Minimal one-shot chat with no tools                       |
| [`tool-use`](examples/tool-use/src/main.rs)            | A typed `#[ailoop_tool]` that the model calls when needed |
| [`mcp-time`](examples/mcp-time/src/main.rs)            | Tools discovered from a real MCP server (`mcp-server-time`) |

```sh
export ANTHROPIC_API_KEY=sk-ant-...
cargo run -p basic-chat
cargo run -p tool-use
cargo run -p mcp-time   # also requires `uvx` on PATH (`pip install uv`)
```

## Adding a provider

A provider is anything that implements `CompletionClient` and `CompletionModel`
from `ailoop-core`. Currently shipped:

- **Anthropic** — `ailoop-anthropic`
- **Azure OpenAI** — `ailoop-azure-openai` (Chat Completions v1; Responses API
  pending)

Other providers (OpenAI public, Bedrock, local models, etc.) are not yet
implemented.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
