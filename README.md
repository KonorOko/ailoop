# Ailoop

Ergonomic, provider-agnostic AI agent engine for Rust.

Ailoop is a Rust toolkit for building streaming chat agents with tool use,
system prompts, and conversation history. The engine is a thin layer over an
open trait-based provider abstraction — provider adapters keep your application
decoupled from any single vendor.

## Features

- Token-level streaming with a unified `StreamChunk` event model
- Type-safe tool use backed by `serde` (`Tool` trait + `ToolRegistry`)
- Per-dispatch `ToolContext` so tools can activate other tools mid-run
  (deferred-tools / `search_tools` patterns) without shared mutable state
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
| `ailoop-history`      | Conversation history and compaction (formerly `ailoop-context`) |
| `ailoop-mcp`          | MCP (Model Context Protocol) adapter (stdio MVP)   |
| `ailoop-prompts`      | Composable system prompt utilities                 |
| `ailoop-tools`        | Tool registry and tool calling primitives          |
| `ailoop-derive`       | Derive macros (proc-macro)                         |

## Quick start

```toml
[dependencies]
ailoop = "1.0.0-rc.3"
ailoop-anthropic = "1.0.0-rc.3"
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

### Multimodal kickoff

`run` / `stream` accept anything that converts into a `Message`, so the
same call site works for plain text, a single image, or a multi-block
user turn that interleaves text and inline media:

```rust
use ailoop::{Conversation, Message, Source, UserBlock};

let png = std::fs::read("chart.png")?;
let outcome = chat
    .run(Message::user_with_blocks([
        UserBlock::text("What's in this chart?"),
        UserBlock::image(Source::Base64 {
            media_type: "image/png".into(),
            data: base64::engine::general_purpose::STANDARD.encode(&png),
        }),
    ]))
    .await?;
```

`Vec<UserBlock>` and a bare `UserBlock` (image-only / document-only
turns) are also accepted directly. Provider adapters map the blocks to
the underlying wire format — Anthropic supports image and document
content natively; the Azure Chat Completions adapter surfaces a typed
error for document blocks it can't represent.

## Examples

The repository ships with five runnable examples. All five require an
`ANTHROPIC_API_KEY` in your environment (a `.env` file at the repo root also
works).

| Example                                                  | Demonstrates                                                |
| -------------------------------------------------------- | ----------------------------------------------------------- |
| [`basic-chat`](examples/basic-chat/src/main.rs)          | Minimal one-shot chat with no tools                         |
| [`tool-use`](examples/tool-use/src/main.rs)              | A typed `#[ailoop_tool]` that the model calls when needed   |
| [`deferred-tools`](examples/deferred-tools/src/main.rs)  | Hide most tools behind a `search_tools` meta-tool that the model uses to activate them on demand |
| [`mcp-time`](examples/mcp-time/src/main.rs)              | Tools discovered from a real MCP server (`mcp-server-time`) |
| [`persistent-chat`](examples/persistent-chat/src/main.rs) | Snapshot / restore a `Conversation` across process restarts |

```sh
export ANTHROPIC_API_KEY=sk-ant-...
cargo run -p basic-chat
cargo run -p tool-use
cargo run -p deferred-tools
cargo run -p mcp-time         # also requires `uvx` on PATH (`pip install uv`)
cargo run -p persistent-chat
```

## Adding a provider

A provider is anything that implements `CompletionClient` and `CompletionModel`
from `ailoop-core`. Currently shipped:

- **Anthropic** — `ailoop-anthropic`
- **Azure OpenAI** — `ailoop-azure-openai` (Chat Completions v1; Responses API
  pending)

Other providers (OpenAI public, Bedrock, local models, etc.) are not yet
implemented.

## Documentation

- API reference: [docs.rs/ailoop](https://docs.rs/ailoop)
- Release notes: [`CHANGELOG.md`](CHANGELOG.md)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
