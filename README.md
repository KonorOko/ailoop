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

## Workspace layout

| Crate                 | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| `ailoop`              | High-level API (`Conversation`, `run_chat`)        |
| `ailoop-core`         | Message, stream, and provider trait definitions    |
| `ailoop-anthropic`    | Anthropic Messages API adapter                     |
| `ailoop-azure-openai` | Azure OpenAI v1 API adapter (Chat Completions)     |
| `ailoop-context`      | Conversation history and context management        |
| `ailoop-prompts`      | Composable system prompt utilities                 |
| `ailoop-tools`        | Tool registry and tool calling primitives          |
| `ailoop-derive`       | Derive macros (proc-macro)                         |

## Quick start

The repository ships with two runnable examples. Both require an
`ANTHROPIC_API_KEY` in your environment (a `.env` file at the repo root also
works).

A minimal streaming chat with no tools:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
cargo run -p basic-chat
```

A chat with a single `add(a, b)` tool that the model calls when asked to sum
two numbers:

```sh
cargo run -p tool-use
```

See [`examples/basic-chat`](examples/basic-chat/src/main.rs) and
[`examples/tool-use`](examples/tool-use/src/main.rs) for the full source.

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
