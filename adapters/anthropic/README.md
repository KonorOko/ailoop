# ailoop-anthropic

[Anthropic Messages API](https://docs.anthropic.com/claude/reference/messages_post)
adapter for [`ailoop`](https://crates.io/crates/ailoop).

Streams Claude responses through ailoop's unified `StreamChunk`
event model, with support for:

- Configurable `anthropic-version` and `anthropic-beta` headers
- Tool use (typed tools registered through `ailoop-tools`)
- Explicit prompt caching via per-block `cache_control` (system +
  messages + tools)
- TTL-broken-down `cache_creation` counters (5m / 1h) and per-turn
  `service_tier` on `StreamChunk::TurnFinished`
- Sampling controls (`temperature`, `top_p`, `top_k`,
  `stop_sequences`) and `tool_choice`

```toml
[dependencies]
ailoop = "1.0.0-rc.1"
ailoop-anthropic = "1.0.0-rc.1"
```

```rust,no_run
use ailoop::Conversation;
use ailoop_anthropic::AnthropicClient;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let model = AnthropicClient::from_env()?.model("claude-sonnet-4-6");
let mut chat = Conversation::builder(model).build()?;
let outcome = chat.run("Hello!").await?;
# Ok(()) }
```

Set `ANTHROPIC_API_KEY` in your environment (a `.env` file at the
repo root also works).

See the [workspace README](https://github.com/KonorOko/ailoop) for
the big picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
