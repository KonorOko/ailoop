# ailoop

High-level façade for building LLM agent loops in Rust.

Most application code only depends on this crate — it re-exports the
vocabulary from `ailoop-core` (messages, stream chunks, hooks) and
from the side crates (`ailoop-context`, `ailoop-tools`,
`ailoop-prompts`) you need to wire a `Conversation` together.

```toml
[dependencies]
ailoop = "1.0.0-rc.1"
ailoop-anthropic = "1.0.0-rc.1"
tokio = { version = "1", features = ["full"] }
```

```rust,no_run
use ailoop::Conversation;
use ailoop_anthropic::AnthropicClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = AnthropicClient::from_env()?.model("claude-sonnet-4-6");
    let mut chat = Conversation::builder(model)
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let outcome = chat.run("What is the speed of light?").await?;
    println!("{}", outcome.final_text.unwrap_or_default());
    Ok(())
}
```

`Conversation::run` drives the agent loop end-to-end. For token-level
streaming, use `Conversation::stream` instead.

See the [workspace README](https://github.com/KonorOko/ailoop) for the
big picture, the [examples](https://github.com/KonorOko/ailoop/tree/main/examples),
and a list of available provider adapters.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
