# ailoop-openai

[OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat)
adapter for [`ailoop`](https://crates.io/crates/ailoop), for OpenAI and
the servers that speak the same API: OpenRouter, Ollama, LM Studio,
vLLM, llama.cpp.

- Configurable base URL and optional API key (local servers need none)
- Streaming text, tool calls and visible reasoning (`reasoning` /
  `reasoning_content`)
- Image input (`image_url`, base64 as `data:` URIs)
- `list_models()` over `GET /models`, and `model_capabilities()` for
  whether a model accepts images and tools, when the server says so
- Errors sorted into what an app shows: offline, local service not
  running, key rejected, rate limit with retry time, quota exhausted,
  model without image/tool support, model not found, context overflow,
  response cut off

```toml
[dependencies]
ailoop = "1.0.0-rc.7"
ailoop-openai = "1.0.0-rc.7"
```

```rust
use ailoop::{Conversation, RetryingModel};
use ailoop_openai::{ErrorCategory, OpenAIClient};

// Any of these:
let client = OpenAIClient::openai(key);                 // https://api.openai.com/v1
let client = OpenAIClient::openrouter(key);             // https://openrouter.ai/api/v1
let client = OpenAIClient::ollama();                    // http://localhost:11434/v1
let client = OpenAIClient::new(user_url).maybe_api_key(user_key);

let models = client.list_models().await?;
let caps = client.model_capabilities(&models[0].id).await?; // images / tools: Option<bool>

let mut chat = Conversation::builder(RetryingModel::new(client.model(&models[0].id))).build()?;
```

`OpenAIError::category()` returns an `ErrorCategory` to pick the
message from. Through `Conversation`, the adapter's error is inside
`EngineError::Model`.

## Notes

- The base URL ends in the version segment, as providers document it.
  A bare host gets `/v1`; a trailing `/chat/completions` is removed.
- `max_completion_tokens` is sent to `api.openai.com` (its reasoning
  models reject `max_tokens`), `max_tokens` everywhere else. Override
  with `max_tokens_field`.
- "Does not support images/tools" is recognized from the server's
  message, and only when the request carried images or tools. A server
  that silently ignores an image cannot be detected; check
  `model_capabilities` first.
- Not covered: the Responses API, reasoning round-trip, documents,
  images inside tool results.

See the [workspace README](https://github.com/KonorOko/ailoop) for the
big picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
