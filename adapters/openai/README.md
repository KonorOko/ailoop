# ailoop-openai

[OpenAI Chat Completions](https://platform.openai.com/docs/api-reference/chat)
adapter for [`ailoop`](https://crates.io/crates/ailoop), for OpenAI,
Azure OpenAI and the servers that speak the same API: OpenRouter,
Ollama, LM Studio, vLLM, llama.cpp.

- Configurable base URL and optional API key (local servers need none)
- Azure OpenAI's v1 API with an `api-key`, a fixed bearer token, or a
  `TokenProvider` for Microsoft Entra ID (`CachingTokenProvider` included)
- Streaming text, tool calls and visible reasoning (`reasoning` /
  `reasoning_content`)
- Image input (`image_url`, base64 as `data:` URIs)
- `list_models()` over `GET /models`, and `model_capabilities()` for
  whether a model accepts images and tools, when the server says so
- Errors sorted into `ailoop_core::ProviderErrorKind` through
  `ProviderError::kind()`: no connection, key rejected, rate limit,
  quota exhausted, model without image/tool support, model not found,
  content filter, context overflow, response cut off

```toml
[dependencies]
ailoop = "1.0.0-rc.7"
ailoop-openai = "1.0.0-rc.7"
```

```rust
use ailoop::{Conversation, RetryingModel};
use ailoop_openai::{AzureAuth, OpenAIClient};

// Any of these:
let client = OpenAIClient::openai(key);                 // https://api.openai.com/v1
let client = OpenAIClient::openrouter(key);             // https://openrouter.ai/api/v1
let client = OpenAIClient::ollama();                    // http://localhost:11434/v1
let client = OpenAIClient::azure(endpoint, AzureAuth::ApiKey(key)); // {endpoint}/openai/v1
let client = OpenAIClient::azure_from_env()?;           // AZURE_OPENAI_ENDPOINT + _API_KEY or _TOKEN
let client = OpenAIClient::new(user_url).maybe_api_key(user_key);

let models = client.list_models().await?;
let caps = client.model_capabilities(&models[0].id).await?; // images / tools: Option<bool>

let mut chat = Conversation::builder(RetryingModel::new(client.model(&models[0].id))).build()?;
```

`ProviderError::kind()` on `OpenAIError` returns a `ProviderErrorKind`
to pick the message from. A connection failure is `Network`;
`client.is_local()` tells "Ollama is not running" from "no network". The
retry delay comes from `Retryable::retry_classification()`. Through
`Conversation`, the adapter's error is inside `EngineError::Model`.

## Notes

- The base URL ends in the version segment, as providers document it.
  A bare host gets `/v1`; a trailing `/chat/completions` is removed.
- `max_completion_tokens` is sent to `api.openai.com` and Azure (their
  reasoning models reject `max_tokens`), `max_tokens` everywhere else.
  Override with `max_tokens_field`.
- On Azure the model name is the deployment name. `list_models()` lists
  the base models the resource can deploy, not your deployments, and
  `model_capabilities()` answers unknown without sending a request.
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
