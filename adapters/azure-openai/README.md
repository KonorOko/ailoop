# ailoop-azure-openai

[Azure OpenAI v1 Chat Completions](https://learn.microsoft.com/azure/ai-services/openai/reference)
adapter for [`ailoop`](https://crates.io/crates/ailoop).

Streams Azure OpenAI responses through ailoop's unified
`StreamChunk` event model, with support for:

- API-key, Bearer, or bring-your-own `TokenProvider` for Entra-ID
  auth
- Tool use with `parallel_tool_calls` (the inverse of Anthropic's
  `disable_parallel_tool_use`)
- Streaming usage with `cached_tokens`
- Sampling controls (`temperature`, `top_p`, `stop_sequences`) and
  `tool_choice` (with `Any` lowered to `"required"`)

```toml
[dependencies]
ailoop = "1.0.0-rc.1"
ailoop-azure-openai = "1.0.0-rc.1"
```

Input images and documents map through `UserBlock` content blocks;
tool-result images fail typed because Chat Completions cannot
represent them. The Responses API path is tracked for a later
release.

See the [workspace README](https://github.com/KonorOko/ailoop) for
the big picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
