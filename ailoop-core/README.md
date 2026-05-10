# ailoop-core

Core vocabulary (messages, streams, hooks, middleware) for the
[`ailoop`](https://crates.io/crates/ailoop) SDK.

Most application code does not depend on `ailoop-core` directly — it
depends on `ailoop`, which re-exports the types you need to build a
`Conversation` and write a `ChatMiddleware`. Reach for `ailoop-core`
when:

- implementing a new `CompletionModel` / `CompletionClient` adapter
  for a provider HTTP API;
- writing a generic middleware reusable across façades, examples, or
  test harnesses;
- building reliability or observability infrastructure on top of the
  shared event vocabulary.

## What lives here

- `Message`, `UserBlock`, `AssistantBlock`, `ToolResultContent`,
  `SystemPrompt` — the wire model exchanged between engine and
  provider.
- `ChatRequest`, `ToolDefinition`, `ToolChoice`, `ToolTag` — the
  per-turn request shape adapters lower to the provider.
- `StreamChunk`, `FinishReason`, `Usage` — the event stream the
  engine emits for every run.
- `ChatMiddleware`, `HookAction`, `ToolDecision` — the extension
  point for transforms, gating, and observation.
- `CompletionClient`, `CompletionModel` — the contract every
  provider adapter implements.
- `RetryingModel`, `Retryable`, `RetryConfig` — backoff decorator
  that wraps any `CompletionModel`.

A `testing` feature gates `ScriptedModel`, used as a dev-dependency
by adapters and the façade.

See the [workspace README](https://github.com/KonorOko/ailoop) for the
big picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
