# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0-rc.1] — 2026-05-10

First release candidate. ailoop iterated under `0.1.x` without
stability guarantees; this is the first version published to
crates.io with a frozen public surface. The freeze landed after a
workspace-wide API audit (closed 2026-05-10).

### What ships

- **Streaming chat** with a unified `StreamChunk` event model
  covering tokens, tool calls, reasoning, history compaction, and
  per-turn / per-run lifecycle events.
- **Provider-agnostic** abstraction over `CompletionClient` +
  `CompletionModel`. Two adapters ship in-tree:
  - `ailoop-anthropic` — Messages API with explicit prompt caching
    (TTL-broken-down `cache_creation` counters, per-turn
    `service_tier`), configurable `anthropic-version` and
    `anthropic-beta` headers, tool use, sampling controls.
  - `ailoop-azure-openai` — v1 Chat Completions with API-key,
    Bearer, and bring-your-own `TokenProvider` (Entra), tool use
    with `parallel_tool_calls`, streaming usage with
    `cached_tokens`.
- **MCP MVP** (`ailoop-mcp`) — stdio transport, `tools/*` surface,
  wraps the official `rmcp` SDK. Tools discovered from any MCP
  server register through `ConversationBuilder::tool_dyn` like
  native ailoop tools.
- **Tool registry** with a type-safe `Tool` trait, the
  `#[ailoop_tool]` proc macro (including capability `tags(...)`),
  capability-based tool filtering with default-deny for untagged
  tools, and `ApprovalMiddleware` for human-in-the-loop gating.
- **Conversation history** with pin-aware compaction that
  preserves tool-call / tool-result pairing. Persistence via
  `Conversation::snapshot()` ↔ `ConversationBuilder::from_snapshot`,
  with an async `HistoryStore` trait (`InMemoryHistoryStore`,
  `JsonFileHistoryStore`).
- **Middleware** surface (`ChatMiddleware`) with `Started` /
  `Finished` lifecycle hooks, request transformation, tool gating,
  and a `Sanitize` middleware for closure-driven text rewriting at
  the model boundary.
- **`Conversation::run`** non-streaming helper for one-shot CLI
  and notebook flows (aborts surface as `FinishReason::Aborted(_)`
  on the outcome, never as `Err`).
- **Multimodal input**: image and document blocks on `UserBlock`,
  with Anthropic and Azure OpenAI Chat Completions mapping (Azure
  fails typed on tool-result images, which it cannot represent).

### Public-surface posture

- `#[non_exhaustive]` on ~24 public types so future variant /
  field additions remain non-breaking.
- Every type that appears in a public signature is nameable via
  `use ailoop::*`. Most application code only needs the
  `ailoop` façade (plus a provider adapter) as a direct
  dependency.
- `ScriptedModel` is opt-in via the `testing` feature on
  `ailoop-core`; production builds do not pull in test
  scaffolding.
- `#![deny(missing_docs)]` workspace-wide; `cargo doc --workspace
  --no-deps` is clean with and without `--features tracing`.

### Known gaps

- The Azure OpenAI adapter implements Chat Completions only; the
  Responses API is tracked for a later release.
- `ailoop-mcp` ships stdio + `tools/*`. SSE / HTTP transports,
  resources, prompts, and sampling are tracked for follow-up.
- Other providers (OpenAI public, Bedrock, Vertex, local engines)
  are not implemented.

[Unreleased]: https://github.com/KonorOko/ailoop/compare/v1.0.0-rc.1...HEAD
[1.0.0-rc.1]: https://github.com/KonorOko/ailoop/releases/tag/v1.0.0-rc.1
