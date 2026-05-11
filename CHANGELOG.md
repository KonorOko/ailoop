# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Usage.reasoning_tokens`: subset of `output_tokens` consumed by
  hidden reasoning steps. Populated by Azure OpenAI when the deployment
  reports a `completion_tokens_details.reasoning_tokens` breakdown
  (o-series, gpt-5). Anthropic folds reasoning into `output_tokens`
  today, so the field stays at `0` there until the API surfaces a
  separate counter. `Usage` is `#[non_exhaustive]`, so the addition is
  non-breaking.

## [1.0.0-rc.3] — 2026-05-11

### Added

- `Conversation::builder().with_history(HistoryBuilder)`: configure
  the internal `History` (token budget, tokenizer, compaction
  strategy, `preserve_n_last`) at build time. Composes with
  `from_snapshot` in any call order — seeded messages and pin mask are
  preserved.
- `DEFAULT_HISTORY_MAX_TOKENS = 100_000`: public constant used when
  `with_history` is not called. Sized for a 200K-context Claude with a
  real tokenizer; ≈ 400 KB of transcript under the `CharTokenizer`
  fallback. Previously the budget was hardcoded to `460` (a test-only
  value) with no way to override it.

### Changed (BREAKING)

- Crate `ailoop-context` renamed to `ailoop-history`. The previous
  name is yanked on crates.io and republished as a `#[deprecated]`
  re-export shim.
- Type `ContextManager` renamed to `History`; `ContextManagerBuilder`
  renamed to `HistoryBuilder`. The crate already used "history" as
  the vocabulary for persistence (`HistoryStore`, `HistoryStore`
  implementations, `ConversationSnapshot`) — the rename aligns the
  in-memory container with that.
- `ConversationBuilder::from_snapshot` no longer eagerly builds the
  internal `History`: it stores the seeded messages until `build()`
  runs, so it composes with `with_history`. Public surface unchanged;
  callers should not observe the difference.

### Migration

```toml
# before
ailoop-context = "1.0.0-rc.2"

# after
ailoop-history = "1.0.0-rc.3"
```

```rust
// before
use ailoop_context::{ContextManager, ContextManagerBuilder};

// after
use ailoop_history::{History, HistoryBuilder};
```

To raise the history budget (recommended for any conversation that
will live beyond a handful of turns under a real tokenizer):

```rust
use ailoop::{Conversation, History};

let chat = Conversation::builder(model)
    .with_history(History::builder(150_000))
    .build()?;
```

The deprecated `ailoop-context` crate keeps re-exporting everything
from `ailoop-history`, so existing code compiles with a
`#[deprecated]` warning until you migrate.

## [1.0.0-rc.2] — 2026-05-10

### Added

- `ToolContext` and `ToolActivation` (`ailoop-tools`): per-dispatch
  context handed to every tool handler, exposing the run/step ids and
  a handle into the per-run active tool set. Tools that need to flip
  other tools on or off mid-run (deferred-tools / `search_tools`-style
  meta-tools) can call `ctx.tools().activate(name)` /
  `ctx.tools().list_inactive()` instead of threading
  `Arc<Mutex<HashSet<String>>>` through middleware.
- `ConversationBuilder::initial_active_tools(...)`: restrict the
  initial active set to a named subset. Other registered tools stay
  in the catalog and can be activated at runtime via the new
  `ToolContext` handle. Composes with `with_capabilities` (capability
  filter applies first).
- `ToolRegistry::tool_call_with_ctx`, `ToolRegistry::catalog_arc`,
  `ToolRegistry::snapshot_active`: lower-level building blocks the
  engine uses to thread `ToolContext` through dispatch.
- `examples/deferred-tools`: end-to-end demonstration of the
  `search_tools` pattern.

### Changed (BREAKING)

- `Tool::call` and `ToolDyn::call` now take an extra `&ToolContext`
  parameter. The `#[ailoop_tool]` macro absorbs this transparently
  for handlers that don't need it; functions that do need it can opt
  in by adding a trailing `ctx: &ToolContext` parameter to the
  function signature, and the macro routes the engine-supplied
  context through.
- Manual `impl ToolDyn for ...` (MCP-style adapters, plugin loaders)
  must add the new `ctx: &ToolContext` parameter to `call`. Handlers
  that don't use it can ignore the argument (`_ctx`).
- `ToolRegistry::tool_call(name, args)` keeps the same signature for
  standalone callers; internally it now constructs a detached
  `ToolContext`. Engine-level dispatch goes through the new
  `tool_call_with_ctx`.

### Migration

```rust
// Before
impl ToolDyn for MyTool {
    async fn call(&self, args: Value) -> ToolResultContent { ... }
}

// After
use ailoop::ToolContext;
impl ToolDyn for MyTool {
    async fn call(&self, args: Value, _ctx: &ToolContext) -> ToolResultContent { ... }
}
```

For the deferred-tools pattern, register every tool but expose only a
meta-tool initially:

```rust
let mut chat = Conversation::builder(model)
    .tool(SearchTools)
    .tool(Add).tool(Multiply).tool(Haversine)
    .initial_active_tools(["search_tools"])
    .build()?;
```

The `search_tools` handler reaches the active set through `ctx`:

```rust
#[ailoop_tool(description = "Activate tools matching a query")]
async fn search_tools(query: String, ctx: &ToolContext) -> String {
    for def in ctx.tools().list_inactive() {
        if def.name.contains(&query) {
            ctx.tools().activate(&def.name).ok();
        }
    }
    "done".into()
}
```

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

[Unreleased]: https://github.com/KonorOko/ailoop/compare/v1.0.0-rc.3...HEAD
[1.0.0-rc.3]: https://github.com/KonorOko/ailoop/releases/tag/v1.0.0-rc.3
[1.0.0-rc.2]: https://github.com/KonorOko/ailoop/releases/tag/v1.0.0-rc.2
[1.0.0-rc.1]: https://github.com/KonorOko/ailoop/releases/tag/v1.0.0-rc.1
