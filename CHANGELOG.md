# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ProviderError` trait (in `ailoop-core`, re-exported from `ailoop`)
  with `is_context_overflow()`, which defaults to `false`. It answers
  "did the provider reject this prompt because it does not fit the
  context window?" without the caller knowing the adapter's error type.
  This lets the engine and `Conversation` compact the history and try
  again in a later release. The trait is separate from `Retryable`:
  overflow is `Permanent` for `RetryingModel`, because resending the
  same prompt fails the same way, but a caller that owns the history
  can still recover. `AnthropicError`, `AzureOpenAIError`, and
  `testing::ScriptedError` implement it. `ScriptedError` reports an
  overflow when its message contains `"context_overflow"`.
- `AnthropicApiErrorKind::ContextOverflow`: a 400
  `invalid_request_error` whose message contains "prompt is too long",
  which the API returns when the input exceeds the model's context
  window. It was classified as `InvalidRequest`, so it could not be told
  apart from other validation failures. It is detected for both HTTP
  error envelopes and mid-stream error events. The new
  `AnthropicApiErrorKind::from_error(type, message)` does the
  message-aware mapping. `from_error_type` is unchanged and still
  returns `InvalidRequest`, because the type alone cannot distinguish
  the two. Other `invalid_request_error`s stay `InvalidRequest`.
- `AzureOpenAIApiErrorKind::ContextOverflow`: `error.code ==
  "context_length_exceeded"`. It was captured as
  `Other("context_length_exceeded")`, which `RetryingModel` treated as
  transient, so a prompt that could never fit was resent up to
  `max_attempts` times. It is now `Permanent`.

- `HistoryBuilder::reserved_tokens(n)` (default 0): headroom subtracted
  from `max_tokens` before `History::compact_if_needed` compares it with
  `estimated_tokens()`. The estimate counts only the history's messages,
  but the same context window also has to fit the system prompt, the
  tool schemas, the requested output `max_tokens`, and the tokenizer's
  estimation error (`CharTokenizer` is a `len() / 4` heuristic).
  Previously the only way to leave room for those was to shrink
  `max_tokens` by a guessed factor; now `max_tokens` can be set to the
  model's context window and the slack stated explicitly. The effective
  threshold is `max_tokens.saturating_sub(reserved)`, so a reserve at or
  above `max_tokens` never panics: every call compacts, and
  `CompactionError::NotEnoughHistory` surfaces once only the preserved
  tail is left. The reserve is builder configuration, not snapshot
  state: it applies through `History::from_messages` and
  `ConversationBuilder::with_history` (including after
  `from_snapshot`), and is not persisted in `ConversationSnapshot`.
  `estimated_tokens()` is unchanged.
- `History::force_compact()`: runs the compaction strategy regardless
  of the token budget. It is for the case where the estimate said the
  history fits and the provider disagreed, for example after a
  context-window overflow. It returns the same `CompactionReport` and
  errors as `compact_if_needed`. The strategy may still be unable to
  shrink the history (everything before the preserved tail is pinned,
  or the tail itself is what does not fit), so compare
  `estimated_tokens()` before and after when that matters.
  `compact_if_needed` now delegates to it.
- `History::needs_compaction()`: exposes the budget check
  `compact_if_needed` runs (`estimated_tokens() >= max_tokens -
  reserved_tokens`) without running the strategy.
- `History::replace_messages(messages, pinned)`: the in-place
  counterpart of `History::from_messages`. It swaps the message vector
  and pin mask and keeps the budget, strategy and tokenizer. Useful
  for rolling back to a captured state or loading a snapshot into a
  live history. If the lengths differ it returns
  `FromMessagesError::LengthMismatch` and leaves the history untouched.
- Re-export `CacheControl`, `SystemBlock`, `SystemPrompt`, and
  `ToolResultBlock` from the `ailoop` façade. Downstream crates that
  write custom `ChatMiddleware`s (setting `SystemPrompt::Blocks` with
  per-block cache breakpoints, building multi-block tool replies via
  `ToolResultContent::from_blocks`, or threading `CacheControl::Ephemeral`
  through the `with_cache_control` builders) no longer need a direct
  dependency on `ailoop-core`.
- `AntiLoop::with_tool_call_identity(|name, args| -> String)`: pluggable
  equivalence key for the tool-call loop detector. The callback maps
  `(name, args)` to a string and the streak counter compares those
  strings instead of the default structural `serde_json::Value`
  `PartialEq`. Closes a real failure mode against coding agents that
  re-issue destructive calls with cosmetic argument variation
  (whitespace inside string fields, reordered keys inside an embedded
  JSON payload, ignored auxiliary fields): the default detector saw
  those as distinct and reset the streak, so the loop slipped through.
  The terminate reason includes the computed identity string for
  diagnostics. Mirrors the existing `with_text_predicate` for the text
  detector, with one intentional asymmetry — text takes a predicate,
  tool-call takes an identity (strictly more expressive, lighter
  per-run state, useful diagnostic key for free). Default behaviour is
  unchanged: when no identity is configured, the path stays on
  `Value::PartialEq` and the legacy reason wording
  ("...called N times in a row with identical args") is preserved.
- Multimodal kickoff for `Conversation`: `run`, `run_with_options`,
  `stream`, and `stream_with_options` now take `impl Into<Message>`
  instead of `impl Into<String>`. Four new `From` impls on
  `ailoop_core::Message` (`&str`, `String`, `UserBlock`,
  `Vec<UserBlock>`) cover the common shapes, and a new
  `Message::user_with_blocks(blocks)` constructor is the idiomatic way
  to build a multi-block user turn (e.g. text + image, text + PDF) for
  the kickoff without writing an attachment middleware. Backward-
  compatible: existing `&str` / `String` callers compile and behave
  identically.
- `ConversationBuilder::tools_with_prompt_file(names, path)`: associate
  one [`PromptSection`] read from disk with a *group* of tool names.
  The section is appended to the system prompt at most once per turn
  when at least one tool in the group is active, fixing the
  duplication that arises when several tools share the same guide —
  previously the only way to attach a guide was per-tool via
  `tool_with_prompt_file`, which keyed sections by tool name and so
  emitted the same guide N times for an N-tool family. Render order
  follows group registration order (not the order of tools in
  `req.tools`). Unlike `tool_with_prompt_file`, this method does *not*
  register the tools — pair it with the usual `.tool(...)` /
  `.tool_dyn(...)` calls. Passing an empty `names` iterator surfaces
  `BuildError::EmptyToolGroup` at `build()` time. The 1:1
  `tool_with_prompt_file` API is unchanged.
- `BuildError::EmptyToolGroup`: new builder-error variant raised when
  `tools_with_prompt_file` is called with an empty tool-name list.
- `ToolContext::cancellation() -> &CancellationToken`: cooperative
  cancellation handle exposed to tool handlers. Mirrors the token the
  caller supplied to `RunConfig.cancellation` (or a fresh
  never-cancelled handle when none was set), built once at run start
  and cloned into every per-dispatch context. The engine already
  drops the tool future via `select!` on cancellation — that cancels
  in-flight async I/O on its own — so this token is the escape hatch
  for cases drop-cancellation doesn't reach: `spawn_blocking` work,
  `tokio::process` children that need an explicit SIGTERM, ordered
  cleanup before the future is dropped, and `JoinSet` fan-out that
  wants to distribute `child_token()` to siblings.

### Changed

- Docstrings for `StreamChunk::TurnFinished` and `Usage` now make
  explicit that the per-turn variant is visible to `ChatMiddleware`
  implementations only — the engine accumulates its `usage` into the
  run total and drops the chunk before the public stream, so stream
  consumers see aggregated `Usage` on `RunFinished.usage`, never the
  per-turn variant. Middleware path points to `on_chunk` for cases
  that need per-turn data (context-size indicator from the final
  turn's `input_tokens`, per-turn latency or service-tier attribution,
  online tokenizer calibration).

### Changed (BREAKING)

- `ToolContext::new` gained a trailing `cancellation: CancellationToken`
  parameter. Engine-internal; external callers rarely construct
  `ToolContext` directly — standalone callers go through
  `ToolContext::detached()`, whose signature is unchanged (it mints a
  fresh never-cancelled token internally).
- `FinishReason::Aborted` now carries a structured `AbortReason`
  (re-exported as `ailoop::AbortReason`) instead of a free-form
  `String`. Callers can tell *why* a run stopped by matching on the
  variant rather than parsing text:
  - `Timeout(Duration)`: `RunConfig::timeout` / `RunOptions::timeout`
    elapsed. It carries the configured duration.
  - `Cancelled`: the `CancellationToken` fired. It still wins over a
    timeout that fires at the same instant.
  - `Terminated { reason }`: a middleware returned
    `HookAction::Terminate` from `on_run_started`.
  - `ToolTerminated { tool_name, reason }`: a middleware (`AntiLoop`,
    `MaxToolCalls`, or your own) returned `ToolDecision::Terminate`. It
    now also names the refused tool.

  `AbortReason` is `#[non_exhaustive]`. Its `Display` renders exactly
  the old strings (`"timeout exceeded after 30s"`,
  `"cancelled by caller"`, or the middleware reason verbatim), so text
  shown to users or to a parent model (for example
  `SubAgentTool`'s `"sub-agent aborted: …"`) is unchanged.
  `JsonTracer`'s `run_finished` payload keeps `reason.detail` and
  gains `reason.abort_kind` (`timeout` / `cancelled` / `terminated` /
  `tool_terminated` / `max_iterations`) plus `reason.tool_name` for
  tool terminations.
- Reaching `RunConfig::max_iterations` (or `RunOptions::max_iterations`
  / `SubAgentConfig::max_iterations`) is now an abort, not an error.
  The run returns `Ok` with
  `FinishReason::Aborted(AbortReason::MaxIterations(n))` instead of
  `Err(EngineError::MaxIterationsExceeded(n))`, and
  `EngineError::MaxIterationsExceeded` is removed. This matches the
  "aborts are not errors" contract and the `RunConfig::max_iterations`
  docs, which already promised `Aborted`. Behavior follows the other
  aborts:
  - Every completed tool_use/tool_result pair is kept in
    `new_messages`, and `Conversation` persists it to history. The
    `Err` path used to drop that work.
  - `on_run_finished` fires exactly once. `on_run_error` no longer
    fires for this case.
  - `SubAgentTool` now returns
    `"sub-agent aborted (agent loop exceeded max iterations (n)): <partial text>"`
    (still `is_error: true`) instead of `"sub-agent error: …"`.
    Whatever the child found before the cap now reaches the parent
    model.

### Fixed

- A `system_prompt` set by a user middleware in `on_chat_request` is no
  longer silently discarded. The internal system-prompt middleware runs
  after user middlewares (it has to, so tool-group sections reflect the
  final `req.tools`) and used to overwrite `req.system_prompt` with the
  builder prompt. It now composes: the builder prompt goes first — a
  stable prefix that keeps prompt-cache hits — and the user's prompt is
  appended after it. `Plain + Plain` is joined with a blank line; a
  `SystemPrompt::Blocks` overlay produces `Blocks`, with the builder
  prompt as a leading block and the user's blocks (and their
  `cache_control` breakpoints) preserved as-is. When the builder prompt
  renders empty, the user's prompt reaches the model untouched. User
  middlewares still do not observe the builder prompt in
  `on_chat_request`, since it is assembled after them.
- `RunOptions::max_tokens` and `SubAgentConfig::max_tokens` are no
  longer silently ignored when the conversation was built with
  `ConversationBuilder::max_tokens`. The builder default used to be
  applied by the internal request-defaults middleware, which overwrote
  `req.max_tokens` unconditionally on every request, so the per-call
  override always lost. **Behavior change:** the effective cap is now
  resolved before the run starts with the precedence `RunOptions` >
  builder > engine default (4096); a user middleware rewriting
  `req.max_tokens` in `on_chat_request` still wins over all of them.
  Callers that passed both and relied on the builder value winning
  must drop the per-call override. Because the resolved value is what
  `RunConfig::max_tokens` carries, `JsonTracer`'s `run_started` event
  now reports the cap actually sent instead of the engine default.

### Migration

```rust
// Before
let ctx = ToolContext::new(run_id, step_id, activation);

// After
use ailoop::CancellationToken;
let ctx = ToolContext::new(run_id, step_id, activation, CancellationToken::new());
```

`FinishReason::Aborted` carries `AbortReason`: match the variant, or
call `.to_string()` where the old `String` was used.

```rust
// Before
match outcome.finish_reason {
    FinishReason::Aborted(msg) if msg.starts_with("timeout") => retry_later(),
    FinishReason::Aborted(msg) => log::warn!("aborted: {msg}"),
    _ => {}
}

// After
use ailoop::AbortReason;
match outcome.finish_reason {
    FinishReason::Aborted(AbortReason::Timeout(_)) => retry_later(),
    FinishReason::Aborted(reason) => log::warn!("aborted: {reason}"),
    _ => {}
}
```

`max_iterations` no longer produces an `Err`; check the outcome
instead. Because the partial turns are now persisted, drop any code
that re-appended them by hand.

```rust
// Before
match chat.run(input).await {
    Err(EngineError::MaxIterationsExceeded(n)) => hit_cap(n),
    Err(e) => return Err(e.into()),
    Ok(outcome) => use_answer(outcome),
}

// After
let outcome = chat.run(input).await?;
match outcome.finish_reason {
    FinishReason::Aborted(AbortReason::MaxIterations(n)) => hit_cap(n),
    _ => use_answer(outcome),
}
```

- `Conversation::stream_with_options` / `run_with_options` plus
  `RunOptions` (`ailoop::RunOptions`): per-call overrides for
  `timeout`, `cancellation`, `max_iterations`, `max_tokens`, and a
  caller-minted `RunId`. Previously the only way to attach a timeout
  or a `CancellationToken` to a `Conversation` run was to drop into
  `advanced::run_chat` and bypass the façade's middleware composition
  entirely. The new options are deliberately narrower than
  `RunConfig`: `middlewares` and `system_prompt` are owned by the
  builder and stay there; the escape hatch for engine-level control
  remains `advanced::run_chat`. Cancellation interrupts every await
  (HTTP setup, SSE chunks, tool execution, retry backoff) under the
  engine's `select!`, so a run can be aborted mid-backoff.
  `stream()` / `run()` keep their signatures and delegate to the new
  methods with `RunOptions::default()` for a no-op overlay.
- `ReasoningEffort` typed knob on `ChatRequest`, surfaced through
  `ConversationBuilder::reasoning_effort(...)`. Variants
  `Minimal | Low | Medium | High` map cross-provider; `Budget(u32)`
  gives exact control over Anthropic's `thinking.budget_tokens` and
  bucketises into the closest Chat Completions string. Mapping table
  documented inline on the enum. Adapters that don't surface a
  reasoning control ignore the field. `ChatRequest` is
  `#[non_exhaustive]`, so the addition is non-breaking; the previous
  `additional_params` escape hatch keeps working.
- `MaxToolCalls` middleware (`ailoop::MaxToolCalls`): flat cap on the
  *total* number of tool invocations across an entire run.
  `RunConfig::max_iterations` only counts steps, so a turn with 30
  parallel tool calls still burns one iteration — `MaxToolCalls`
  closes that gap. On the (N+1)-th call the middleware returns
  `ToolDecision::Terminate`, which the engine surfaces as
  `FinishReason::Aborted` while preserving prior tool results.
  Composes with `AntiLoop`.
- `TimeoutTool<T: ToolDyn>` in `ailoop-tools` (re-exported as
  `ailoop::TimeoutTool`): per-tool wall-clock cap that wraps any
  `ToolDyn`. When the wrapped call exceeds its budget the wrapper
  returns an `is_error: true` `ToolResultContent` and the engine
  feeds the error back to the model — the run keeps going. Distinct
  tools (e.g. `get_weather` vs `run_terraform_apply`) deserve
  distinct caps; the run-wide `RunConfig::timeout` stays the right
  knob for the overall run.
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
