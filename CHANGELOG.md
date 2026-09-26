# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release contains breaking changes. Every entry describes the
change against 1.0.0-rc.3. The Migration section at the end starts with
an upgrade checklist and has a before/after snippet for each breaking
change.

### Added

- `RunOptions` with `Conversation::run_with_options` /
  `stream_with_options`: per-call overrides for `timeout`,
  `cancellation`, `max_iterations`, `max_tokens` and a caller-minted
  `RunId`. Before, attaching a timeout or a `CancellationToken` to a
  `Conversation` run meant dropping into `advanced::run_chat` and
  bypassing the conversation's middleware composition. The options are
  deliberately narrower than `RunConfig`: middlewares and the system
  prompt belong to the builder, and `advanced::run_chat` remains the
  escape hatch for engine-level control. Cancellation interrupts every
  await (HTTP setup, stream chunks, tool execution, retry backoff), so a
  run can be aborted mid-backoff. `run()` / `stream()` delegate to them
  with `RunOptions::default()`.

- `ConversationBuilder::max_iterations(n)`: a default iteration cap for
  every run of a conversation, like `ConversationBuilder::max_tokens`.
  Precedence, highest first: `RunOptions::max_iterations` (and
  `SubAgentConfig::max_iterations`, which maps to it) > the builder
  default > the engine default. The resolved value is what
  `RunConfig::max_iterations` carries, so `on_run_started` observers and
  `JsonTracer` see the cap in effect. On a sub-agent's child
  conversation, it sets the child's default cap.

- `DEFAULT_MAX_ITERATIONS` (`usize`, 25) and `DEFAULT_MAX_TOKENS`
  (`u32`, 4096). They name the defaults `RunConfig::default()` uses (and
  `ChatRequest::default()` for `max_tokens`), which were bare literals,
  so code that derives its own limits from them (for example "twice the
  default" for a sub-agent) follows any future change instead of copying
  the number. Same style as `DEFAULT_HISTORY_MAX_TOKENS`.

- Multimodal kickoff: `Conversation::run`, `run_with_options`, `stream`
  and `stream_with_options` take `impl Into<Message>` instead of
  `impl Into<String>`. `Message` gains `From` impls for `&str`, `String`,
  `UserBlock` and `Vec<UserBlock>`, and `Message::user_with_blocks(blocks)`
  builds a multi-block user turn (text + image, text + PDF) without an
  attachment middleware. Existing `&str` / `String` callers compile and
  behave the same.

- `ReasoningEffort` on `ChatRequest`, set through
  `ConversationBuilder::reasoning_effort(...)`. `Minimal`, `Low`,
  `Medium` and `High` map across providers; `Budget(u32)` sets
  Anthropic's `thinking.budget_tokens` exactly and maps to the closest
  Chat Completions level. The mapping table is on the enum. Adapters
  without a reasoning control ignore it; `additional_params` keeps
  working.

- `RunError<E>`: the error of a failed run. It wraps the `EngineError`
  cause (`kind()`, `into_kind()`), what the run spent (`usage()`) and
  the messages of the steps it completed before failing
  (`partial_messages()`). `into_parts()` returns a `#[non_exhaustive]`
  `RunErrorParts` with the three owned parts; destructure it with `..`.
  - `partial_messages()`: until now those steps were lost when a run
    failed. The history is rolled back on `Err`, so the record of tools
    that already ran (writes, API calls) disappeared and the next turn
    could repeat them. The list is exactly the `new_messages_so_far` of
    the last `StepFinished`: every `tool_use` in it has its
    `tool_result`, and the step that failed is left out, including any
    text it streamed. A model error in the middle of a response arrives
    before that step's tools run, so no executed tool is missing. The
    rollback guarantee is unchanged; the list is a copy the caller can
    keep or drop.
  - `usage()`: counted like `RunFinished.usage` (finished provider turns
    plus usage reported by tools, sub-agents included). Before, a run
    ending in `Err` lost its usage entirely, so a token or cost budget
    was blind exactly on the runs that fail. The turn that failed is not
    counted: a response cut off by an error never reports its usage,
    although the provider may bill it. The usage can include the failed
    step's turn and tool reports (for example when a tool registry error
    follows a finished turn) even though that step is left out of
    `partial_messages`. Errors raised before the run starts (history
    compaction in `stream_with_options`) carry zero usage.

- `Conversation::extend_messages(messages)`: append several messages
  without running, like `push_message`. Use it to keep the steps of a
  failed run: `chat.extend_messages(err.partial_messages().iter().cloned())`.

- `ChatMiddleware::on_turn_end` and `ContinueDecision`: native support
  for the "completion gate" pattern. When the model ends a turn with no
  tool calls to run, the engine asks each middleware, in registration
  order, before it finishes the run. Returning
  `ContinueDecision::Continue { blocks }` (or
  `ContinueDecision::continue_with(text)`) adds a user message and runs
  another iteration of the same run. The first `Continue` wins, and the
  remaining middlewares are not asked for that turn. A `Continue` with
  no blocks counts as `Stop` (and the next middleware is asked), since
  continuing without a new user message would send a request that ends
  on the assistant's own turn. Until now this took an outer loop around
  the run, which reset `iteration` to 0, split `usage` across runs and
  emitted one `RunFinished` per attempt. With the hook, iterations keep
  counting, usage accumulates and a single `RunFinished` closes the run.
  The injected message is part of the history and of `new_messages`,
  and shows up in the `StepFinished` of the step that asked to continue.
  The hook fires for `EndTurn`, `MaxTokens`, `StopSequence` and `Other`,
  and receives the reason so a gate can filter. It never fires for
  `ToolUse` or `Aborted`. Every continuation counts against
  `max_iterations`, so a gate that never passes ends the run with
  `AbortReason::MaxIterations`. If the turn also completed tool calls
  (possible with `MaxTokens`), the injected blocks join the tool results
  in one user message. A turn that produced no assistant content at all
  (an empty `EndTurn`) leaves nothing between the previous user message
  and the injected one; providers merge consecutive user turns. A gate
  in a `SubAgentTool` child can override the wrap-up turn of
  `SubAgentConfig::wrap_up`, so it should let `EndTurn` through once the
  budget is nearly spent. The default returns `ContinueDecision::Stop`.
  The rustdoc of `on_chat_request` also explains how to track the
  current iteration: record `StepStarted { iteration }` from `on_chunk`
  and `max_iterations` from `on_run_started`, keyed by `run_id`.

- `ChatMiddleware::on_run_dropped(&self, run_id)`: a third closing hook,
  fired when the caller drops a run's stream before the run closed (a
  `select!` that picks another branch, an outer timeout, a client that
  disconnects). Before, a dropped run fired neither `on_run_finished`
  nor `on_run_error`, so a middleware that kept per-run state keyed by
  `RunId`, the pattern the trait docs recommend, never released it and
  a long-lived conversation grew it without bound. Each middleware now
  gets exactly one of `on_run_finished`, `on_run_error` or
  `on_run_dropped` per run; a drop during the closing hooks only reaches
  the middlewares not called yet. The hook is synchronous because it
  runs inside `Drop`: it must not block, only clean up (use a
  `std::sync::Mutex` or `try_lock`). A stream dropped before its first
  poll never started a run and fires nothing. The default does nothing.

- `ToolCallInfo`: the identity of one tool call, passed to every tool
  hook of `ChatMiddleware`. Public fields `run_id`, `step_id`, `call_id`
  (the provider-assigned id, the same as `call_id` on
  `ToolCallFinished` and `ToolResult`) and `name`. It is
  `#[non_exhaustive]` so more per-call data can be added without
  breaking middlewares; `ToolCallInfo::new` builds one to unit-test a
  hook.

- `JsonTracer` and `TracingMiddleware` report more of each run, all
  additive within `schema: 1`:
  - `call_id` on `before_tool_call` / `after_tool_call` lines and on the
    "tool call starting" / "tool call finished" events. Before, those
    could not be matched to `tool_call_finished` / `tool_result`, or to
    each other when a step called the same tool twice.
  - `usage` and `partial_messages` (a count) on `run_error`
    (`input_tokens` / `output_tokens` / `partial_messages` fields on the
    tracing event).
  - `tool_call_malformed` lines (with `raw` only in verbose mode) and a
    `warn` event for `StreamChunk::ToolCallMalformed`.
  - `reason.abort_kind` (`timeout` / `cancelled` / `terminated` /
    `tool_terminated` / `max_iterations`) on `run_finished`, plus
    `reason.tool_name` and `reason.call_id` for tool terminations.
    `reason.detail` is unchanged.

- `RunConfig` implements `Debug`. Middlewares are shown as a count,
  since they are trait objects.

- `ToolContext::cancellation() -> &CancellationToken`: the run's
  cancellation handle (the token passed through `RunOptions` /
  `RunConfig`, or a never-cancelled one), cloned into every dispatch.
  The engine already drops the tool future on cancellation, which
  cancels in-flight async I/O on its own; this token covers what
  dropping does not reach: `spawn_blocking` work, `tokio::process`
  children that need an explicit signal, ordered cleanup, and `JoinSet`
  fan-out that hands `child_token()` to each task.

- `ToolContext::call_id()`: the id of the call being dispatched, so a
  tool can correlate its own logs or side effects with the `ToolResult`
  chunk and the middleware hooks. `ToolContext::detached` mints a
  synthetic one.

- `ToolContext::report_usage(Usage)` and `UsageSink`: a tool that spends
  tokens outside the engine's own provider turns (a sub-agent, a tool
  that calls an LLM directly) reports them here and the engine adds them
  to the run's usage total. Reports count as soon as they are made, so a
  tool dropped by a timeout or cancellation still contributes what it
  already spent. `ToolContext::usage_sink()` exposes the sink so a tool
  can hand it to a nested run or spawned task;
  `ToolContext::with_usage_sink(sink)` replaces it, which is useful in
  tests to read back what a tool reported.

- `ToolRegistry::all_tools()`: every registered tool, active and
  inactive, in registration order. It is the catalog a handler can reach
  with `ToolActivation::activate`, so policies that must cover every
  tool a run could dispatch (like approval gating) should be computed
  over it rather than over `active_tools()`.

- `ToolRegistry::retain_by_tags(tags)`: unregister every tool whose tags
  do not overlap with `tags`. Unlike `deactivate_by_tags`, the removed
  tools can no longer be dispatched or activated at runtime.
  `ConversationBuilder::capabilities` is built on it.

- `ToolRegistry` implements `Default` (an empty registry, same as
  `ToolRegistry::new()`), so it works with `..Default::default()`,
  `#[derive(Default)]` and `std::mem::take`.

- `TimeoutTool<T: ToolDyn>`: a per-tool wall-clock cap around any
  `ToolDyn`. When the call exceeds its budget the wrapper returns an
  `is_error: true` result and the run keeps going, so the model can
  react. Different tools deserve different caps; `RunConfig::timeout`
  stays the knob for the whole run.

- `MaxToolCalls` middleware: a cap on the total number of tool calls in
  a run. `max_iterations` counts steps, so one turn with 30 parallel
  calls uses a single iteration. On the (N+1)-th call it returns
  `ToolDecision::Terminate`, which ends the run as
  `FinishReason::Aborted(AbortReason::ToolTerminated { .. })` with the
  earlier results kept. Composes with `AntiLoop`.

- `AntiLoop::with_tool_call_identity(|name, args| -> String)`: a
  pluggable equivalence key for the tool-call loop detector. The streak
  counter compares the returned strings instead of the arguments'
  structural equality, which catches agents that repeat a destructive
  call with cosmetic variation (whitespace inside a string, reordered
  keys in an embedded JSON payload, ignored auxiliary fields). The
  terminate reason includes the computed key. Without it, behavior and
  wording are unchanged.

- `StreamChunk::ToolCallMalformed { call_id, name, raw, error }`: closes
  a streamed tool call whose argument text is not a JSON object, in
  place of `ToolCallFinished` (see Fixed).
  `StreamChunk::tool_call_from_raw_args(call_id, name, raw)` builds the
  right closing chunk from the accumulated text; both built-in adapters
  use it and third-party adapters should too. Empty or whitespace-only
  text is still a valid `{}` (tools without parameters).

- `ConversationBuilder::tools_with_prompt_file(names, path)`: one
  `PromptSection` read from disk for a group of tools. It is appended to
  the system prompt at most once per turn when at least one tool of the
  group is active. With `tool_with_prompt_file`, which is per tool,
  a guide shared by N tools was emitted N times. Sections render in
  group registration order. It does not register the tools; pair it
  with `.tool(...)`. An empty `names` fails at `build()` with the new
  `BuildError::EmptyToolGroup`.

- `ApprovalRequest`: what an approval callback receives for one gated
  call. Public fields `run_id`, `step_id`, `call_id`, `name` (the tool's
  wire name), `args`, `tags` (the tool's declared tags) and `messages`
  (the context sent to the model on the step that produced the call,
  after every middleware's `on_chat_request`). `call_id` tells apart two
  identical calls in one step and matches the decision to the
  `ToolResult` chunk. The type is `#[non_exhaustive]`;
  `ApprovalRequest::new(ToolCallInfo, args)` plus `with_tags` /
  `with_messages` build one outside the crate, e.g. to unit-test a
  verifier. Its rustdoc describes a model-based risky-action verifier:
  tags pick what gets reviewed, the verifier allows, denies or escalates
  to a human, it fails closed on error or timeout, and it reads intent
  from the user's text only, never from tool results, which can carry
  injected instructions.

- `Usage::new(input_tokens, output_tokens)` (a `const fn`) builds a
  `Usage` with the other counters at zero, so a tool outside the crate
  can write `ctx.report_usage(Usage::new(1_200, 350))` instead of
  starting from `Usage::default()`. `Usage` also implements `Sum` (over
  `Usage` and `&Usage`, saturating like `+`) and derives `PartialEq` and
  `Eq`.

- `Usage::reasoning_tokens`: the part of `output_tokens` spent on hidden
  reasoning. Azure OpenAI fills it when the deployment reports
  `completion_tokens_details.reasoning_tokens` (o-series, gpt-5).
  Anthropic folds reasoning into `output_tokens`, so it stays `0` there.

- `ProviderError` trait with `is_context_overflow()`, which defaults to
  `false`. It answers "did the provider reject this prompt because it
  does not fit the context window?" without the caller knowing the
  adapter's error type, and is what lets `Conversation` recover from an
  overflow (see below). It is separate from `Retryable`: an overflow is
  `Permanent` for `RetryingModel`, because resending the same prompt
  fails the same way, but a caller that owns the history can still
  recover. `AnthropicError`, `AzureOpenAIError`, `ScriptedError` and
  `std::convert::Infallible` implement it; `ScriptedError` reports an
  overflow when its message contains `"context_overflow"`.

- `AnthropicApiErrorKind::ContextOverflow`: a 400
  `invalid_request_error` whose message contains "prompt is too long".
  It was classified as `InvalidRequest`. It is detected for both HTTP
  error bodies and mid-stream error events. The new
  `AnthropicApiErrorKind::from_error(type, message)` does the
  message-aware mapping; `from_error_type` still returns
  `InvalidRequest`, because the type alone cannot tell the two apart.

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
  error (`CharTokenizer` is a `len() / 4` heuristic). Now `max_tokens`
  can be the model's context window and the slack is stated explicitly.
  The threshold is `max_tokens.saturating_sub(reserved)`, so a reserve
  at or above `max_tokens` never panics: every call compacts, and
  `CompactionError::NotEnoughHistory` surfaces once only the preserved
  tail is left. The reserve is builder configuration, not snapshot
  state: it applies through `History::from_messages` and
  `ConversationBuilder::history` (including after `from_snapshot`), and
  is not persisted in `ConversationSnapshot`.

- `History::force_compact()`: runs the compaction strategy regardless of
  the token budget, for when the estimate said the history fits and the
  provider disagreed. It returns the same `CompactionStats` and errors
  as `compact_if_needed`, which now delegates to it. The strategy may
  still be unable to shrink the history (everything before the preserved
  tail is pinned, or the tail itself is what does not fit), so compare
  `estimated_tokens()` before and after when that matters.

- `History::needs_compaction()`: the budget check `compact_if_needed`
  runs (`estimated_tokens() >= max_tokens - reserved_tokens`), without
  running the strategy.

- `History::replace_messages(messages, pinned)`: the in-place
  counterpart of `History::from_messages`. It swaps the messages and pin
  mask and keeps the budget, strategy and tokenizer, for rolling back to
  a captured state or loading a snapshot into a live history. If the
  lengths differ it returns `FromMessagesError::LengthMismatch` and
  leaves the history untouched.

- In-run context management for `Conversation`. Before, the history was
  compacted once, before the run, and each tool result grew the context
  until the provider rejected it. Two builder switches cover this:
  - `ConversationBuilder::compact_between_iterations(bool)` (default
    `false`): checks the history budget before every model call after
    the first, not only at run start. When it compacts, the stream
    carries a `StreamChunk::HistoryCompacted` between the previous
    `StepFinished` and the next `StepStarted`. The built-in strategies
    cut only at a user message that is not a tool result, which inside a
    run means the run's own kickoff, so mid-run compaction reclaims
    earlier turns, never splits the run's tool_use / tool_result pairs,
    and cannot shrink the run itself. If a compaction leaves the history
    over budget, or does not reduce it, the check is skipped for the
    rest of the run. It is off by default because each compaction has a
    cost (an extra model call with `SummarizeStrategy`).
  - `ConversationBuilder::recover_from_context_overflow(bool)` (default
    `true`): when opening the model stream fails with an error whose
    `is_context_overflow()` is `true`, the engine forces a compaction,
    emits `HistoryCompacted`, and reissues the request once.
    `on_chat_request` runs again for the same step.

  When the overflow cannot be recovered (the retry overflowed too, or
  the forced compaction had nothing to drop or did not reduce the
  estimate), the run fails with the new `EngineError::ContextOverflow(E)`,
  carrying the provider's last error. The typical cause is a single turn
  (for example a huge tool result) larger than the window. The history
  is rolled back to its state at run start, which also undoes the
  compactions done during the run.

- `SubAgentConfig` and `SubAgentTool::with_config(name, description,
  child, config)`: per-invocation budgets for a sub-agent (`timeout`,
  `max_iterations`, `max_tokens`, `wrap_up`), so the parent can cap a
  runaway child without rebuilding the child `Conversation`. Each call
  builds a fresh `RunOptions` from the config. `SubAgentTool::new` uses
  `SubAgentConfig::default()`, which overrides nothing.

- `SubAgentConfig::wrap_up(WrapUp)`: an opt-in graceful cutoff. Without
  it, when a child hits its `timeout` or `max_iterations`, the parent
  gets `"sub-agent aborted (…)"` with `is_error: true`, and whatever the
  child found after its last text block is lost. With it, the child's
  last turn before the hard cutoff is forced to be a summary: it runs on
  the last allowed iteration, or on the first request after
  `timeout * WrapUp::time_fraction` (default `0.8`). That request sets
  `tool_choice` to `ToolChoice::None` but keeps the tool definitions, so
  the prompt cache survives, and appends `WrapUp::instruction` (default
  `DEFAULT_WRAP_UP_INSTRUCTION`) to the last user message, only in that
  request, never in the child's history. If the summary finishes, the
  parent receives `"[partial: sub-agent reached its time budget]\n<summary>"`
  (or `iteration budget`) with `is_error: false`; the signal lives in
  the text because the parent model only sees the tool result. The hard
  timeout is still absolute: if it fires before the summary is done, the
  result is the usual abort.

- `SubAgentTool` accepts an optional `attachments` array next to
  `prompt`, turned into image and document blocks on the child's
  kickoff message. Each entry uses Anthropic's content shape:
  `{"type": "image" | "document", "source": {"type": "base64",
  "media_type": …, "data": …} | {"type": "url", "url": …} |
  {"type": "file_id", "id": …}}`. Malformed attachments come back as a
  `"sub-agent error: invalid attachments: …"` result with
  `is_error: true`. The child's reply is still text only.

- `CachingTokenProvider` (azure-openai): wraps any `TokenProvider` and
  reuses its token for `ttl - refresh_skew`
  (`CachingTokenProvider::new(inner, ttl, refresh_skew)`). The
  `TokenProvider` docs told callers to cache, but the crate offered no
  way to do it. The refresh happens outside the lock, so an expired
  cache hit by a concurrent burst may fetch more than once; a failed
  fetch is not cached.

- Re-exports from the `ailoop` facade, so users no longer need to depend
  on the sub-crates to name them: `CacheControl`, `SystemBlock`,
  `SystemPrompt` and `ToolResultBlock` (custom system prompts and
  multi-block tool results with cache breakpoints); `CompactionOutput`,
  `CompactionStats` and `DEFAULT_SUMMARIZER_PROMPT` (custom
  `CompactionStrategy`s); and `PromptError` (the payload of
  `BuildError::Prompt`). Every public item of the sub-crates is now
  reachable through `ailoop`.

- `ailoop::async_trait`: the facade re-exports the `async_trait`
  attribute macro. `ChatMiddleware`, `CompactionStrategy` and
  `CompletionModel` are `async_trait` traits, so implementing one meant
  adding `async-trait` to your own `Cargo.toml` and keeping its version
  in line with ailoop's. Write `#[ailoop::async_trait]` on the `impl`
  block instead; the expansion does not refer to the `async-trait`
  crate, so no direct dependency is needed.

- `testing` feature on `ailoop`, which exposes `ailoop::testing` (a
  re-export of `ailoop_core::testing`: `ScriptedModel`, `ScriptedError`,
  `ScriptedTurn`), so you can test your own middlewares and tools
  against a scripted run without depending on `ailoop-core`. Enable it
  under `[dev-dependencies]`:
  `ailoop = { version = "…", features = ["testing"] }`.

- `HookAction` and `ToolDecision` derive `Debug`, and `FinishReason`
  derives `PartialEq` and `Eq`, so a test can write
  `assert_eq!(outcome.finish_reason, FinishReason::EndTurn)`. The
  decision enums get only `Debug` on purpose: a future variant may carry
  a value that cannot be cloned or compared, and adding a derive later
  is compatible while removing one is not.

- Minimum supported Rust version: 1.88, declared as `rust-version` on
  every crate. Edition 2024 needs 1.85, and the let chains used across
  the crates were stabilized in 1.88. Cargo now reports an old toolchain
  up front instead of failing on a syntax error, and the MSRV-aware
  resolver picks dependency versions that build on 1.88. CI checks it.

### Changed

- `RunFinished.usage`, `RunOutcome.usage` and the usage passed to
  `on_run_finished` now mean **everything the run spent**: its own
  provider turns plus usage reported by tools. `SubAgentTool` reports
  its child run's usage, and nested sub-agents roll up recursively.
  Before, the child's tokens were dropped and the parent's total
  undercounted the real cost. The spend is forwarded while it happens,
  so it also counts when the child aborts and when the parent aborts
  while the child is still running. `TurnFinished.usage` is unchanged:
  it covers only the run's own model; sum it in a middleware to split
  own vs. delegated spend. If you were adding a sub-agent's usage to the
  parent's total yourself, stop, or it will be counted twice.

- `SubAgentTool` passes the parent's cancellation to the child run (a
  `child_token()` of `ToolContext::cancellation`), so cancelling or
  timing out the parent stops an in-flight sub-agent at its next await
  instead of letting it keep spending. A child run that aborts or fails
  now comes back with `is_error: true` (the text is the same
  `"sub-agent aborted …"` / `"sub-agent error: …"`), so the parent model
  can tell a failure from a normal reply.

- Documented contract: when the model requests several tools in one
  turn, the order **between** those calls is not guaranteed. The engine
  still runs them one at a time in the model's order; the contract is
  written down now so a later 1.x release can run a step's tools
  concurrently without a breaking change. What stays guaranteed: the
  hook order within one call (`on_before_tool_call_mut` →
  `on_before_tool_call` → tool → `on_after_tool_call_mut` →
  `on_after_tool_call` → `ToolResult` chunk), tool results in history in
  the order of the model's calls, and every call of a step finishing
  before `on_turn_end` or the next `on_chat_request`. Middlewares with
  per-call state should key it by `(run_id, call_id)` behind a lock. See
  "Tool calls within a step" on `ChatMiddleware`.

- The docs of `StreamChunk::TurnFinished` and `Usage` state that the
  per-turn chunk reaches only `ChatMiddleware::on_chunk`: the engine
  adds its usage to the run total and drops it before the public stream,
  so stream consumers see the total on `RunFinished.usage`.

### Changed (BREAKING)

- `Conversation::run`, `run_with_options`, `stream`,
  `stream_with_options`, the `RunStream` error item and
  `advanced::run_chat` (its result and its stream items) fail with
  `RunError<M::Error>` instead of `EngineError<M::Error>`, so the steps
  completed before the failure and the run's usage reach the caller (see
  `RunError` under Added). `From<RunError<E>> for EngineError<E>` keeps
  `?` working in functions that return `EngineError`; only code that
  matches the error directly needs `.kind()` or `.into_kind()`.

- The run-level hooks of `ChatMiddleware` take one context struct in
  place of positional arguments: `on_run_started(&RunStartInfo)`,
  `on_chat_request(&StepInfo, req)`, `on_run_finished(&RunFinishedInfo)`
  and `on_run_error(&RunErrorInfo)`; the new `on_turn_end` takes
  `&TurnEndInfo`. Each positional argument added to a hook broke every
  middleware that overrode it. The structs are `#[non_exhaustive]`, so
  later releases can add fields without breaking implementations. Their
  fields are public and keep the old parameters' names, except `err`,
  which is now `error`. `RunErrorInfo` also carries the run's `usage`
  and `partial_messages` (the same values as `RunError`), so a
  persistence or accounting middleware sees what a failed run did and
  spent; `on_run_error` still fires only for runs that started, so a
  pre-run compaction error reaches the caller without it. The borrowed
  structs borrow from the engine for the duration of the hook, so
  nothing is cloned, and each has a `new(...)` constructor for
  unit-testing a middleware. `ChatRequest` stays a separate argument of
  `on_chat_request` because the hook borrows it mutably.

- The four tool hooks of `ChatMiddleware` take a `&ToolCallInfo` in
  place of `run_id, step_id, name`: `on_before_tool_call(call, args)`,
  `on_before_tool_call_mut(call, args)`,
  `on_after_tool_call(call, args, result)` and
  `on_after_tool_call_mut(call, args, result)`. The hooks never saw the
  provider's call id, so a middleware could not pair an
  `on_after_tool_call` with its `on_before_tool_call`: two calls to the
  same tool with the same arguments in one step were indistinguishable.
  A struct rather than one more parameter lets later per-call data be
  added without breaking the trait again.

- Approval callbacks take an `ApprovalRequest` instead of
  `(String, Value)`: `ConversationBuilder::approval`,
  `approval_for_tags`, `approval_for_all`,
  `ApprovalMiddleware::approve_all` and `approve_named`. A callback that
  only saw the tool name and arguments could not tell a reasonable call
  from a dangerous one: `rm -rf build/` is fine after "clean the build"
  and alarming after "summarize this file". To fill `messages`,
  `ApprovalMiddleware` records each step's request in `on_chat_request`,
  keyed by run so one instance can be shared across concurrent runs, and
  drops it when the run closes. The messages are copied once per step
  into an `Arc<[Message]>` shared by every gated call of that step, and
  only when a gate is installed. `tags` is filled for gates installed
  through the builder, which sees the whole tool catalog; `approve_all`
  and `approve_named` leave it empty.

- `FinishReason::Aborted` carries a structured `AbortReason` instead of
  a `String`, so callers match on why a run stopped instead of parsing
  text:
  - `Timeout(Duration)`: the run's timeout elapsed.
  - `Cancelled`: the `CancellationToken` fired. It wins over a timeout
    that fires at the same instant.
  - `Terminated { reason }`: a middleware returned
    `HookAction::Terminate` from `on_run_started`.
  - `ToolTerminated { tool_name, call_id, reason }`: a middleware
    (`AntiLoop`, `MaxToolCalls`, or your own) returned
    `ToolDecision::Terminate` for that call.
  - `MaxIterations(n)`: the run reached its iteration cap (see next
    entry).

  `AbortReason` is `#[non_exhaustive]`, and so are its struct variants:
  match them with `..` and build them with `AbortReason::terminated` /
  `AbortReason::tool_terminated`. Its `Display` renders the old strings
  (`"timeout exceeded after 30s"`, `"cancelled by caller"`, or the
  middleware reason verbatim), so text shown to users or to a parent
  model is unchanged.

- Reaching `max_iterations` is an abort, not an error. The run returns
  `Ok` with `FinishReason::Aborted(AbortReason::MaxIterations(n))`
  instead of `Err(EngineError::MaxIterationsExceeded(n))`, and that
  variant is removed. This matches the "aborts are not errors" contract
  the `RunConfig::max_iterations` docs already promised. It behaves like
  the other aborts: every completed tool_use / tool_result pair is kept
  in `new_messages` and `Conversation` persists it (the `Err` path used
  to drop that work), `on_run_finished` fires exactly once and
  `on_run_error` no longer does, and `SubAgentTool` returns
  `"sub-agent aborted (agent loop exceeded max iterations (n)): <partial text>"`
  instead of `"sub-agent error: …"`, so what the child found before the
  cap reaches the parent model.

- `RunConfig::default().max_iterations` is 25 (was 10). One iteration is
  one model turn plus the tool calls it triggers; every
  `ContinueDecision::Continue` also counts, and a sub-agent's wrap-up
  uses the last iteration. So 10 ran out on ordinary agent tasks (read,
  search, edit, verify), and the run then ended with an easy-to-miss
  `Ok` / `Aborted(MaxIterations(10))`. The new default applies to every
  run that does not set a cap, including `SubAgentTool` children with no
  `SubAgentConfig::max_iterations`. It is a safety brake, not a budget:
  without a token or cost cap, the worst-case run of a caller who relies
  on the default is now 2.5× longer. In production, set
  `max_iterations` explicitly, together with `timeout` and
  `MaxToolCalls`.

- A context-window overflow no longer ends a `Conversation` run as a
  plain model error. With the default
  `recover_from_context_overflow(true)`, the engine compacts and retries
  once, and fails with `EngineError::ContextOverflow` if that does not
  help (see Added). As a consequence:
  - `Conversation`'s methods and `SubAgentTool`'s `ToolDyn` impl require
    `M::Error: ProviderError`. The built-in adapters, `ScriptedError` and
    `Infallible` implement it; a custom model's error type needs an
    `impl ProviderError for MyError {}`, whose default reports no
    overflow. `advanced::run_chat` has no history to compact and gains
    no bound.
  - `on_chat_request` can fire twice for the same `step_id` when an
    overflow is recovered. The retried request is rebuilt from the
    compacted history.

- The engine only runs tools in the run's active set. A call to a
  registered but inactive tool (deferred with `initial_active_tools` and
  not yet activated) no longer runs it: the model gets the same error
  result as for an unknown name (`Tool '<name>' not found. Available
  tools: [...]`, listing only active tools) and the run goes on. The
  active set is read when the call is dispatched, so a tool activated by
  an earlier call of the same step can run. Calls rejected this way, and
  calls to unknown names, no longer fire the tool hooks or the approval
  callback; only their `ToolResult` chunk goes out. `MaxToolCalls` and
  `AntiLoop` therefore do not count them; `max_iterations` still bounds
  a model that keeps calling a hidden tool. See Fixed.

- `ConversationBuilder::capabilities` (formerly `with_capabilities`)
  removes the filtered tools from the conversation instead of
  deactivating them. They no longer show up in
  `ToolActivation::list_all` / `list_inactive`,
  `ToolActivation::activate` returns `NotFound` for them, and
  `initial_active_tools` ignores their names. Until now they stayed in
  the catalog, so the "default-deny" promise did not hold (see Fixed).

- The seven token counters on `Usage` (`input_tokens`, `output_tokens`,
  `cached_input_tokens`, `cache_creation_input_tokens`,
  `cache_creation_5m_tokens`, `cache_creation_1h_tokens`,
  `reasoning_tokens`) are `u64` instead of `u32`. A
  single turn never gets near `u32::MAX`, but `Usage` is also an
  accumulator: the run total, the `UsageSink` that adds nested
  sub-agents, and whatever an application sums across runs (per-tenant
  billing over weeks), where about 4.29 billion tokens is reachable.
  `Usage` has no serde derive, so there is no stored format to migrate,
  and `JsonTracer` / `TracingMiddleware` emit the same numbers. Request
  parameters (`max_tokens` on `ChatRequest`, `RunConfig`,
  `SubAgentConfig`, `SummarizeStrategy`) stay `u32`: the provider caps
  them. `OnlineCalibratedTokenizer::observe` takes
  `observed_tokens: u64` to match.

- `CompletionModel` requires `Send + Sync`. Its docs already said so,
  since the engine and `RetryingModel` hold models across `.await`s, but
  the trait did not declare it, so every function generic over a model
  had to spell out `M: CompletionModel + Send + Sync`. Every model that
  worked with the engine already met it.

- The `StreamChunk` variants the engine emits (`RunStarted`,
  `StepStarted`, `StepFinished`, `ToolResult`, `RunFinished`,
  `HistoryCompacted`) are `#[non_exhaustive]`, so their fields can grow
  later. Match them with `..`. Outside `ailoop-core`, build them with
  `StreamChunk::run_started`, `step_started`, `step_finished`,
  `tool_result`, `run_finished` and `history_compacted` (e.g. to feed a
  middleware's `on_chunk` in a test). The variants providers emit
  (`TextDelta`, `ToolCall*`, `Reasoning*`, `TurnFinished`) keep struct
  syntax, so adapters and `ScriptedModel` scripts are unchanged.

- The tool call id is `call_id` everywhere. `AssistantBlock::ToolCall`
  and the `ToolCallStarted` / `ToolCallArgsDelta` / `ToolCallFinished`
  chunks called it `id`, while `UserBlock::ToolResult` and the
  `ToolResult` chunk called the same value `call_id`.
  `AssistantBlock::ToolCall` now serializes the field as `call_id`;
  deserialization still accepts `id`, so snapshots saved with
  1.0.0-rc.3 load unchanged (an rc.3 build cannot read snapshots written
  by this version).

- The modules of `ailoop-core`, `ailoop-history` and `ailoop-tools` are
  private. Every public item was already re-exported at the crate root
  (and from `ailoop`), so each type had two paths and moving a type
  between files would have been a breaking change. Only the root paths
  remain. `ailoop_core::testing` (behind the `testing` feature) and
  `ailoop::advanced` stay public.

- `RunId` and `StepId` keep their `Uuid` private. The public tuple field
  made the representation part of the API, so switching to another id
  scheme would have been breaking. Build one from an outer trace id with
  `RunId::from(uuid)`, and read it back with `as_uuid()` or
  `Uuid::from(id)`.

- `ToolContext::new`, `ToolActivation::new`, `ToolRegistry::catalog_arc`
  and `ToolRegistry::snapshot_active` are hidden from the docs and not
  covered by semver. They are the engine's plumbing: their signatures
  expose `indexmap` types, and `ToolContext::new` now also takes the
  call id and the cancellation token. Tools and tests build a context
  with `ToolContext::detached()`, whose signature is stable.

- Renames, so the 1.0 API follows one naming scheme:
  - `ConversationBuilder` setters drop the `with_` prefix:
    `with_history` → `history`, `with_capabilities` → `capabilities`,
    `with_approval` → `approval`, `with_approval_for_tags` →
    `approval_for_tags`, `with_approval_for_all` → `approval_for_all`.
    Rule for 1.x: builders (types that end in `build()`) use bare nouns;
    value types configured by chaining (`AntiLoop`, `SummarizeStrategy`,
    `ApprovalRequest`, message blocks) keep `with_*`.
  - `Conversation::history_messages()` → `messages()` and
    `history_push(message)` → `push_message(message)`, matching
    `History::messages()` and `ConversationSnapshot::messages`.
  - `ToolRegistry::activate_tool` / `deactivate_tool` → `activate` /
    `deactivate`, matching `ToolActivation` and `activate_by_tags`.
  - `ApprovalMiddleware::for_named` → `approve_named`, next to
    `approve_all`.
  - `PromptSection::with_name(name, content)` → `named(name, content)`:
    it is a constructor, and `with_*` is kept for methods that modify a
    value. `Prompt::sections()` returns `&[PromptSection]` instead of
    `&Vec<PromptSection>`.
  - `CompactionReport` → `CompactionStats`, so it no longer reads as a
    synonym of `CompactionOutput` (what a strategy returns).
  - `ChatRequest::disable_parallel_tool_use` → `parallel_tool_use`, and
    `ConversationBuilder::disable_parallel_tool_use(bool)` →
    `parallel_tool_use(bool)`. The flag now states the positive:
    `Some(false)` limits the model to one tool call per turn,
    `Some(true)` allows several, `None` leaves the provider default. The
    old name copied Anthropic's wire field, so every other provider
    needed a negation.
  - `ToolChoice::None_` → `ToolChoice::None`. A variant is always
    reached through its type, so there was no clash with `Option::None`
    to avoid.

- `ToolDyn::name` returns `&str` instead of `String`, like
  `CompletionModel::name`. The registry and the engine read tool names
  on every request and dispatch, and each call allocated a copy.

- Anthropic adapter:
  - `AnthropicModel` → `AnthropicChatModel`, matching
    `AzureOpenAIChatModel`.
  - `AnthropicApiErrorKind::Api` → `ServerError`, the name the Azure
    adapter already uses for the same case. It is still transient for
    `RetryingModel`.
  - `AnthropicClient::from_env` / `from_env_var` return
    `Result<Self, AnthropicError>` instead of `Result<Self, VarError>`,
    like `AzureOpenAIClient::from_env`. A missing or non-Unicode key is
    the new `AnthropicError::Config(String)`, whose message names the
    variable. `Config` is permanent for `RetryingModel`. Code that
    propagates the error with `?` into `Box<dyn Error>` or `anyhow` is
    unaffected.

- Azure OpenAI adapter: `AzureOpenAIError::Provider` carries
  `kind: AzureOpenAIApiErrorKind` instead of `error_type: String`,
  mirroring `AnthropicError::Provider`. The kind comes from the event's
  `code` (falling back to `type`) with the same mapping as HTTP errors,
  so `RetryingModel` and `is_context_overflow` read it like an `Api`
  error instead of treating every mid-stream failure as permanent.

- The `Api` and `Provider` variants of `AnthropicError` and
  `AzureOpenAIError` are `#[non_exhaustive]`, so the adapters can
  surface more of the response later (a request id, the raw body)
  without another breaking change. Patterns need `..`, and only the
  adapters can build these variants.

### Fixed

- **Security:** tools hidden from the model could still run. The engine
  looked the called name up in the whole catalog, not in the active
  set, so a model that named a deferred tool directly ran it without the
  activation step; a tool filtered out by `with_capabilities` ran if the
  model called it by name or a handler activated it; and
  `initial_active_tools` could make a filtered tool active from the
  first turn. With no approval gate nothing stood in the way. The engine
  now rejects calls to inactive tools in-band, and `capabilities`
  removes the filtered tools from the catalog, so none of these paths
  can run them.

- **Security:** the approval gate from `with_approval` /
  `with_approval_for_tags` let deferred tools run unapproved. The set of
  gated names was resolved at `build()` from the initial active set
  only, so a `Destructive` / `WritesFiles` tool that started inactive
  and was later activated by a handler (the `search_tools` pattern)
  never reached the callback, silently bypassing a `Skip` / `Terminate`
  policy. The gated set is now resolved over the whole catalog.
  `with_approval_for_all` and `ApprovalMiddleware::for_named` were not
  affected.

- Tool calls with malformed arguments no longer run with `{}`. Both
  adapters parsed the accumulated argument JSON with a silent fallback
  to `{}`, so a call truncated by `max_tokens` (or invalid JSON from
  Anthropic's fine-grained tool streaming) ran the tool with no
  arguments, which is unsafe for a tool with side effects. Now the
  adapters emit `StreamChunk::ToolCallMalformed` and the engine never
  runs the tool: it records the call in history with an empty-object
  input (providers require an object there) and answers it with an error
  `tool_result`: `Invalid JSON arguments for tool '<name>': <parser
  error>. The tool was not run; call it again with complete, valid
  JSON.` followed by `{"INVALID_JSON": "<raw>"}` (the wrapper Anthropic
  recommends; `raw` capped at 1 KiB). No tool hook fires for it, so
  approval callbacks, `MaxToolCalls`, `AntiLoop` and
  `Sanitize::on_tool_result` do not see it; the `ToolResult` chunk still
  goes through `on_chunk`. If the turn ended in `ToolUse` the model sees
  the error and can retry in the same run; if it ended in `MaxTokens`
  the run ends as before, with the call and its error paired in history
  (an `on_turn_end` middleware can continue it).

- A run that aborted part-way through a step left tool calls without a
  `tool_result`. When a `ToolDecision::Terminate`, a timeout, a
  cancellation or an abort inside a tool hook stopped the run, only the
  calls that had already finished were answered; the rest stayed as
  `tool_use` blocks with no result in `RunFinished.new_messages`, and so
  did calls the model had finished streaming when the run aborted
  mid-turn. `Conversation` kept that history, so the next request was
  rejected by the provider (HTTP 400). Now every call of the step is
  answered, in the model's order. Completed calls keep their result; the
  others get an `is_error` result reading
  `"Tool not run: the run was aborted (<reason>)"`, where `<reason>` is
  the `AbortReason` as text (for `Terminate`, its `reason`). A call with
  malformed arguments keeps its usual error. Each of these is
  also emitted as a `ToolResult` chunk before `RunFinished`. No tool
  hook fires for them, because the tool never ran.

- `Conversation::run` / `run_with_options` panicked ("engine guarantees
  a RunFinished chunk before the stream terminates") when a middleware's
  `on_chunk_mut` replaced the terminal `RunFinished` with another
  variant, and a streaming consumer never saw the run end. A middleware
  may still rewrite its fields, but if the chunk comes back as another
  variant, the original is restored before `on_chunk` observers and the
  consumer see it.

- `ApprovalMiddleware`, `MaxToolCalls` and `AntiLoop` leaked a run's
  state when the caller dropped the stream mid-run: they cleared their
  per-`RunId` maps only in `on_run_finished` / `on_run_error`, which a
  dropped run never fires. They now clear it in `on_run_dropped` too.
  `MaxToolCalls` and `AntiLoop` switched their internal lock from
  `tokio::sync::Mutex` to `std::sync::Mutex` so the synchronous hook can
  take it; neither holds it across an `.await`.

- A panic on one thread while it held an internal lock no longer makes
  every later use of that lock panic too. `UsageSink`, `ToolActivation`,
  the engine's active-tool snapshot, `SubAgentTool`'s wrap-up budget,
  `OnlineCalibratedTokenizer`, `CachingTokenProvider`,
  `InMemoryHistoryStore` and `ScriptedModel` used `expect` on the lock,
  so a single panic poisoned it and took down unrelated runs sharing the
  handle. They now recover the guard, as `MaxToolCalls`, `AntiLoop` and
  `ApprovalMiddleware` already did; the data they protect is updated in
  one step, so it is still consistent.

- Adding `Usage` values (`+`, `+=`) saturates each counter at `u64::MAX`
  instead of overflowing. Before, an overflow panicked in debug builds
  (inside `UsageSink::report`, with the lock held) and wrapped around
  silently in release builds, leaving a wrong total.

- The Azure OpenAI adapter silently dropped mid-stream error events.
  When the service fails after the response has started, it sends
  `data: {"error":{...}}` in place of a chunk; the adapter parsed it as
  an empty chunk and ended the stream without a finish reason, so the
  run finished as if the model had stopped. The stream now yields
  `AzureOpenAIError::Provider` with the typed kind and the message.

- A `system_prompt` set by a user middleware in `on_chat_request` is no
  longer discarded. The internal system-prompt middleware runs after
  user middlewares and used to overwrite `req.system_prompt` with the
  builder prompt. It now composes: the builder prompt goes first (a
  stable prefix that keeps prompt-cache hits) and the user's prompt
  after it. `Plain + Plain` is joined with a blank line; a
  `SystemPrompt::Blocks` overlay produces `Blocks`, with the builder
  prompt as a leading block and the user's blocks (and their
  `cache_control` breakpoints) kept as they are. When the builder prompt
  renders empty, the user's prompt reaches the model untouched.

- `RunOptions::max_tokens` and `SubAgentConfig::max_tokens` are no
  longer ignored when the conversation was built with
  `ConversationBuilder::max_tokens`: the builder default used to
  overwrite `req.max_tokens` on every request. The cap is now resolved
  before the run with the precedence `RunOptions` > builder > engine
  default; a user middleware rewriting `req.max_tokens` in
  `on_chat_request` still wins. `RunConfig::max_tokens` carries the
  resolved value, so `JsonTracer`'s `run_started` reports the cap
  actually sent.

- The `StreamChunk::HistoryCompacted` that `Conversation` emits for the
  compaction before a run now also passes through
  `ChatMiddleware::on_chunk_mut`, like every other engine chunk.

### Migration

Upgrade checklist, in the order of the snippets below. Most code only
hits the first group.

1. Match run errors through `RunError` (`.kind()` / `.into_kind()`).
2. Middlewares: rewrite the run-level hooks to take context structs and
   the tool hooks to take `&ToolCallInfo`.
3. Approval callbacks take one `ApprovalRequest`.
4. Match `FinishReason::Aborted(AbortReason::…)` instead of a string;
   `max_iterations` is now one of those aborts, not an `Err`.
5. Set `max_iterations` explicitly if you relied on the old default of
   10.
6. Custom models: `impl ProviderError` for the error type; match
   `EngineError::ContextOverflow` for overflows.
7. Activate deferred tools before the model calls them; do not activate
   tools filtered by `capabilities`.
8. Widen stored `Usage` counters to `u64`.
9. Mechanical: `..` in engine-chunk patterns, `call_id` instead of `id`,
   root import paths, `RunId::from`, `ToolContext::detached`, and the
   renames.
10. Adapters: `AnthropicChatModel`, `ServerError`, `from_env` errors,
    Azure `Provider { kind, .. }`, `..` in `Api` / `Provider` patterns.

A failed run returns `RunError`. `?` into `EngineError` still compiles;
direct matches go through `.kind()` or `.into_kind()`:

```rust
// Before (1.0.0-rc.3)
match chat.run("go").await {
    Err(EngineError::Model(e)) => report(e),
    Err(e) => return Err(e),
    Ok(outcome) => use_it(outcome),
}

// After
match chat.run("go").await {
    Err(err) => match err.into_kind() {
        EngineError::Model(e) => report(e),
        e => return Err(e),
    },
    Ok(outcome) => use_it(outcome),
}

// New: keep the steps that completed before the failure
if let Err(err) = chat.run("go").await {
    chat.extend_messages(err.partial_messages().iter().cloned());
}
```

Run-level hooks take a context struct; read the old arguments from its
fields:

```rust
use ailoop::{RunErrorInfo, RunFinishedInfo, RunStartInfo, StepInfo, TurnEndInfo};

// Before (1.0.0-rc.3)
async fn on_run_started(&self, run_id: &RunId, messages: &[Message], config: &RunConfig) -> HookAction {
    self.start(run_id, config.max_iterations);
    HookAction::Continue
}
async fn on_chat_request(&self, run_id: &RunId, step_id: &StepId, req: &mut ChatRequest) { /* .. */ }
async fn on_run_finished(&self, run_id: &RunId, reason: &FinishReason, usage: &Usage, new_messages: &[Message]) {
    self.finish(run_id, usage);
}
async fn on_run_error(&self, run_id: &RunId, err: &(dyn Error + Send + Sync)) {
    self.fail(run_id, err);
}

// After
async fn on_run_started(&self, run: &RunStartInfo<'_>) -> HookAction {
    self.start(run.run_id, run.config.max_iterations);
    HookAction::Continue
}
async fn on_chat_request(&self, step: &StepInfo, req: &mut ChatRequest) {
    // step.run_id, step.step_id
}
async fn on_run_finished(&self, run: &RunFinishedInfo<'_>) {
    self.finish(run.run_id, run.usage);
}
async fn on_run_error(&self, run: &RunErrorInfo<'_>) {
    self.fail(run.run_id, run.error); // also run.usage, run.partial_messages
}
// New hook
async fn on_turn_end(&self, turn: &TurnEndInfo<'_>) -> ContinueDecision {
    // turn.run_id, turn.step_id, turn.reason, turn.new_messages
    ContinueDecision::Stop
}
```

To unit-test a middleware, build the context with its constructor,
e.g. `RunFinishedInfo::new(&run_id, &FinishReason::EndTurn, &usage, &[])`.

Tool hooks take a `&ToolCallInfo`; read the run, step, call id and tool
name from its fields:

```rust
use ailoop::ToolCallInfo;

// Before (1.0.0-rc.3)
async fn on_before_tool_call(
    &self,
    run_id: &RunId,
    step_id: &StepId,
    name: &str,
    args: &Value,
) -> ToolDecision {
    if name == "rm" { self.seen(run_id); }
    ToolDecision::Continue
}
async fn on_after_tool_call(
    &self,
    run_id: &RunId,
    step_id: &StepId,
    name: &str,
    args: &Value,
    result: &ToolResultContent,
) {}

// After
async fn on_before_tool_call(&self, call: &ToolCallInfo, args: &Value) -> ToolDecision {
    if call.name == "rm" { self.seen(&call.run_id); }
    ToolDecision::Continue
}
async fn on_after_tool_call(
    &self,
    call: &ToolCallInfo,
    args: &Value,
    result: &ToolResultContent,
) {}
```

`on_before_tool_call_mut` and `on_after_tool_call_mut` change the same
way. Per-call state that paired `before` and `after` by position should
be keyed by `(call.run_id.clone(), call.call_id.clone())` instead. To
call a hook directly in a test, build the argument with
`ToolCallInfo::new(run_id, step_id, "toolu_1", "rm")`.

Approval callbacks take one `ApprovalRequest`; read the name and
arguments from its fields (the builder method also drops `with_`):

```rust
// Before (1.0.0-rc.3)
.with_approval(|name, args| async move {
    ask_human(&name, &args).await
})

// After
.approval(|req| async move {
    ask_human(&req.name, &req.args).await
})
```

The same applies to `approval_for_tags`, `approval_for_all`,
`ApprovalMiddleware::approve_all` and `ApprovalMiddleware::approve_named`.

`FinishReason::Aborted` carries `AbortReason`: match the variant, or
call `.to_string()` where the old `String` was used.

```rust
// Before (1.0.0-rc.3)
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

`max_iterations` no longer produces an `Err`; check the outcome instead.
Because the partial turns are now persisted, drop any code that
re-appended them by hand.

```rust
// Before (1.0.0-rc.3)
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

To keep the old cap of 10 iterations, set it explicitly:

```rust
// Every run of a conversation (also a sub-agent's child)
let chat = Conversation::builder(model).max_iterations(10).build()?;

// Per run
chat.run_with_options(input, RunOptions::new().max_iterations(10)).await?;

// Engine entry point
let config = RunConfig { max_iterations: 10, ..Default::default() };

// Sub-agent children
SubAgentTool::with_config(name, description, child, SubAgentConfig::new().max_iterations(10));
```

Custom `CompletionModel` error types used with `Conversation` must
implement `ProviderError`. An empty impl keeps the old behavior (no
overflow is ever reported); override `is_context_overflow` to opt in to
recovery. Generic code over `Conversation<M>` gains the bound, and
drops `Send + Sync`, which `CompletionModel` now implies:

```rust
// Before (1.0.0-rc.3)
async fn ask<M: CompletionModel + Send + Sync>(chat: &mut Conversation<M>) { /* ... */ }

// After
impl ailoop::ProviderError for MyError {} // or override is_context_overflow

async fn ask<M>(chat: &mut Conversation<M>)
where
    M: CompletionModel,
    M::Error: ailoop::ProviderError,
{ /* ... */ }
```

Overflow errors surface as a dedicated variant after recovery. To keep
the old behavior (no compaction, no retry), build the conversation with
`.recover_from_context_overflow(false)`:

```rust
// Before (1.0.0-rc.3): an overflow was an opaque model error
match chat.run(input).await {
    Err(EngineError::Model(e)) if looks_like_overflow(&e) => start_new_session(),
    other => handle(other),
}

// After
match chat.run(input).await {
    Err(err) if matches!(err.kind(), EngineError::ContextOverflow(_)) => start_new_session(),
    other => handle(other),
}
```

A model can no longer call a deferred tool it has not activated. If you
relied on that, activate the tool before the model calls it (from a
meta-tool with `ctx.tools().activate(name)`), or start with it active:

```rust
// Before (1.0.0-rc.3): "fetch" was callable by name while hidden
Conversation::builder(model)
    .tool(SearchTools)
    .tool(Fetch)
    .initial_active_tools(["search_tools"])

// After: list it if the model must be able to call it from the start
Conversation::builder(model)
    .tool(SearchTools)
    .tool(Fetch)
    .initial_active_tools(["search_tools", "fetch"])
```

Tools filtered out by `capabilities` cannot be activated any more. To
expose them on demand, register them without a capability filter, defer
them with `initial_active_tools`, and add `approval` if they need a
gate.

Code that stores `Usage` counters in `u32` converts them, or widens its
own type. `OnlineCalibratedTokenizer::observe` takes a `u64`:

```rust
// Before (1.0.0-rc.3)
let input: u32 = usage.input_tokens;
tokenizer.observe(chars, billed_u32);

// After
let input: u64 = usage.input_tokens;
// or, where a u32 is required:
let input = u32::try_from(usage.input_tokens).unwrap_or(u32::MAX);
tokenizer.observe(chars, u64::from(billed_u32));
```

Engine-emitted chunks need `..` in patterns and a constructor outside
`ailoop-core`:

```rust
// Before (1.0.0-rc.3)
if let StreamChunk::StepStarted { run_id, step_id, iteration } = &chunk { /* .. */ }
let chunk = StreamChunk::RunFinished { run_id, reason, usage, new_messages };

// After
if let StreamChunk::StepStarted { run_id, step_id, iteration, .. } = &chunk { /* .. */ }
let chunk = StreamChunk::run_finished(run_id, reason, usage, new_messages);
```

Rename `id` to `call_id` when matching or building tool call blocks and
chunks:

```rust
// Before (1.0.0-rc.3)
if let StreamChunk::ToolCallFinished { id, name, args } = chunk { /* .. */ }
AssistantBlock::ToolCall { id, name, args, .. } => { /* .. */ }

// After
if let StreamChunk::ToolCallFinished { call_id, name, args } = chunk { /* .. */ }
AssistantBlock::ToolCall { call_id, name, args, .. } => { /* .. */ }
```

Import from the crate root (or from `ailoop`) instead of a module path:

```rust
// Before (1.0.0-rc.3)
use ailoop_core::stream::StreamChunk;
use ailoop_tools::errors::ToolRegistryError;

// After
use ailoop::{StreamChunk, ToolRegistryError};
```

Replace the tuple constructor and `.0` access on `RunId` / `StepId`:

```rust
// Before (1.0.0-rc.3)
let run_id = RunId(trace_uuid);
let uuid = run_id.0;

// After
let run_id = RunId::from(trace_uuid);
let uuid = *run_id.as_uuid();
```

Build a `ToolContext` outside the engine with `detached()`:

```rust
// Before (1.0.0-rc.3)
let ctx = ToolContext::new(run_id, step_id, activation);

// After
let ctx = ToolContext::detached();
```

Drop `with_` from the `ConversationBuilder` setters:

```rust
// Before (1.0.0-rc.3)
Conversation::builder(model)
    .with_history(History::builder(100_000))
    .with_capabilities(&[ToolTag::ReadOnly])
    .with_approval_for_tags(&[ToolTag::Destructive], gate)

// After
Conversation::builder(model)
    .history(History::builder(100_000))
    .capabilities(&[ToolTag::ReadOnly])
    .approval_for_tags(&[ToolTag::Destructive], gate)
```

Rename the `Conversation` history accessors:

```rust
// Before (1.0.0-rc.3)
let msgs = chat.history_messages();
chat.history_push(Message::user("seed"));

// After
let msgs = chat.messages();
chat.push_message(Message::user("seed"));
```

Rename the `ToolRegistry` activation methods and
`ApprovalMiddleware::for_named`:

```rust
// Before (1.0.0-rc.3)
registry.activate_tool("search")?;
registry.deactivate_tool("search")?;
ApprovalMiddleware::for_named(["delete_file"], gate)

// After
registry.activate("search")?;
registry.deactivate("search")?;
ApprovalMiddleware::approve_named(["delete_file"], gate)
```

Build named prompt sections with `named`, and rename `CompactionReport`:

```rust
// Before (1.0.0-rc.3)
PromptSection::with_name("Tone", "Be concise.")
let report: Option<CompactionReport> = history.compact_if_needed().await?;

// After
PromptSection::named("Tone", "Be concise.")
let stats: Option<CompactionStats> = history.compact_if_needed().await?;
```

Invert the value when moving to `parallel_tool_use`, and drop the
trailing underscore from `ToolChoice::None_`:

```rust
// Before (1.0.0-rc.3)
builder.disable_parallel_tool_use(true);
req.disable_parallel_tool_use = Some(true);
builder.tool_choice(ToolChoice::None_)

// After
builder.parallel_tool_use(false);
req.parallel_tool_use = Some(false);
builder.tool_choice(ToolChoice::None)
```

Manual `ToolDyn` impls return a borrowed name:

```rust
// Before (1.0.0-rc.3)
fn name(&self) -> String {
    self.name.clone()
}

// After
fn name(&self) -> &str {
    &self.name
}
```

Anthropic adapter: rename the model type and the server-error kind, and
match the typed error from `from_env`:

```rust
// Before (1.0.0-rc.3)
let model: AnthropicModel = client.model("claude-sonnet-4-6");
AnthropicApiErrorKind::Api => { /* ... */ }
match AnthropicClient::from_env() {
    Err(std::env::VarError::NotPresent) => { /* ... */ }
    // ...
}

// After
let model: AnthropicChatModel = client.model("claude-sonnet-4-6");
AnthropicApiErrorKind::ServerError => { /* ... */ }
match AnthropicClient::from_env() {
    Err(AnthropicError::Config(msg)) => { /* ... */ }
    // ...
}
```

Azure OpenAI adapter: match the typed kind on mid-stream errors. In both
adapters, add `..` when destructuring `Api` / `Provider`:

```rust
// Before (1.0.0-rc.3)
AzureOpenAIError::Provider { error_type, message } => { /* ... */ }
AnthropicError::Api { status, kind, message, retry_after } => { /* ... */ }

// After
AzureOpenAIError::Provider { kind, message, .. } => { /* ... */ }
AnthropicError::Api { status, kind, message, retry_after, .. } => { /* ... */ }
```

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
