# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ToolCallInfo` (re-exported from `ailoop`): the identity of one tool
  call, passed to every tool hook of `ChatMiddleware`. Public fields
  `run_id`, `step_id`, `call_id` (the provider-assigned id, the same as
  `id` on `ToolCallFinished` and `call_id` on `ToolResult`) and `name`.
  It is `#[non_exhaustive]` so more per-call data can be added without
  breaking middlewares; `ToolCallInfo::new` builds one to unit-test a
  hook.

- `ToolContext::call_id()`: the id of the call being dispatched, so a
  tool can correlate its own logs or side effects with the
  `ToolResult` chunk and the middleware hooks. `ToolContext::detached`
  mints a synthetic one.

- `ApprovalRequest::call_id` and `ApprovalRequest::with_call_id`.
  `ApprovalMiddleware` fills it, so an approval UI or verifier can
  tell apart two identical calls in one step and match its decision
  to the `ToolResult` chunk. `ApprovalRequest::new` leaves it empty.

- `JsonTracer` writes `call_id` on `before_tool_call` and
  `after_tool_call` lines, and `TracingMiddleware` adds a `call_id`
  field to its "tool call starting" / "tool call finished" events.
  Before, those events could not be matched to `tool_call_finished` /
  `tool_result`, or to each other when a step called the same tool
  twice. Additive within `schema: 1`.

- `RunError::usage()`: what a failed run spent before the error, counted
  like `RunFinished.usage` (finished provider turns plus usage reported
  by tools, sub-agents included). Before, a run ending in `Err` lost its
  usage entirely, and the delegated part could not be rebuilt from
  `TurnFinished` chunks, so a token or cost budget was blind exactly on
  the runs that fail. The turn that failed is not counted: a response
  cut off by an error never reports its usage, although the provider
  may bill it. The usage can include the failed step's turn and its
  tool reports (for example when a tool registry error follows a
  finished turn) even though that step is left out of
  `partial_messages`. Errors raised before the run starts (history
  compaction in `Conversation::stream_with_options`) carry zero usage.
  `into_parts` is unchanged.

- `Usage` derives `PartialEq` and `Eq`.

- `ToolRegistry::retain_by_tags(tags)`: unregister every tool whose
  tags do not overlap with `tags`. Unlike `deactivate_by_tags`, the
  removed tools can no longer be dispatched or activated at runtime.
  `ConversationBuilder::with_capabilities` is now built on it.

- `ApprovalRequest` (re-exported from `ailoop`): what an approval
  callback receives for one gated call. Public fields `run_id`,
  `step_id`, `tool_name`, `args`, `tags` (the tool's declared tags) and
  `messages` (the context sent to the model on the step that produced
  the call, after every middleware's `on_chat_request`). The type is
  `#[non_exhaustive]` so more context can be added without breaking
  callbacks; `ApprovalRequest::new` plus `with_tags` / `with_messages`
  build one outside the crate, e.g. to unit-test a verifier. Its
  rustdoc describes a model-based risky-action verifier: tags pick what
  gets reviewed, the verifier allows, denies or escalates to a human,
  it fails closed on error or timeout, and it reads intent from the
  user's text only, never from tool results, which can carry injected
  instructions.

- `ToolContext::report_usage(Usage)` and `UsageSink` (re-exported from
  `ailoop`): a tool that spends tokens outside the engine's own
  provider turns (a sub-agent, a tool that calls an LLM directly)
  reports them here and the engine adds them to the run's usage
  total. Reports count as soon as they are made, so a tool dropped by
  a timeout or cancellation still contributes what it already spent.
  `ToolContext::usage_sink()` exposes the sink so a tool can hand it to
  a nested run or spawned task. `ToolContext::with_usage_sink(sink)`
  replaces it, which is useful in tests to read back what a tool
  reported. `ToolContext::new` is unchanged.

- `RunError<E>` (re-exported from `ailoop`): the error of a failed run.
  It wraps the `EngineError` cause (`kind()`, `into_kind()`) and the
  messages of the steps the run completed before failing
  (`partial_messages()`, `into_parts()`). Until now those steps were
  lost when a run failed: the history is rolled back on `Err`, so the
  record of tools that already ran (writes, API calls) disappeared and
  the next turn could repeat them. The partial list is exactly the
  `new_messages_so_far` of the last `StepFinished`: every `tool_use` in
  it has its `tool_result`, and the step that failed is left out,
  including any text it streamed. A model error in the middle of a
  response arrives before that step's tools run, so no executed tool
  is missing. The rollback guarantee is unchanged; the list is a copy
  the caller can keep or drop.
- `Conversation::history_extend(messages)`: append several messages
  without running, like `history_push`. Use it to keep the steps of a
  failed run: `chat.history_extend(err.partial_messages().iter().cloned())`.
- `StreamChunk::ToolCallMalformed { id, name, raw, error }`: closes a
  streamed tool call whose argument text is not a JSON object, in place
  of `ToolCallFinished`. `StreamChunk::tool_call_from_raw_args(id, name,
  raw)` builds the right closing chunk from the accumulated text and is
  what both built-in adapters now use; third-party adapters should call
  it too. Empty or whitespace-only text is still a valid `{}` (tools
  without parameters). `JsonTracer` logs the new chunk as
  `tool_call_malformed` (with `raw` only in verbose mode) and
  `TracingMiddleware` as a `warn` event. Consumers that match on
  `StreamChunk` already need a wildcard arm (`#[non_exhaustive]`).

- `ToolRegistry::all_tools()`: iterate over every registered tool,
  active and inactive, in registration order. It is the catalog a
  handler can reach with `ToolActivation::activate`, so policies that
  must cover every tool a run could dispatch (like approval gating)
  should be computed over it rather than over `active_tools()`.
- `ChatMiddleware::on_turn_end` and `ContinueDecision` (re-exported
  from `ailoop`): native support for the "completion gate" pattern.
  When the model ends a turn with no tool calls to run, the engine
  asks each middleware, in registration order, before it finishes the
  run. Returning `ContinueDecision::Continue { blocks }` (or
  `ContinueDecision::continue_with(text)`) adds a user message and
  runs another iteration of the same run. The first `Continue` wins,
  and the remaining middlewares are not asked for that turn. A
  `Continue` with no blocks counts as `Stop` (and the next middleware
  is asked), since continuing without a new user message would send a
  request that ends on the assistant's own turn. Until now
  this took an outer loop around `stream_with_options`, which reset
  `iteration` to 0, split `usage` across runs and emitted one
  `RunFinished` per attempt. With the hook, iterations keep counting,
  usage accumulates and a single `RunFinished` closes the run. The
  injected message is part of the history and of `new_messages`, and
  shows up in the `StepFinished` of the step that asked to continue.
  The hook fires for `EndTurn`, `MaxTokens`, `StopSequence` and
  `Other`, and receives the reason so a gate can filter. It never
  fires for `ToolUse` or `Aborted`. Every continuation counts against
  `max_iterations`, so a gate that never passes ends the run with
  `AbortReason::MaxIterations`. If the turn also completed tool calls
  (possible with `MaxTokens`), the injected blocks join the tool
  results in one user message instead of following them as a second
  one. A turn that produced no assistant content at all (an empty
  `EndTurn`) leaves nothing between the previous user message and the
  injected one; providers merge consecutive user turns. A gate in a
  `SubAgentTool` child can override the wrap-up turn of
  `SubAgentConfig::wrap_up`, so it should let `EndTurn` through once
  the budget is nearly spent. The default returns `ContinueDecision::Stop`, so
  existing middlewares are unaffected. The rustdoc of
  `on_chat_request` now also explains how to track the current
  iteration: record `StepStarted { iteration }` from `on_chunk` and
  `max_iterations` from `on_run_started`, keyed by `run_id`.
- `SubAgentConfig::wrap_up(WrapUp)`: an opt-in graceful cutoff for
  `SubAgentTool`. Today, when a child run hits its `timeout` or
  `max_iterations`, the parent gets `"sub-agent aborted (…)"` with
  `is_error: true`, and whatever the child found after its last text
  block is lost. With `wrap_up`, the child's last turn before the hard
  cutoff is forced to be a summary: it runs on the last allowed
  iteration, or on the first request after `timeout *
  WrapUp::time_fraction` (default `0.8`). That request sets
  `tool_choice` to `ToolChoice::None_` but keeps the tool definitions,
  so the prompt cache survives. It also appends `WrapUp::instruction`
  (default `DEFAULT_WRAP_UP_INSTRUCTION`) to the last user message.
  The instruction only goes into that request, never into the child's
  history. If the summary finishes, the parent receives
  `"[partial: sub-agent reached its time budget]\n<summary>"` (or
  `iteration budget`) with `is_error: false`. The signal lives in the
  text because the parent model only sees the tool result. The hard
  timeout is still absolute: if it fires before the summary is done,
  the result is the usual abort with `is_error: true`. Without
  `wrap_up`, behavior is unchanged. `WrapUp` and
  `DEFAULT_WRAP_UP_INSTRUCTION` are re-exported from `ailoop`.
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
- In-run context management for `Conversation`. Before, the history
  was compacted once, before the run, and each tool result grew the
  context until the provider rejected it, which ended the run with
  `EngineError::Model`. Two builder switches cover this:
  - `ConversationBuilder::compact_between_iterations(bool)` (default
    `false`): checks the history budget before every model call after
    the first, not only at run start. When it compacts, the stream
    carries a `StreamChunk::HistoryCompacted` with the run's `RunId`
    between the previous `StepFinished` and the next `StepStarted`.
    The built-in strategies cut only at a user message that is not a
    tool result, which inside a run means the run's own kickoff. So
    mid-run compaction reclaims earlier turns, never splits the run's
    tool_use / tool_result pairs, and cannot shrink the run itself.
    If a compaction leaves the history over budget, or does not reduce
    it, the check is skipped for the rest of the run instead of being
    repeated every iteration. It is off by default because each
    compaction has a cost (an extra model call with
    `SummarizeStrategy`).
  - `ConversationBuilder::recover_from_context_overflow(bool)` (default
    `true`): when opening the model stream fails with an error whose
    `ProviderError::is_context_overflow()` is `true`, the engine forces
    a compaction (`History::force_compact`), emits `HistoryCompacted`,
    and reissues the request once. `on_chat_request` runs again for
    the same step.
- `EngineError::ContextOverflow(E)`: returned by `Conversation` when an
  overflow could not be recovered. That is, the retry overflowed too,
  or the forced compaction had nothing to drop or did not reduce the
  estimated tokens. The typical cause is a single turn (for example a
  huge tool result) that is larger than the window. It carries the
  provider's last error. The history is rolled back to its state when
  the run started, so no half-finished tool turn is persisted. This is
  the same rollback that already applied to any run ending in `Err` or
  dropped mid-stream; it now also undoes compactions done during the
  run.
- `ProviderError` is implemented for `std::convert::Infallible`, so
  models with `type Error = Infallible` satisfy the new bound without
  boilerplate.
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

- `RunFinished.usage`, `RunOutcome.usage` and the `usage` passed to
  `ChatMiddleware::on_run_finished` now mean **everything the run
  spent**: its own provider turns plus usage reported by tools.
  `SubAgentTool` reports its child run's usage, and nested sub-agents
  roll up recursively. Before, the child's tokens were dropped and the
  parent's total undercounted the real cost, so a token or cost budget
  could not see sub-agents. The spend is forwarded while it happens, so it
  also counts when the child aborts (`SubAgentConfig::timeout`,
  `max_iterations`, wrap-up) and when the parent aborts while the child
  is still running. `TurnFinished.usage` is unchanged: it covers only
  the run's own model. Sum it in a middleware to split own vs.
  delegated spend. If you were adding a sub-agent's usage to the
  parent's total yourself, stop, or it will be counted twice.

- Docstrings for `StreamChunk::TurnFinished` and `Usage` now make
  explicit that the per-turn variant is visible to `ChatMiddleware`
  implementations only — the engine accumulates its `usage` into the
  run total and drops the chunk before the public stream, so stream
  consumers see aggregated `Usage` on `RunFinished.usage`, never the
  per-turn variant. Middleware path points to `on_chunk` for cases
  that need per-turn data (context-size indicator from the final
  turn's `input_tokens`, per-turn latency or service-tier attribution,
  online tokenizer calibration).

- Documented contract: when the model requests several tools in one
  turn, the order **between** those calls is not guaranteed. The engine
  still runs them one at a time in the model's order, and nothing
  changes in this release. The contract is written down now so a later
  1.x release can run a step's tools concurrently without it being a
  breaking change. What stays guaranteed: the hook order within one
  call (`on_before_tool_call_mut` → `on_before_tool_call` → tool →
  `on_after_tool_call_mut` → `on_after_tool_call` → `ToolResult`
  chunk), tool results in history in the order of the model's calls,
  and every call of a step finishing before `on_turn_end` or the next
  `on_chat_request`. Middlewares with state should key it by `RunId`
  (and `StepId`) behind a lock, and should not assume that an
  `on_after_tool_call` belongs to the latest `on_before_tool_call`. See
  "Tool calls within a step" on `ChatMiddleware`.

### Changed (BREAKING)

- `RunConfig::default().max_iterations` is now 25 (was 10). One
  iteration is one model turn plus the tool calls it triggers. Every
  `ContinueDecision::Continue` from `on_turn_end` also counts, and a
  sub-agent's wrap-up uses the last iteration. So 10 ran out on
  ordinary agent tasks (read, search, edit, verify). The run then
  ended with `Ok` / `FinishReason::Aborted(AbortReason::MaxIterations(10))`,
  which is easy to miss. The new default applies to every run that
  does not set a cap, including `SubAgentTool` children with no
  `SubAgentConfig::max_iterations`. It is a safety brake, not a
  budget. Without a token or cost cap, the worst-case run of a caller
  who relies on the default is now 2.5× longer. In production, set
  `max_iterations` explicitly, together with `timeout` and
  `MaxToolCalls`.

- The four tool hooks of `ChatMiddleware` take a `&ToolCallInfo` in
  place of `run_id, step_id, name`: `on_before_tool_call(call, args)`,
  `on_before_tool_call_mut(call, args)`,
  `on_after_tool_call(call, args, result)` and
  `on_after_tool_call_mut(call, args, result)`. The hooks never saw the
  provider's call id, so a middleware could not pair an
  `on_after_tool_call` with its `on_before_tool_call`: two calls to the
  same tool with the same arguments in one step were
  indistinguishable. The order between the calls of a step is not
  guaranteed (they may run concurrently in a later release), so
  per-call state has to be keyed by `(run_id, call_id)`, and the
  "Tool calls within a step" section of `ChatMiddleware` now says so.
  A struct rather than one more `&str` parameter lets later per-call
  data be added without breaking the trait again after 1.0. The
  engine's behaviour is unchanged.
- `ToolContext::new` takes the call id after `step_id`:
  `ToolContext::new(run_id, step_id, call_id, activation, cancellation)`.
  Only the engine builds contexts for real dispatches; an optional
  setter would have left an empty id possible there.

- The engine only runs tools in the run's active set. A call to a
  registered but inactive tool (deferred with `initial_active_tools`
  and not yet activated) no longer runs it: the model gets the same
  error result as for an unknown name (`Tool '<name>' not found.
  Available tools: [...]`, listing only active tools) and the run goes
  on. The active set is read when the call is dispatched, so a tool
  activated by an earlier call of the same step can run. Calls
  rejected this way, and calls to unknown names, no longer fire the
  tool hooks (`on_before_tool_call[_mut]`, `on_after_tool_call[_mut]`)
  or the approval callback; only their `ToolResult` chunk goes out,
  the same as for malformed arguments. `MaxToolCalls` and `AntiLoop`
  therefore do not count them; `max_iterations` still bounds a model
  that keeps calling a hidden tool.
- `ConversationBuilder::with_capabilities` removes the filtered tools
  from the conversation instead of deactivating them. They no longer
  show up in `ToolActivation::list_all` / `list_inactive`,
  `ToolActivation::activate` returns `NotFound` for them, and
  `initial_active_tools` ignores their names. Until now they stayed in
  the catalog, so the "default-deny" promise did not hold (see Fixed).

- Approval callbacks take an `ApprovalRequest` instead of
  `(String, Value)`: `ConversationBuilder::with_approval`,
  `with_approval_for_tags`, `with_approval_for_all`,
  `ApprovalMiddleware::approve_all` and `ApprovalMiddleware::for_named`.
  A callback that only saw the tool name and arguments could not tell
  a reasonable call from a dangerous one: `rm -rf build/` is fine after
  "clean the build" and alarming after "summarize this file". The
  request now carries the user's messages, the tool's tags and the run
  and step ids, and freezing the two-argument form for 1.0 would have
  made adding them later a breaking change. To fill `messages`,
  `ApprovalMiddleware` records each step's request in
  `on_chat_request`, keyed by run so one instance can be shared across
  concurrent runs, and drops it in `on_run_finished` / `on_run_error`.
  The messages are copied once per step into an `Arc<[Message]>` shared
  by every gated call of that step, and only when a gate is installed.
  `tags` is filled for gates installed through the builder, which sees
  the whole tool catalog; `approve_all` and `for_named` leave it
  empty. The call id is not included yet.

- `Conversation::run`, `run_with_options`, `stream`,
  `stream_with_options`, the `RunStream` error item and
  `advanced::run_chat` (both its result and its stream items) now fail
  with `RunError<M::Error>` instead of `EngineError<M::Error>`, so the
  steps completed before the failure reach the caller (see `RunError`
  under Added). `From<RunError<E>> for EngineError<E>` keeps `?`
  working in functions that return `EngineError`; only code that
  matches the error directly needs `.kind()` or `.into_kind()`.
- `ChatMiddleware::on_run_error` gained a `partial_messages: &[Message]`
  parameter with the same list, mirroring the `new_messages` of
  `on_run_finished`, so a persistence middleware can record what the
  run did before failing. `JsonTracer` adds `partial_messages` (a
  count) to its `run_error` event and `TracingMiddleware` adds it as a
  field.
- `ChatMiddleware::on_run_error` also gained a `usage: &Usage`
  parameter, between `err` and `partial_messages` (the same order as
  `on_run_finished`), carrying the same value as `RunError::usage()`.
  It was the only place a middleware could see delegated usage (tool
  and sub-agent reports) on a failed run; adding it after 1.0 would
  break every implementor again. `JsonTracer` adds `usage` to its
  `run_error` event and `TracingMiddleware` adds `input_tokens` /
  `output_tokens` fields. The hook still fires only for runs that
  started: a pre-run compaction error reaches the caller without it.
- `Conversation`'s methods and `SubAgentTool`'s `ToolDyn` impl now
  require `M::Error: ProviderError`, so the engine can tell a
  context-window overflow from other model errors. The built-in
  adapters (`AnthropicError`, `AzureOpenAIError`), `ScriptedError` and
  `Infallible` already implement it. A custom model's error type needs
  an `impl ProviderError for MyError {}`, whose defaults report no
  overflow. Generic code over `M: CompletionModel` that calls
  `Conversation` needs the extra bound. `advanced::run_chat` is
  unchanged: it has no history to compact and gains no bound.
- **Behavior change:** a context-window overflow no longer ends a
  `Conversation` run immediately as `Err(EngineError::Model(_))`.
  With the default `recover_from_context_overflow(true)`, the engine
  compacts and retries once, and returns
  `Err(EngineError::ContextOverflow(_))` if that does not help. Code
  that matched `EngineError::Model(e)` with `e.is_context_overflow()`
  should match `ContextOverflow`, or opt out with
  `recover_from_context_overflow(false)`.
- `ChatMiddleware::on_chat_request` can fire twice for the same
  `step_id` when an overflow is recovered. The retried request is
  rebuilt from the compacted history.
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

- A run that aborted part-way through a step left tool calls without a
  `tool_result`. When a tool's `ToolDecision::Terminate`, a timeout, a
  cancellation or an abort inside a tool hook stopped the run, only the
  calls that had already finished were answered. The call that stopped
  the run and the calls after it stayed as `tool_use` blocks with no
  result in `RunFinished.new_messages`. The same happened to calls the
  model had finished streaming when the run aborted mid-turn.
  `Conversation` kept that history, so the next request was rejected
  by the provider (HTTP 400). Now every call of the step is answered,
  in the model's order. Calls that completed keep their result. The
  others get an `is_error` result reading
  `"Tool not run: the run was aborted (<reason>)"`, where `<reason>` is
  the `AbortReason` as text (for `Terminate`, its `reason`). A call
  with malformed arguments keeps its usual error. Each of these
  results is also emitted as a `ToolResult` chunk before `RunFinished`,
  so every `ToolCallFinished` gets its `ToolResult`. No tool hook fires
  for them, because the tool never ran.

- **Security:** tools hidden from the model could still run. The
  engine looked the called name up in the whole catalog, not in the
  active set, so:
  - a model that named a deferred tool directly, without going through
    the `search_tools` / activation step, ran it;
  - a tool filtered out by `with_capabilities` ran if the model called
    it by name or a handler activated it with `ctx.tools().activate(...)`;
  - `initial_active_tools` could list a filtered tool and make it
    active from the first turn, contrary to its documentation.

  With no approval gate installed nothing stood in the way. The
  engine now rejects calls to inactive tools in-band, and
  `with_capabilities` removes the filtered tools from the catalog, so
  none of these paths can run them.

- **Security:** the approval gate from
  `ConversationBuilder::with_approval` / `with_approval_for_tags` no
  longer lets deferred tools run unapproved. The set of gated tool
  names was resolved at `build()` time from the *initial active set*
  only, so a `Destructive` / `WritesFiles` tool that started inactive
  (via `initial_active_tools`, or filtered out by `with_capabilities`)
  and was later activated by a handler with `ctx.tools().activate(...)`
  (the `search_tools` pattern) executed without ever reaching the
  approval callback — a `Skip` / `Terminate` policy for destructive
  actions was silently bypassed. The gated set is now resolved over
  the whole registered catalog, so the callback fires for those tools
  exactly as for tools active from the start. `with_approval_for_all`
  and `ApprovalMiddleware::for_named` were not affected. **Behavior
  change:** tools filtered out by `with_capabilities` now also go
  through the callback if they get activated; previously the docs
  stated they never triggered it.

- Tool calls with malformed arguments no longer run with `{}`. The
  Anthropic and Azure OpenAI adapters parsed the accumulated argument
  JSON with a silent fallback to `{}`, so a call truncated by
  `max_tokens` (or invalid JSON from Anthropic's fine-grained tool
  streaming) executed the tool with no arguments. For a tool with side
  effects that is unsafe, and the model got a confusing validation
  error about fields it did send. Now the adapters emit
  `StreamChunk::ToolCallMalformed` and the engine never runs the tool:
  it records the call in history with an empty-object input (providers
  require an object there) and answers it with an error `tool_result`:
  `Invalid JSON arguments for tool '<name>': <parser error>. The tool
  was not run; call it again with complete, valid JSON.` followed by
  `{"INVALID_JSON": "<raw>"}` (the wrapper Anthropic recommends; `raw`
  capped at 1 KiB). No tool hook fires for it (`on_before_tool_call*`,
  `on_after_tool_call*`), so approval callbacks, `MaxToolCalls` and
  `AntiLoop` do not see it and `Sanitize::on_tool_result` does not
  rewrite it; the synthesized `ToolResult` still goes through
  `on_chunk`. The continuation rule is unchanged: if the turn ended in
  `ToolUse` the model sees the error and can retry in the same run; if
  it ended in `MaxTokens` the run ends as before, with the call and its
  error result paired in history (an `on_turn_end` middleware can
  continue it).
- The `StreamChunk::HistoryCompacted` that `Conversation` emits for
  the compaction before a run now also passes through
  `ChatMiddleware::on_chunk_mut`, like every other engine chunk. It
  used to reach only `on_chunk`.
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

To keep the old cap of 10 iterations, set it explicitly:

```rust
// Per run
chat.run_with_options(input, RunOptions::new().max_iterations(10)).await?;

// Engine entry point
let config = RunConfig { max_iterations: 10, ..Default::default() };

// Sub-agent children
SubAgentTool::with_config(name, description, child, SubAgentConfig::new().max_iterations(10));
```

Tool hooks take a `&ToolCallInfo`; read the run, step, call id and
tool name from its fields:

```rust
use ailoop::ToolCallInfo;

// Before
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
way. Per-call state that paired `before` and `after` by position
should be keyed by `(call.run_id.clone(), call.call_id.clone())`
instead. To call a hook directly in a test, build the argument with
`ToolCallInfo::new(run_id, step_id, "toolu_1", "rm")`.

`ToolContext::new` takes the call id:

```rust
// Before
ToolContext::new(run_id, step_id, activation, cancellation)
// After
ToolContext::new(run_id, step_id, "toolu_1", activation, cancellation)
```

A model can no longer call a deferred tool it has not activated. If
you relied on that, activate the tool before the model calls it (from
a meta-tool with `ctx.tools().activate(name)`), or start with it
active:

```rust
// Before: "fetch" was callable by name while hidden
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

Tools filtered out by `with_capabilities` cannot be activated any
more. To expose them on demand, register them without a capability
filter and defer them with `initial_active_tools` instead; add
`with_approval` if they need a gate.

Approval callbacks take one `ApprovalRequest`; read the name and
arguments from its fields:

```rust
// Before
.with_approval(|name, args| async move {
    ask_human(&name, &args).await
})

// After
.with_approval(|req| async move {
    ask_human(&req.tool_name, &req.args).await
})
```

The same applies to `with_approval_for_tags`, `with_approval_for_all`,
`ApprovalMiddleware::approve_all` and `ApprovalMiddleware::for_named`.

A failed run returns `RunError`. `?` into `EngineError` still compiles;
direct matches go through `.kind()`:

```rust
// Before
match chat.run("go").await {
    Err(EngineError::ContextOverflow(e)) => shrink_input(e),
    Err(e) => return Err(e),
    Ok(outcome) => use_it(outcome),
}

// After
match chat.run("go").await {
    Err(err) => match err.into_kind() {
        EngineError::ContextOverflow(e) => shrink_input(e),
        e => return Err(e),
    },
    Ok(outcome) => use_it(outcome),
}

// New: keep the steps that completed before the failure
if let Err(err) = chat.run("go").await {
    chat.history_extend(err.partial_messages().iter().cloned());
}
```

`on_run_error` overrides take the run's usage and the partial messages:

```rust
// Before
async fn on_run_error(&self, run_id: &RunId, err: &(dyn std::error::Error + Send + Sync)) {}

// After
async fn on_run_error(
    &self,
    run_id: &RunId,
    err: &(dyn std::error::Error + Send + Sync),
    usage: &Usage,
    partial_messages: &[Message],
) {}
```

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

Custom `CompletionModel` error types used with `Conversation` must
implement `ProviderError`. An empty impl keeps the old behavior (no
overflow is ever reported); override `is_context_overflow` to opt in
to recovery.

```rust
// Before
impl CompletionModel for MyModel {
    type Error = MyError;
    // ...
}

// After
impl ailoop::ProviderError for MyError {} // or override is_context_overflow
```

Generic code over `Conversation<M>` gains the bound:

```rust
// Before
async fn ask<M: CompletionModel + Send + Sync>(chat: &mut Conversation<M>) { /* ... */ }

// After
async fn ask<M>(chat: &mut Conversation<M>)
where
    M: CompletionModel + Send + Sync,
    M::Error: ailoop::ProviderError,
{ /* ... */ }
```

Overflow errors surface as a dedicated variant after recovery:

```rust
// Before
match chat.run(input).await {
    Err(EngineError::Model(e)) if e.is_context_overflow() => start_new_session(),
    other => handle(other),
}

// After
match chat.run(input).await {
    Err(EngineError::ContextOverflow(_)) => start_new_session(),
    other => handle(other),
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
