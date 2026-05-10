# ailoop-tools

Tool registry and trait surface for
[`ailoop`](https://crates.io/crates/ailoop).

Defines:

- `Tool` trait — type-safe input/output backed by `serde`. Implement
  it directly or use `#[ailoop_tool]` from `ailoop-derive` for the
  ergonomic path.
- `ToolRegistry` — collects `Tool` impls, dispatches the
  model-requested invocations, returns typed errors.
- `ToolTag` — capability declarations used by
  `Conversation::with_capabilities` and the `ApprovalMiddleware`.
- `ToolJsonType` — derive-friendly JSON Schema fragment generator
  for tool parameter types.

Most application code reaches the tool surface through the `ailoop`
façade re-exports rather than depending on this crate directly.

See the [workspace README](https://github.com/KonorOko/ailoop) for
the big picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
