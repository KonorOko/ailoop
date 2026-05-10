# ailoop-derive

Procedural macros for [`ailoop`](https://crates.io/crates/ailoop):

- `#[ailoop_tool]` — annotate an `async fn` to register it as an
  agent tool. Generates parameter parsing, JSON Schema, and the
  `Tool` impl.
- `#[derive(ToolJsonType)]` — derive a JSON Schema fragment for a
  struct used as a tool parameter type.

These macros are re-exported from the `ailoop` façade, so most
application code does not need a direct dependency on
`ailoop-derive`.

See the [workspace README](https://github.com/KonorOko/ailoop) for
the big picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
