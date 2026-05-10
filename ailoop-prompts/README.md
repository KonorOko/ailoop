# ailoop-prompts

Composable system-prompt assembly for
[`ailoop`](https://crates.io/crates/ailoop).

Provides:

- `PromptBuilder` — assembles a `SystemPrompt` from a sequence of
  fragments (static strings, async-loaded files, dynamic builders).
- `SystemPromptMiddleware` integration so prompt fragments can be
  attached to a `Conversation` and re-evaluated per turn when
  appropriate.

Most application code reaches the prompt surface through the
`ailoop` façade re-exports rather than depending on this crate
directly.

See the [workspace README](https://github.com/KonorOko/ailoop) for
the big picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
