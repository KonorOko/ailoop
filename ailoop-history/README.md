# ailoop-history

Conversation history management and compaction for
[`ailoop`](https://crates.io/crates/ailoop).

Provides:

- `History` — keeps the conversation message vector and the pin mask
  used by compaction strategies.
- `Compactor` strategies — drop / summarize old turns while
  preserving tool-call / tool-result pairing. Emits
  `StreamChunk::HistoryCompacted` with strategy name and counts.
- `HistoryStore` — async trait for persisting and restoring
  conversation state. Ships with `InMemoryHistoryStore` and
  `JsonFileHistoryStore`.
- `Tokenizer` / `CharTokenizer` — pluggable token-estimation
  abstraction used by budgeting strategies.
- `ConversationSnapshot` — versioned, JSON-serializable
  representation of an active `Conversation`, ready for restart.

Most application code reaches the history surface through the
`ailoop` façade re-exports rather than depending on this crate
directly.

> Renamed from `ailoop-context` in 1.0.0-rc.3. The published
> `ailoop-context` crate is yanked; install `ailoop-history` instead.

See the [workspace README](https://github.com/KonorOko/ailoop) for
the big picture.

## License

Licensed under either of Apache-2.0 or MIT, at your option.
