//! Conversation history management for ailoop.
//!
//! Owns the message vector that drives every [`ChatRequest`] and
//! enforces the token-budget contract: [`History`] tracks the
//! running history, applies a [`CompactionStrategy`] when the wired
//! [`Tokenizer`] reports the budget is exceeded, and persists the pin
//! mask that keeps system-style anchors and live `ToolCall ↔ ToolResult`
//! pairs intact across compactions.
//!
//! Most application code does not construct a `History` directly
//! — the [`ailoop`](https://docs.rs/ailoop) façade wires one inside
//! [`Conversation`](https://docs.rs/ailoop) with a sensible default
//! budget. Reach for this crate when you need a custom strategy
//! ([`SummarizeStrategy`] vs [`TruncateStrategy`]), a non-default
//! tokenizer (e.g. `ailoop_anthropic::OnlineCalibratedTokenizer`),
//! direct access to `compact_if_needed` from
//! [`advanced::run_chat`](https://docs.rs/ailoop), or your own
//! [`HistoryStore`] backend for persistence.
//!
//! ## Mini-index
//!
//! - [`History`] + [`HistoryBuilder`] — the history
//!   container and its configuration entry point.
//! - [`CompactionStrategy`] — trait implemented by reduction
//!   algorithms; ships with [`TruncateStrategy`] and
//!   [`SummarizeStrategy`].
//! - [`CompactionOutput`], [`CompactionReport`], [`CompactionError`],
//!   [`FromMessagesError`] — the surrounding vocabulary.
//! - [`HistoryStore`] — async persistence trait. [`InMemoryHistoryStore`]
//!   and [`JsonFileHistoryStore`] cover tests and single-file durable
//!   storage; implement the trait for Redis, Postgres, etc.
//! - [`ConversationSnapshot`] — versioned wire format for
//!   [`HistoryStore`] payloads. Validated at deserialize time so a
//!   malformed file fails loudly rather than panicking later.
//! - [`Tokenizer`] / [`CharTokenizer`] — re-exported from
//!   [`ailoop_core`] so [`HistoryBuilder::tokenizer`] callers
//!   can stay on this crate's import surface.
//!
//! [`ChatRequest`]: ailoop_core::ChatRequest

#![deny(missing_docs)]

pub mod compaction;
pub mod errors;
pub mod history;
pub mod history_store;
pub mod snapshot;

pub use compaction::{
    CompactionOutput, CompactionStrategy, DEFAULT_SUMMARIZER_PROMPT, SummarizeStrategy,
    TruncateStrategy,
};
pub use history::{CompactionReport, History, HistoryBuilder};
pub use errors::{CompactionError, FromMessagesError};
pub use history_store::{
    HistoryStore, InMemoryHistoryStore, JsonFileHistoryStore, JsonFileHistoryStoreError,
};
pub use snapshot::ConversationSnapshot;

// `Tokenizer` lives in `ailoop-core` (its canonical home alongside
// `Message`). Re-export at the root so callers wiring
// `HistoryBuilder::tokenizer` can stay on this crate's import
// surface without dragging `ailoop-core` into their dep tree.
// `CharTokenizer` is the dev/test fallback.
pub use ailoop_core::{CharTokenizer, Tokenizer};
