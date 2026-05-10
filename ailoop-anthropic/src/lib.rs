//! Anthropic Messages API adapter for ailoop.
//!
//! Implements [`CompletionClient`] / [`CompletionModel`] against
//! Anthropic's `POST /v1/messages` streaming endpoint (Server-Sent
//! Events). Plug an [`AnthropicModel`] into a
//! [`Conversation`](https://docs.rs/ailoop) by chaining
//! `Conversation::builder(client.model("claude-…"))`.
//!
//! ## Provider features honoured
//!
//! - Configurable `anthropic-version` and `anthropic-beta` headers via
//!   [`AnthropicClient::version`] / [`AnthropicClient::beta`].
//! - Per-block `cache_control` (system + messages + tools) emitted on
//!   the wire whenever the corresponding [`CacheControl`] is attached
//!   in [`ailoop_core`]; the adapter surfaces TTL-broken-down
//!   `cache_creation_5m_tokens` / `cache_creation_1h_tokens` plus
//!   per-turn `service_tier` on [`StreamChunk::TurnFinished`].
//! - Extended thinking blocks (`thinking` / `redacted_thinking`)
//!   round-tripped end-to-end via [`AssistantBlock::Reasoning`] /
//!   [`AssistantBlock::RedactedReasoning`].
//! - Typed HTTP errors via [`AnthropicError`] +
//!   [`AnthropicApiErrorKind`], wired into [`Retryable`] for
//!   [`RetryingModel`] backoff.
//!
//! ## Mini-index
//!
//! - [`AnthropicClient`] — connection + auth + header configuration.
//! - [`AnthropicModel`] — the [`CompletionModel`] implementation
//!   built by [`AnthropicClient::model`] /
//!   [`CompletionClient::completion_model`](ailoop_core::CompletionClient::completion_model).
//! - [`AnthropicError`] / [`AnthropicApiErrorKind`] — typed failure
//!   surface.
//! - [`OnlineCalibratedTokenizer`] — recommended production
//!   [`Tokenizer`](ailoop_core::Tokenizer) implementation;
//!   self-tunes from observed `Usage` reports.
//!
//! [`CompletionClient`]: ailoop_core::CompletionClient
//! [`CompletionModel`]: ailoop_core::CompletionModel
//! [`CacheControl`]: ailoop_core::CacheControl
//! [`StreamChunk::TurnFinished`]: ailoop_core::StreamChunk::TurnFinished
//! [`AssistantBlock::Reasoning`]: ailoop_core::AssistantBlock::Reasoning
//! [`AssistantBlock::RedactedReasoning`]: ailoop_core::AssistantBlock::RedactedReasoning
//! [`Retryable`]: ailoop_core::Retryable
//! [`RetryingModel`]: ailoop_core::RetryingModel

#![deny(missing_docs)]

mod client;
mod error_body;
mod errors;
mod events;
mod model;
mod request;
mod stream;
mod tokenizer;

pub use client::AnthropicClient;
pub use errors::{AnthropicApiErrorKind, AnthropicError};
pub use model::AnthropicModel;
pub use tokenizer::OnlineCalibratedTokenizer;
