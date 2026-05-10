//! Azure OpenAI v1 Chat Completions adapter for ailoop.
//!
//! Implements [`CompletionClient`] / [`CompletionModel`] against the
//! `POST {endpoint}/openai/v1/chat/completions` streaming endpoint
//! (Server-Sent Events). Plug an [`AzureOpenAIChatModel`] into a
//! [`Conversation`](https://docs.rs/ailoop) by chaining
//! `Conversation::builder(client.model("my-deployment"))`.
//!
//! ## Auth
//!
//! [`AzureOpenAIAuth`] covers the three deployment realities:
//!
//! - `ApiKey` — the static `api-key: <key>` header.
//! - `Token` — a fixed `Authorization: Bearer <token>` for callers
//!   that already hold a short-lived token.
//! - `Provider` — delegates to a caller-supplied [`TokenProvider`]
//!   that fetches a fresh token per request. Use this for Microsoft
//!   Entra ID / managed identity flows where tokens expire.
//!
//! ## Reasoning
//!
//! The Chat Completions wire format has no slot for reasoning blocks;
//! [`AssistantBlock::Reasoning`] / [`AssistantBlock::RedactedReasoning`]
//! are silently dropped on send. Reasoning round-trip on Azure waits
//! on the Responses API model, which is out of scope for 1.0 — see
//! the README. Reasoning round-trip on Anthropic is unaffected.
//!
//! ## Mini-index
//!
//! - [`AzureOpenAIClient`] — endpoint + auth configuration.
//! - [`AzureOpenAIAuth`] — three-arm enum for auth modes.
//! - [`TokenProvider`] — async source of bearer tokens (you implement
//!   this around `azure_identity` / `msal` / your own credential
//!   pipeline).
//! - [`AzureOpenAIChatModel`] — the [`CompletionModel`]
//!   implementation built by [`AzureOpenAIClient::model`] /
//!   [`CompletionClient::completion_model`](ailoop_core::CompletionClient::completion_model).
//! - [`AzureOpenAIError`] / [`AzureOpenAIApiErrorKind`] — typed
//!   failure surface, wired into [`Retryable`] for
//!   [`RetryingModel`](ailoop_core::RetryingModel).
//!
//! [`CompletionClient`]: ailoop_core::CompletionClient
//! [`CompletionModel`]: ailoop_core::CompletionModel
//! [`AssistantBlock::Reasoning`]: ailoop_core::AssistantBlock::Reasoning
//! [`AssistantBlock::RedactedReasoning`]: ailoop_core::AssistantBlock::RedactedReasoning
//! [`Retryable`]: ailoop_core::Retryable

#![deny(missing_docs)]

mod client;
mod error_body;
mod errors;
mod events;
mod model;
mod request;
mod stream;

pub use client::{AzureOpenAIAuth, AzureOpenAIClient, TokenProvider};
pub use errors::{AzureOpenAIApiErrorKind, AzureOpenAIError};
pub use model::AzureOpenAIChatModel;
