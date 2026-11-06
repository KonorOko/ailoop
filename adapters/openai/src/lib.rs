//! OpenAI Chat Completions adapter for ailoop, for OpenAI and the servers
//! that speak the same API: OpenRouter, Ollama, LM Studio, vLLM,
//! llama.cpp.
//!
//! Implements [`CompletionClient`] / [`CompletionModel`] against
//! `POST {base_url}/chat/completions` with streaming (Server-Sent
//! Events), and lists models with `GET {base_url}/models`.
//!
//! ```no_run
//! use ailoop_openai::OpenAIClient;
//!
//! // OpenAI, OpenRouter, or any URL with an optional key.
//! let cloud = OpenAIClient::openai("sk-...").model("gpt-5-mini");
//! let local = OpenAIClient::ollama().model("llama3.2");
//! let custom = OpenAIClient::new("http://192.168.1.20:1234/v1")
//!     .maybe_api_key(None::<String>)
//!     .model("qwen2.5-vl-7b-instruct");
//! ```
//!
//! ## Errors
//!
//! [`OpenAIError::category`] sorts every failure into what an app shows:
//! offline, local service not running (with its URL), key rejected, rate
//! limit (with the retry delay and time), quota exhausted, model without
//! image or tool support, model not found, context overflow, or a
//! response cut off mid-stream. [`OpenAIError`] also implements
//! [`Retryable`] for [`RetryingModel`](ailoop_core::RetryingModel) and
//! [`ProviderError`](ailoop_core::ProviderError) for history compaction.
//!
//! "Does not support images/tools" is recognized from the server's
//! message, since no server has a code for it, and only when the request
//! carried images or tools.
//!
//! ## Capabilities
//!
//! [`OpenAIClient::model_capabilities`] answers whether a model accepts
//! images and tools when the server documents it (OpenRouter's listing,
//! Ollama's `/api/show`, LM Studio's `/api/v1/models`). OpenAI's API does
//! not publish it; those fields stay `None` rather than being guessed
//! from the model's name.
//!
//! ## Not covered
//!
//! - The Responses API, and reasoning round-trip: reasoning blocks are
//!   received but not sent back.
//! - Documents (PDF), images in tool results, and `Source::FileId`
//!   images: they fail with [`OpenAIError::UnsupportedContent`] before
//!   any request is made.
//!
//! [`CompletionClient`]: ailoop_core::CompletionClient
//! [`CompletionModel`]: ailoop_core::CompletionModel
//! [`Retryable`]: ailoop_core::Retryable

#![deny(missing_docs)]

mod client;
mod error_body;
mod errors;
mod events;
mod model;
mod models;
mod request;
mod stream;

pub use client::{DEFAULT_CONNECT_TIMEOUT, MaxTokensField, OpenAIClient};
pub use errors::{Capability, ErrorCategory, OpenAIApiErrorKind, OpenAIError};
pub use model::OpenAIChatModel;
pub use models::{ModelCapabilities, ModelInfo};
