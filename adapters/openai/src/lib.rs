//! OpenAI Chat Completions adapter for ailoop, for OpenAI, Azure OpenAI
//! and the servers that speak the same API: OpenRouter, Ollama, LM
//! Studio, vLLM, llama.cpp.
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
//! ## Azure OpenAI
//!
//! [`OpenAIClient::azure`] targets a resource's v1 API
//! (`{endpoint}/openai/v1`), where the model name is the deployment
//! name. [`AzureAuth`] picks the credential: an `api-key`, a fixed bearer
//! token, or a [`TokenProvider`] for Microsoft Entra ID, which
//! [`CachingTokenProvider`] can wrap.
//!
//! ```no_run
//! use ailoop_openai::{AzureAuth, OpenAIClient};
//!
//! let azure = OpenAIClient::azure(
//!     "https://my-resource.openai.azure.com",
//!     AzureAuth::ApiKey("...".into()),
//! )
//! .model("my-gpt-4o-deployment");
//! // Or from AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY / _TOKEN:
//! let from_env = OpenAIClient::azure_from_env();
//! ```
//!
//! ## Errors
//!
//! [`OpenAIError`] implements [`ProviderError::kind`](ailoop_core::ProviderError::kind),
//! which sorts every failure into a provider-agnostic
//! [`ProviderErrorKind`](ailoop_core::ProviderErrorKind): no connection,
//! key rejected, rate limit, quota exhausted, model without image or
//! tool support, model not found, content filter, context overflow, or a
//! response cut off mid-stream. A connection failure is `Network`; tell
//! "the local service is not running" from "no network" with
//! [`OpenAIClient::is_local`]. [`Retryable`] gives the retry decision and
//! the delay the server asked for, for
//! [`RetryingModel`](ailoop_core::RetryingModel).
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

mod azure;
mod caching_provider;
mod client;
mod error_body;
mod errors;
mod events;
mod model;
mod models;
mod request;
mod stream;

pub use azure::{AzureAuth, TokenProvider};
pub use caching_provider::CachingTokenProvider;
pub use client::{DEFAULT_CONNECT_TIMEOUT, MaxTokensField, OpenAIClient};
pub use errors::{OpenAIApiErrorKind, OpenAIError};
pub use model::OpenAIChatModel;
pub use models::{ModelCapabilities, ModelInfo};
