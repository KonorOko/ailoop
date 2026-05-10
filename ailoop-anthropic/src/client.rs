use std::env::VarError;

use ailoop_core::CompletionClient;
use reqwest::Client as HttpClient;

use crate::model::AnthropicModel;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages API client.
///
/// Holds the API key, the base URL (override only for proxies / VPC
/// endpoints), the `anthropic-version` to send on every request, and
/// the comma-joined `anthropic-beta` feature list. Cheap to clone —
/// the inner `reqwest::Client` is reference-counted, so a single
/// configured instance can fan out across many [`AnthropicModel`]s.
#[derive(Clone)]
pub struct AnthropicClient {
    pub(crate) http_client: HttpClient,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) anthropic_version: String,
    pub(crate) beta_features: Vec<String>,
}

impl AnthropicClient {
    /// Environment variable [`from_env`](Self::from_env) reads.
    pub const API_KEY_ENV: &'static str = "ANTHROPIC_API_KEY";

    /// Build a client targeting the public Anthropic endpoint with the
    /// supplied API key. Uses the bundled default `anthropic-version`.
    pub fn new(api_key: String) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Read [`API_KEY_ENV`](Self::API_KEY_ENV) (loading `.env` first
    /// if present) and build a client with the public endpoint.
    /// Returns [`VarError`] when the variable is missing or unset.
    pub fn from_env() -> Result<Self, VarError> {
        Self::from_env_var(Self::API_KEY_ENV)
    }

    /// Like [`from_env`](Self::from_env) but reads `name` instead of
    /// the default variable. Useful when multiple keys live in the
    /// same environment.
    pub fn from_env_var(name: &str) -> Result<Self, VarError> {
        dotenvy::dotenv().ok();
        let api_key = std::env::var(name)?;

        Ok(Self::new(api_key))
    }

    /// Build a client pointing at a custom `base_url` — typically a
    /// VPC proxy or the Anthropic-on-Bedrock-style passthrough URL.
    /// The URL must accept the same `POST /v1/messages` shape as the
    /// public endpoint.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            http_client: HttpClient::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            anthropic_version: DEFAULT_ANTHROPIC_VERSION.into(),
            beta_features: Vec::new(),
        }
    }

    /// Override the `anthropic-version` header sent on every request.
    /// Default is the most recent stable version this crate was
    /// shipped against.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
        self
    }

    /// Add `feature` to the `anthropic-beta` header. Successive calls
    /// accumulate; multiple features are joined with `,` on the wire.
    pub fn beta(mut self, feature: impl Into<String>) -> Self {
        self.beta_features.push(feature.into());
        self
    }

    /// Take ownership of the client and produce an [`AnthropicModel`]
    /// bound to `model` (e.g. `"claude-sonnet-4-6"`). Use when one
    /// client maps to one model; for one-to-many use
    /// [`CompletionClient::completion_model`] which clones internally.
    pub fn model(self, model: impl Into<String>) -> AnthropicModel {
        AnthropicModel::new(self, model)
    }
}

impl CompletionClient for AnthropicClient {
    type Model = AnthropicModel;

    fn completion_model(&self, model_name: impl Into<String>) -> Self::Model {
        AnthropicModel::new(self.clone(), model_name.into())
    }
}
