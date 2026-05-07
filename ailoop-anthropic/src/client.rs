use std::env::VarError;

use ailoop_core::CompletionClient;
use reqwest::Client as HttpClient;

use crate::model::AnthropicModel;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Clone)]
pub struct AnthropicClient {
    pub(crate) http_client: HttpClient,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) anthropic_version: String,
    pub(crate) beta_features: Vec<String>,
}

impl AnthropicClient {
    pub const API_KEY_ENV: &'static str = "ANTHROPIC_API_KEY";

    pub fn new(api_key: String) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    pub fn from_env() -> Result<Self, VarError> {
        Self::from_env_var(Self::API_KEY_ENV)
    }

    pub fn from_env_var(name: &str) -> Result<Self, VarError> {
        dotenvy::dotenv().ok();
        let api_key = std::env::var(name)?;

        Ok(Self::new(api_key))
    }

    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            http_client: HttpClient::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            anthropic_version: DEFAULT_ANTHROPIC_VERSION.into(),
            beta_features: Vec::new(),
        }
    }

    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
        self
    }

    pub fn beta(mut self, feature: impl Into<String>) -> Self {
        self.beta_features.push(feature.into());
        self
    }

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
