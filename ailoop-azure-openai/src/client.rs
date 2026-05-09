use ailoop_core::CompletionClient;
use reqwest::Client as HttpClient;

use crate::errors::AzureOpenAIError;
use crate::model::AzureOpenAIChatModel;

/// How the client authenticates against Azure OpenAI.
///
/// `ApiKey` sends `api-key: <key>`. `Token` sends `Authorization: Bearer
/// <token>` and is what you want for Microsoft Entra ID / managed identity
/// flows. The two mechanisms are mutually exclusive on a given request.
#[derive(Clone)]
pub enum AzureOpenAIAuth {
    ApiKey(String),
    Token(String),
}

/// Client for the Azure OpenAI v1 API surface.
///
/// Targets the `/openai/v1/...` paths that no longer require an
/// `api-version` query parameter and accept the deployment name as the
/// `model` field in the request body. If you need the legacy dated
/// API surface (`/openai/deployments/{id}/...?api-version=...`), open
/// an issue — the client can be extended to dispatch on a per-request
/// basis without breaking this one.
#[derive(Clone)]
pub struct AzureOpenAIClient {
    pub(crate) http_client: HttpClient,
    pub(crate) endpoint: String,
    pub(crate) auth: AzureOpenAIAuth,
}

impl AzureOpenAIClient {
    pub const ENDPOINT_ENV: &'static str = "AZURE_OPENAI_ENDPOINT";
    pub const API_KEY_ENV: &'static str = "AZURE_OPENAI_API_KEY";
    pub const TOKEN_ENV: &'static str = "AZURE_OPENAI_TOKEN";

    pub fn new(endpoint: impl Into<String>, auth: AzureOpenAIAuth) -> Self {
        Self {
            http_client: HttpClient::new(),
            endpoint: endpoint.into(),
            auth,
        }
    }

    /// Build a client from environment variables. Reads:
    ///
    /// - `AZURE_OPENAI_ENDPOINT` (required) — e.g.
    ///   `https://my-resource.openai.azure.com`.
    /// - Exactly one of `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_TOKEN`
    ///   (required). Setting both is a configuration error.
    ///
    /// Loads `.env` via `dotenvy` if present.
    pub fn from_env() -> Result<Self, AzureOpenAIError> {
        dotenvy::dotenv().ok();
        let endpoint = std::env::var(Self::ENDPOINT_ENV)
            .map_err(|_| AzureOpenAIError::Config(format!("{} is required", Self::ENDPOINT_ENV)))?;
        let api_key = std::env::var(Self::API_KEY_ENV).ok();
        let token = std::env::var(Self::TOKEN_ENV).ok();
        let auth = match (api_key, token) {
            (Some(_), Some(_)) => {
                return Err(AzureOpenAIError::Config(format!(
                    "{} and {} are mutually exclusive",
                    Self::API_KEY_ENV,
                    Self::TOKEN_ENV,
                )));
            }
            (Some(k), None) => AzureOpenAIAuth::ApiKey(k),
            (None, Some(t)) => AzureOpenAIAuth::Token(t),
            (None, None) => {
                return Err(AzureOpenAIError::Config(format!(
                    "one of {} or {} is required",
                    Self::API_KEY_ENV,
                    Self::TOKEN_ENV,
                )));
            }
        };
        Ok(Self::new(endpoint, auth))
    }

    /// Convenience: take ownership and produce a chat model in one step.
    /// `Client::from_env()?.model("my-deployment")`.
    pub fn model(self, deployment: impl Into<String>) -> AzureOpenAIChatModel {
        AzureOpenAIChatModel::new(self, deployment)
    }
}

impl CompletionClient for AzureOpenAIClient {
    type Model = AzureOpenAIChatModel;

    fn completion_model(&self, model_name: impl Into<String>) -> Self::Model {
        AzureOpenAIChatModel::new(self.clone(), model_name.into())
    }
}
