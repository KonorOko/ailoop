use std::sync::Arc;

use ailoop_core::CompletionClient;
use reqwest::Client as HttpClient;

use crate::errors::AzureOpenAIError;
use crate::model::AzureOpenAIChatModel;

/// Async source of bearer tokens for Microsoft Entra ID / Azure AD.
///
/// Implementations are responsible for their own caching and refresh
/// policy: the client invokes `token()` on every request and trusts the
/// provider to return a valid token quickly when one is cached. A naive
/// implementation that fetches on every call works but burns one network
/// round-trip per request — wrap your fetch in a cache that respects the
/// token's `expires_on`.
///
/// This crate is intentionally bring-your-own. To wire `azure_identity`
/// or `msal`, implement this trait around the credential type from your
/// chosen library. A trivial mock for tests:
///
/// ```ignore
/// use std::sync::Arc;
/// use ailoop_azure_openai::{AzureOpenAIAuth, AzureOpenAIError, TokenProvider};
///
/// struct StaticToken(String);
///
/// #[async_trait::async_trait]
/// impl TokenProvider for StaticToken {
///     async fn token(&self) -> Result<String, AzureOpenAIError> {
///         Ok(self.0.clone())
///     }
/// }
///
/// let auth = AzureOpenAIAuth::Provider(Arc::new(StaticToken("...".into())));
/// ```
#[async_trait::async_trait]
pub trait TokenProvider: Send + Sync {
    async fn token(&self) -> Result<String, AzureOpenAIError>;
}

/// How the client authenticates against Azure OpenAI.
///
/// - `ApiKey` sends `api-key: <key>`.
/// - `Token` sends `Authorization: Bearer <token>` with a fixed string —
///   useful when you already hold a short-lived token from another
///   process. Note: the token will not refresh; use `Provider` for
///   anything long-running.
/// - `Provider` delegates to a [`TokenProvider`] that fetches a fresh
///   token on each request. This is the right choice for Microsoft Entra
///   ID / managed identity flows where tokens expire (~1h).
#[derive(Clone)]
pub enum AzureOpenAIAuth {
    ApiKey(String),
    Token(String),
    Provider(Arc<dyn TokenProvider>),
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

    /// Convenience: build a client authenticated with a static API key.
    /// Equivalent to `Client::new(endpoint, AzureOpenAIAuth::ApiKey(key))`.
    pub fn with_api_key(endpoint: impl Into<String>, key: impl Into<String>) -> Self {
        Self::new(endpoint, AzureOpenAIAuth::ApiKey(key.into()))
    }

    /// Convenience: build a client authenticated with a fixed bearer token.
    /// The token will not refresh — use [`Self::with_provider`] for any
    /// long-running process where the token can expire.
    pub fn with_token(endpoint: impl Into<String>, token: impl Into<String>) -> Self {
        Self::new(endpoint, AzureOpenAIAuth::Token(token.into()))
    }

    /// Convenience: build a client that fetches a bearer token via the
    /// supplied [`TokenProvider`] on each request. Hides both the
    /// `AzureOpenAIAuth::Provider(...)` and `Arc::new(...)` wrapping.
    pub fn with_provider<P: TokenProvider + 'static>(
        endpoint: impl Into<String>,
        provider: P,
    ) -> Self {
        Self::new(
            endpoint,
            AzureOpenAIAuth::Provider(Arc::new(provider)),
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticProvider(String);

    #[async_trait::async_trait]
    impl TokenProvider for StaticProvider {
        async fn token(&self) -> Result<String, AzureOpenAIError> {
            Ok(self.0.clone())
        }
    }

    struct FailingProvider;

    #[async_trait::async_trait]
    impl TokenProvider for FailingProvider {
        async fn token(&self) -> Result<String, AzureOpenAIError> {
            Err(AzureOpenAIError::Config("token fetch failed".into()))
        }
    }

    #[tokio::test]
    async fn token_provider_returns_token() {
        let p = StaticProvider("fake-token".into());
        assert_eq!(p.token().await.unwrap(), "fake-token");
    }

    #[tokio::test]
    async fn token_provider_can_return_error() {
        let p = FailingProvider;
        match p.token().await {
            Err(AzureOpenAIError::Config(msg)) => assert!(msg.contains("token fetch failed")),
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn with_provider_wraps_concrete_type_in_arc() {
        // Smoke test: confirm the generic bound on with_provider accepts a
        // concrete TokenProvider impl and that the resulting client carries
        // the Provider auth variant.
        let client =
            AzureOpenAIClient::with_provider("https://x.openai.azure.com", StaticProvider("t".into()));
        assert!(matches!(client.auth, AzureOpenAIAuth::Provider(_)));
    }

    #[test]
    fn with_api_key_and_with_token_construct_correct_variants() {
        let by_key = AzureOpenAIClient::with_api_key("https://x.openai.azure.com", "k");
        assert!(matches!(by_key.auth, AzureOpenAIAuth::ApiKey(_)));

        let by_token = AzureOpenAIClient::with_token("https://x.openai.azure.com", "t");
        assert!(matches!(by_token.auth, AzureOpenAIAuth::Token(_)));
    }

    #[test]
    fn auth_provider_clones_via_arc_refcount() {
        // Verifies that AzureOpenAIAuth::Provider doesn't deep-copy the
        // provider, so Client::clone() stays cheap regardless of how
        // expensive the underlying TokenProvider is to construct.
        let provider: Arc<dyn TokenProvider> = Arc::new(StaticProvider("x".into()));
        assert_eq!(Arc::strong_count(&provider), 1);

        let auth = AzureOpenAIAuth::Provider(Arc::clone(&provider));
        assert_eq!(Arc::strong_count(&provider), 2);

        let _auth_clone = auth.clone();
        assert_eq!(Arc::strong_count(&provider), 3);
    }
}
