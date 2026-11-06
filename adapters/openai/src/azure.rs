//! Azure OpenAI auth: [`AzureAuth`] and [`TokenProvider`].

use std::sync::Arc;

use crate::errors::OpenAIError;

/// Async source of bearer tokens for Microsoft Entra ID (Azure AD).
///
/// The client calls `token()` on every request and trusts the provider
/// to answer quickly when it has a cached token. A provider that fetches
/// on every call works, but costs one network round-trip per request:
/// wrap it in [`CachingTokenProvider`](crate::CachingTokenProvider).
///
/// This crate is bring-your-own: implement the trait around the
/// credential type of `azure_identity`, `msal` or your own pipeline.
///
/// ```
/// use std::sync::Arc;
/// use ailoop_openai::{AzureAuth, OpenAIError, TokenProvider};
///
/// struct StaticToken(String);
///
/// #[async_trait::async_trait]
/// impl TokenProvider for StaticToken {
///     async fn token(&self) -> Result<String, OpenAIError> {
///         Ok(self.0.clone())
///     }
/// }
///
/// let auth = AzureAuth::Provider(Arc::new(StaticToken("...".into())));
/// ```
#[async_trait::async_trait]
pub trait TokenProvider: Send + Sync {
    /// A token for the `Authorization: Bearer <token>` header. A failed
    /// fetch is usually reported as [`OpenAIError::Config`]; the request
    /// fails with that error before anything is sent.
    async fn token(&self) -> Result<String, OpenAIError>;
}

/// How [`OpenAIClient::azure`](crate::OpenAIClient::azure) authenticates.
#[derive(Clone)]
#[non_exhaustive]
pub enum AzureAuth {
    /// A resource API key, sent as `api-key: <key>`.
    ApiKey(String),
    /// A fixed bearer token, sent as `Authorization: Bearer <token>`. It
    /// does not refresh; use [`Provider`](Self::Provider) for anything
    /// long-running.
    Token(String),
    /// A bearer token fetched from a [`TokenProvider`] on every request.
    /// The right choice for Entra ID and managed identity, whose tokens
    /// expire after about an hour.
    Provider(Arc<dyn TokenProvider>),
}

impl AzureAuth {
    /// Wrap `provider` in [`AzureAuth::Provider`].
    pub fn provider<P: TokenProvider + 'static>(provider: P) -> Self {
        Self::Provider(Arc::new(provider))
    }
}

impl std::fmt::Debug for AzureAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Secrets never reach logs.
        f.write_str(match self {
            AzureAuth::ApiKey(_) => "ApiKey(<redacted>)",
            AzureAuth::Token(_) => "Token(<redacted>)",
            AzureAuth::Provider(_) => "Provider(..)",
        })
    }
}
