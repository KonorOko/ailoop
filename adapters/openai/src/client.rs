use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use ailoop_core::CompletionClient;
use reqwest::header::{HeaderName, HeaderValue};
use reqwest::{Client as HttpClient, RequestBuilder, Url};

use crate::azure::{AzureAuth, TokenProvider};
use crate::errors::OpenAIError;
use crate::model::OpenAIChatModel;

/// How requests are authenticated.
#[derive(Clone)]
pub(crate) enum Auth {
    /// No credentials: local servers.
    None,
    /// `Authorization: Bearer <token>`: OpenAI, OpenRouter, a fixed Azure
    /// token.
    Bearer(String),
    /// `api-key: <key>`: an Azure resource key.
    AzureKey(String),
    /// `Authorization: Bearer <token>` from a provider, per request.
    Provider(Arc<dyn TokenProvider>),
}

impl Auth {
    fn label(&self) -> &'static str {
        match self {
            Auth::None => "none",
            Auth::Bearer(_) => "bearer <redacted>",
            Auth::AzureKey(_) => "api-key <redacted>",
            Auth::Provider(_) => "token provider",
        }
    }
}

/// Which body field carries [`ChatRequest::max_tokens`](ailoop_core::ChatRequest::max_tokens).
///
/// OpenAI deprecated `max_tokens` in favour of `max_completion_tokens`,
/// and its reasoning models (o-series, gpt-5) reject the old name. Most
/// compatible servers (OpenRouter, Ollama, LM Studio, vLLM) still read
/// `max_tokens`, and some of them ignore the new name, which would
/// leave the output unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MaxTokensField {
    /// Send `max_tokens`. The default for every base URL except
    /// `api.openai.com`.
    MaxTokens,
    /// Send `max_completion_tokens`. The default for `api.openai.com`.
    MaxCompletionTokens,
}

/// Client for the OpenAI Chat Completions API and the servers that
/// implement it: OpenAI itself, OpenRouter, Ollama, LM Studio, vLLM,
/// llama.cpp and the like.
///
/// The base URL is the prefix the provider documents for its
/// OpenAI-compatible API, up to and including the version segment
/// (`https://api.openai.com/v1`, `http://localhost:11434/v1`). The
/// client appends `/chat/completions` and `/models`. A URL with no path
/// (`http://localhost:11434`) gets `/v1`, and a trailing
/// `/chat/completions` is removed, so the URL a user pastes from their
/// provider's docs usually works as is.
///
/// The API key is optional: without one no `Authorization` header is
/// sent, which is what local servers expect.
///
/// Azure OpenAI is built with [`azure`](Self::azure), which targets the
/// resource's v1 API (`{endpoint}/openai/v1`) and authenticates with an
/// [`AzureAuth`].
#[derive(Clone)]
pub struct OpenAIClient {
    pub(crate) http_client: HttpClient,
    pub(crate) base_url: String,
    pub(crate) auth: Auth,
    pub(crate) azure: bool,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) max_tokens_field: MaxTokensField,
    pub(crate) stream_usage: bool,
}

impl std::fmt::Debug for OpenAIClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The key never reaches logs.
        f.debug_struct("OpenAIClient")
            .field("base_url", &self.base_url)
            .field("auth", &self.auth.label())
            .field("azure", &self.azure)
            .field(
                "headers",
                &self.headers.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            )
            .field("max_tokens_field", &self.max_tokens_field)
            .field("stream_usage", &self.stream_usage)
            .finish()
    }
}

/// How long [`OpenAIClient`] waits for a TCP connection before it gives
/// up and reports [`OpenAIError::Unreachable`]. There is no timeout on
/// the response itself: a local server can take a long time to load a
/// model before the first token.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

impl OpenAIClient {
    /// Environment variable [`from_env`](Self::from_env) reads for the
    /// API key. Optional.
    pub const API_KEY_ENV: &'static str = "OPENAI_API_KEY";
    /// Environment variable [`from_env`](Self::from_env) reads for the
    /// base URL. Optional; defaults to [`OPENAI_BASE_URL`](Self::OPENAI_BASE_URL).
    pub const BASE_URL_ENV: &'static str = "OPENAI_BASE_URL";

    /// Environment variable [`azure_from_env`](Self::azure_from_env)
    /// reads for the resource endpoint.
    pub const AZURE_ENDPOINT_ENV: &'static str = "AZURE_OPENAI_ENDPOINT";
    /// Environment variable [`azure_from_env`](Self::azure_from_env)
    /// reads for API-key auth.
    pub const AZURE_API_KEY_ENV: &'static str = "AZURE_OPENAI_API_KEY";
    /// Environment variable [`azure_from_env`](Self::azure_from_env)
    /// reads for fixed bearer-token auth.
    pub const AZURE_TOKEN_ENV: &'static str = "AZURE_OPENAI_TOKEN";

    /// OpenAI's own API.
    pub const OPENAI_BASE_URL: &'static str = "https://api.openai.com/v1";
    /// OpenRouter's OpenAI-compatible API.
    pub const OPENROUTER_BASE_URL: &'static str = "https://openrouter.ai/api/v1";
    /// Ollama's default local address.
    pub const OLLAMA_BASE_URL: &'static str = "http://localhost:11434/v1";
    /// LM Studio's default local server address.
    pub const LM_STUDIO_BASE_URL: &'static str = "http://localhost:1234/v1";

    /// Build a client for `base_url` without an API key. Add one with
    /// [`api_key`](Self::api_key).
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = normalize_base_url(&base_url.into());
        let max_tokens_field = if host_of(&base_url).as_deref() == Some("api.openai.com") {
            MaxTokensField::MaxCompletionTokens
        } else {
            MaxTokensField::MaxTokens
        };
        Self {
            http_client: default_http_client(),
            base_url,
            auth: Auth::None,
            azure: false,
            headers: Vec::new(),
            max_tokens_field,
            stream_usage: true,
        }
    }

    /// OpenAI's API with `api_key`.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(Self::OPENAI_BASE_URL).api_key(api_key)
    }

    /// OpenRouter with `api_key`.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::new(Self::OPENROUTER_BASE_URL).api_key(api_key)
    }

    /// Ollama at its default local address, without a key.
    pub fn ollama() -> Self {
        Self::new(Self::OLLAMA_BASE_URL)
    }

    /// LM Studio at its default local address, without a key.
    pub fn lm_studio() -> Self {
        Self::new(Self::LM_STUDIO_BASE_URL)
    }

    /// Azure OpenAI's v1 API on the resource at `endpoint`
    /// (`https://my-resource.openai.azure.com`). Requests go to
    /// `{endpoint}/openai/v1/...`, without an `api-version` parameter, and
    /// the model name passed to [`model`](Self::model) is the
    /// **deployment** name.
    ///
    /// A pasted endpoint that already ends in `/openai` or `/openai/v1`
    /// is accepted. The output limit is sent as `max_completion_tokens`,
    /// which Azure documents as replacing the deprecated `max_tokens`
    /// (rejected by the o-series reasoning models); override with
    /// [`max_tokens_field`](Self::max_tokens_field).
    ///
    /// [`list_models`](Self::list_models) on Azure lists the base models
    /// the resource can deploy, not your deployments, so its ids are not
    /// valid for [`model`](Self::model). [`model_capabilities`](Self::model_capabilities)
    /// answers `None` for both fields without sending any request.
    pub fn azure(endpoint: impl Into<String>, auth: AzureAuth) -> Self {
        let endpoint = endpoint.into();
        let mut root = endpoint.trim().trim_end_matches('/');
        for suffix in ["/openai/v1", "/openai"] {
            if let Some(stripped) = root.strip_suffix(suffix) {
                root = stripped.trim_end_matches('/');
                break;
            }
        }
        let auth = match auth {
            AzureAuth::ApiKey(key) => Auth::AzureKey(key.trim().to_string()),
            AzureAuth::Token(token) => Auth::Bearer(token.trim().to_string()),
            AzureAuth::Provider(provider) => Auth::Provider(provider),
        };
        Self {
            http_client: default_http_client(),
            base_url: format!("{root}/openai/v1"),
            auth,
            azure: true,
            headers: Vec::new(),
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            stream_usage: true,
        }
    }

    /// Build an Azure client from the environment. Reads:
    ///
    /// - `AZURE_OPENAI_ENDPOINT` (required), the resource endpoint.
    /// - Exactly one of `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_TOKEN`.
    ///
    /// Loads `.env` via `dotenvy` if present. Fails with
    /// [`OpenAIError::Config`] when the endpoint is missing, or when both
    /// or neither credential is set. For Entra ID with refreshing tokens,
    /// use [`azure`](Self::azure) with [`AzureAuth::Provider`].
    pub fn azure_from_env() -> Result<Self, OpenAIError> {
        dotenvy::dotenv().ok();
        let read = |name: &str| {
            std::env::var(name)
                .ok()
                .filter(|value| !value.trim().is_empty())
        };
        let endpoint = read(Self::AZURE_ENDPOINT_ENV).ok_or_else(|| {
            OpenAIError::Config(format!("{} is required", Self::AZURE_ENDPOINT_ENV))
        })?;
        let auth = match (read(Self::AZURE_API_KEY_ENV), read(Self::AZURE_TOKEN_ENV)) {
            (Some(_), Some(_)) => {
                return Err(OpenAIError::Config(format!(
                    "{} and {} are mutually exclusive",
                    Self::AZURE_API_KEY_ENV,
                    Self::AZURE_TOKEN_ENV
                )));
            }
            (Some(key), None) => AzureAuth::ApiKey(key),
            (None, Some(token)) => AzureAuth::Token(token),
            (None, None) => {
                return Err(OpenAIError::Config(format!(
                    "one of {} or {} is required",
                    Self::AZURE_API_KEY_ENV,
                    Self::AZURE_TOKEN_ENV
                )));
            }
        };
        Ok(Self::azure(endpoint, auth))
    }

    /// Build a client from the environment. Reads `OPENAI_BASE_URL`
    /// (optional, defaults to OpenAI) and `OPENAI_API_KEY` (optional).
    /// Loads `.env` via `dotenvy` if present.
    ///
    /// Fails with [`OpenAIError::Config`] when the base URL is OpenAI's
    /// and no key is set, since every request would be rejected.
    pub fn from_env() -> Result<Self, OpenAIError> {
        dotenvy::dotenv().ok();
        let base_url =
            std::env::var(Self::BASE_URL_ENV).unwrap_or_else(|_| Self::OPENAI_BASE_URL.into());
        let client = Self::new(base_url);
        let client = match std::env::var(Self::API_KEY_ENV) {
            Ok(key) => client.api_key(key),
            Err(_) => client,
        };
        if matches!(client.auth, Auth::None)
            && host_of(&client.base_url).as_deref() == Some("api.openai.com")
        {
            return Err(OpenAIError::Config(format!(
                "{} is required for {}",
                Self::API_KEY_ENV,
                Self::OPENAI_BASE_URL
            )));
        }
        Ok(client)
    }

    /// Set the API key, sent as `Authorization: Bearer <key>`. An empty
    /// or blank key counts as no key, so a settings field the user left
    /// empty can be passed through unchanged. Replaces any earlier
    /// credentials, an [`AzureAuth`] included.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        let key = api_key.into().trim().to_string();
        self.auth = if key.is_empty() {
            Auth::None
        } else {
            Auth::Bearer(key)
        };
        self
    }

    /// Like [`api_key`](Self::api_key), for a key that may be absent.
    pub fn maybe_api_key(self, api_key: Option<impl Into<String>>) -> Self {
        match api_key {
            Some(key) => self.api_key(key),
            None => Self {
                auth: Auth::None,
                ..self
            },
        }
    }

    /// Add a header to every request, for example OpenRouter's
    /// `HTTP-Referer` and `X-Title` attribution headers. An invalid name
    /// or value fails the request with [`OpenAIError::Config`].
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Override which field carries the output limit. See
    /// [`MaxTokensField`].
    pub fn max_tokens_field(mut self, field: MaxTokensField) -> Self {
        self.max_tokens_field = field;
        self
    }

    /// Whether to ask for token usage in the stream
    /// (`stream_options.include_usage`). On by default; turn it off for
    /// a server that rejects the field. Without it, usage is reported
    /// as zero unless the server sends it anyway.
    pub fn stream_usage(mut self, enabled: bool) -> Self {
        self.stream_usage = enabled;
        self
    }

    /// Use your own `reqwest::Client`, for proxies, custom TLS roots or
    /// other timeouts. The default one only sets
    /// [`DEFAULT_CONNECT_TIMEOUT`].
    pub fn http_client(mut self, client: HttpClient) -> Self {
        self.http_client = client;
        self
    }

    /// The normalized base URL requests are built from.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Whether the base URL points at this machine or the local network
    /// (loopback, private and link-local addresses, `localhost`,
    /// `*.local`). Decides whether a connection failure reads as
    /// "the local service is not running" or as "no connection".
    pub fn is_local(&self) -> bool {
        is_local_url(&self.base_url)
    }

    /// Whether the client was built with [`azure`](Self::azure).
    pub fn is_azure(&self) -> bool {
        self.azure
    }

    /// Take ownership and produce a chat model in one step.
    pub fn model(self, model: impl Into<String>) -> OpenAIChatModel {
        OpenAIChatModel::new(self, model)
    }

    pub(crate) fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    /// Root of the server without the API version segment, for the
    /// native endpoints of Ollama and LM Studio.
    pub(crate) fn server_root(&self) -> String {
        self.base_url
            .strip_suffix("/v1")
            .unwrap_or(&self.base_url)
            .to_string()
    }

    /// Apply the credentials and the extra headers to `builder`. A
    /// [`TokenProvider`] is asked for a token here, so its errors
    /// surface before the request is sent.
    pub(crate) async fn authorize(
        &self,
        mut builder: RequestBuilder,
    ) -> Result<RequestBuilder, OpenAIError> {
        match &self.auth {
            Auth::None => {}
            Auth::Bearer(token) => builder = builder.bearer_auth(token),
            Auth::AzureKey(key) => {
                let value = HeaderValue::from_str(key)
                    .map_err(|_| OpenAIError::Config("invalid Azure API key".into()))?;
                builder = builder.header("api-key", value);
            }
            Auth::Provider(provider) => {
                let token = provider.token().await?;
                builder = builder.bearer_auth(token.trim());
            }
        }
        for (name, value) in &self.headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| OpenAIError::Config(format!("invalid header name {name:?}")))?;
            let value = HeaderValue::from_str(value)
                .map_err(|_| OpenAIError::Config(format!("invalid value for header {name}")))?;
            builder = builder.header(name, value);
        }
        Ok(builder)
    }

    /// Turn a failure to send into [`OpenAIError::Unreachable`] when no
    /// connection was made, keeping the URL that failed.
    pub(crate) fn send_error(&self, url: &str, error: reqwest::Error) -> OpenAIError {
        if error.is_connect() || (error.is_timeout() && !error.is_body()) {
            OpenAIError::Unreachable {
                url: url.to_string(),
                local: self.is_local(),
                source: error,
            }
        } else {
            OpenAIError::Http(error)
        }
    }
}

impl CompletionClient for OpenAIClient {
    type Model = OpenAIChatModel;

    fn completion_model(&self, model_name: impl Into<String>) -> Self::Model {
        OpenAIChatModel::new(self.clone(), model_name.into())
    }
}

fn default_http_client() -> HttpClient {
    HttpClient::builder()
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .build()
        .unwrap_or_default()
}

/// Trim whitespace and trailing slashes, drop a pasted
/// `/chat/completions` suffix, and add `/v1` to a bare host.
pub(crate) fn normalize_base_url(raw: &str) -> String {
    let mut url = raw.trim().trim_end_matches('/').to_string();
    if let Some(stripped) = url.strip_suffix("/chat/completions") {
        url = stripped.trim_end_matches('/').to_string();
    }
    if let Ok(parsed) = Url::parse(&url)
        && (parsed.path() == "/" || parsed.path().is_empty())
    {
        url.push_str("/v1");
    }
    url
}

fn host_of(url: &str) -> Option<String> {
    Url::parse(url)
        .ok()?
        .host_str()
        .map(|h| h.to_ascii_lowercase())
}

pub(crate) fn is_local_url(url: &str) -> bool {
    let Some(host) = host_of(url) else {
        return false;
    };
    // `host_str` keeps the brackets around an IPv6 literal.
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = bare.parse::<IpAddr>() {
        return is_local_ip(ip);
    }
    host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local")
}

fn is_local_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => {
            // Unique local (fc00::/7) and link-local (fe80::/10).
            let first = v6.segments()[0];
            v6.is_loopback() || (first & 0xfe00) == 0xfc00 || (first & 0xffc0) == 0xfe80
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_pasted_urls() {
        for (raw, expected) in [
            ("http://localhost:11434", "http://localhost:11434/v1"),
            ("http://localhost:11434/", "http://localhost:11434/v1"),
            ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
            (
                " https://openrouter.ai/api/v1 ",
                "https://openrouter.ai/api/v1",
            ),
            (
                "https://api.openai.com/v1/chat/completions",
                "https://api.openai.com/v1",
            ),
            ("http://host:8000/openai", "http://host:8000/openai"),
        ] {
            assert_eq!(normalize_base_url(raw), expected, "{raw}");
        }
    }

    #[test]
    fn local_hosts() {
        for url in [
            "http://localhost:11434/v1",
            "http://127.0.0.1:1234/v1",
            "http://[::1]:1234/v1",
            "http://192.168.1.20:1234/v1",
            "http://10.0.0.5/v1",
            "http://mac-mini.local:11434/v1",
        ] {
            assert!(is_local_url(url), "{url}");
        }
        for url in [
            "https://api.openai.com/v1",
            "https://openrouter.ai/api/v1",
            "http://8.8.8.8/v1",
        ] {
            assert!(!is_local_url(url), "{url}");
        }
    }

    #[test]
    fn presets_and_defaults() {
        let openai = OpenAIClient::openai("sk-test");
        assert_eq!(openai.base_url(), OpenAIClient::OPENAI_BASE_URL);
        assert_eq!(openai.max_tokens_field, MaxTokensField::MaxCompletionTokens);
        let ollama = OpenAIClient::ollama();
        assert!(matches!(ollama.auth, Auth::None));
        assert!(ollama.is_local());
        assert_eq!(ollama.max_tokens_field, MaxTokensField::MaxTokens);
        assert_eq!(ollama.server_root(), "http://localhost:11434");
        assert_eq!(
            OpenAIClient::openrouter("k").endpoint("models"),
            "https://openrouter.ai/api/v1/models"
        );
    }

    #[test]
    fn blank_key_counts_as_none_and_debug_hides_it() {
        assert!(matches!(
            OpenAIClient::lm_studio().api_key("  ").auth,
            Auth::None
        ));
        let client = OpenAIClient::openai("sk-secret");
        assert!(!format!("{client:?}").contains("sk-secret"));
        assert!(matches!(
            OpenAIClient::openai("sk")
                .maybe_api_key(None::<String>)
                .auth,
            Auth::None
        ));
    }

    #[test]
    fn azure_endpoint_and_defaults() {
        for endpoint in [
            "https://res.openai.azure.com",
            "https://res.openai.azure.com/",
            " https://res.openai.azure.com/openai/ ",
            "https://res.openai.azure.com/openai/v1",
        ] {
            let client = OpenAIClient::azure(endpoint, AzureAuth::ApiKey("k".into()));
            assert_eq!(
                client.base_url(),
                "https://res.openai.azure.com/openai/v1",
                "{endpoint}"
            );
        }
        let client = OpenAIClient::azure(
            "https://res.openai.azure.com",
            AzureAuth::ApiKey("k".into()),
        );
        assert!(client.is_azure());
        assert!(!client.is_local());
        assert_eq!(client.max_tokens_field, MaxTokensField::MaxCompletionTokens);
        assert!(matches!(client.auth, Auth::AzureKey(_)));
        assert_eq!(
            client.endpoint("chat/completions"),
            "https://res.openai.azure.com/openai/v1/chat/completions"
        );
        let by_token =
            OpenAIClient::azure("https://res.openai.azure.com", AzureAuth::Token("t".into()));
        assert!(matches!(by_token.auth, Auth::Bearer(_)));
        assert!(!OpenAIClient::ollama().is_azure());
    }

    #[test]
    fn azure_debug_hides_secrets() {
        let client = OpenAIClient::azure(
            "https://res.openai.azure.com",
            AzureAuth::ApiKey("azure-secret".into()),
        );
        assert!(!format!("{client:?}").contains("azure-secret"));
        assert!(!format!("{:?}", AzureAuth::Token("tok-secret".into())).contains("tok-secret"));
    }

    struct StaticProvider(&'static str);

    #[async_trait::async_trait]
    impl TokenProvider for StaticProvider {
        async fn token(&self) -> Result<String, OpenAIError> {
            Ok(self.0.into())
        }
    }

    #[test]
    fn provider_auth_clones_by_refcount() {
        let provider: Arc<dyn TokenProvider> = Arc::new(StaticProvider("x"));
        let client = OpenAIClient::azure(
            "https://res.openai.azure.com",
            AzureAuth::Provider(Arc::clone(&provider)),
        );
        assert_eq!(Arc::strong_count(&provider), 2);
        let _copy = client.clone();
        assert_eq!(Arc::strong_count(&provider), 3);
    }
}
