use crate::client::AnthropicClient;
use crate::error_body::{classify_http_error, parse_retry_after};
use crate::errors::AnthropicError;
use crate::request::build_body;
use crate::stream::process_response;
use crate::thinking::{Shape, ThinkingMode};

use ailoop_core::{ChatRequest, CompletionModel, StreamChunk};
use futures::stream::BoxStream;

/// [`CompletionModel`] implementation bound to a specific Anthropic
/// model id (e.g. `"claude-sonnet-4-6"`). Constructed via
/// [`AnthropicClient::model`] or
/// [`CompletionClient::completion_model`](ailoop_core::CompletionClient::completion_model).
///
/// Honours every per-block [`CacheControl`](ailoop_core::CacheControl)
/// supplied on the [`ChatRequest`] (system / messages / tools), emits
/// the TTL-broken-down `cache_creation_5m_tokens` /
/// `cache_creation_1h_tokens` counters on
/// [`StreamChunk::TurnFinished`], and round-trips reasoning blocks via
/// [`AssistantBlock::Reasoning`](ailoop_core::AssistantBlock::Reasoning).
///
/// [`CompletionClient`]: ailoop_core::CompletionClient
/// [`StreamChunk::TurnFinished`]: ailoop_core::StreamChunk::TurnFinished
#[derive(Clone)]
pub struct AnthropicChatModel {
    client: AnthropicClient,
    model: String,
    thinking_mode: Option<ThinkingMode>,
    server_fallbacks: bool,
}

/// Beta that gates `fallbacks: "default"`. The array form of
/// `fallbacks` uses a different header (`server-side-fallback-2026-06-01`)
/// and is rejected under this one.
const SERVER_FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

impl AnthropicChatModel {
    /// Bind `client` to a specific `model` id. Prefer
    /// [`AnthropicClient::model`] for the one-client-one-model case
    /// — this constructor is for adapter authors composing their own
    /// pipelines.
    pub fn new(client: AnthropicClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            thinking_mode: None,
            server_fallbacks: false,
        }
    }

    /// Force the wire shape used for
    /// [`ChatRequest::reasoning_effort`](ailoop_core::ChatRequest::reasoning_effort).
    ///
    /// Without this, the shape comes from the model id: Claude 4.5 and
    /// earlier (`claude-haiku-4-5`, `claude-sonnet-4-5`,
    /// `claude-opus-4-5`, `claude-opus-4-1`, `claude-*-4-0`) and the
    /// whole `claude-3-*` family use [`ThinkingMode::Budget`]; every
    /// other id, including ones the adapter does not know, uses
    /// [`ThinkingMode::Adaptive`]. Set it for an id the list cannot
    /// read, such as an alias behind a proxy.
    pub fn thinking_mode(mut self, mode: ThinkingMode) -> Self {
        self.thinking_mode = Some(mode);
        self
    }

    /// Opt into Anthropic's server-side fallbacks.
    ///
    /// When `true`, every request carries the
    /// `anthropic-beta: server-side-fallback-2026-07-01` header and
    /// `"fallbacks": "default"` in the body. When a safety classifier
    /// declines the request, the API re-runs it on the model Anthropic
    /// recommends for that refusal category and returns that answer in
    /// the same response (and on the same stream). Off by default.
    ///
    /// Both are sent for every model id; the adapter never drops them
    /// based on the model. A model without a server-side fallback
    /// (Claude Haiku 5.5, for example) returns its refusal as before.
    ///
    /// What to read on the turn:
    ///
    /// - `StreamChunk::TurnFinished::model` names the model that
    ///   produced the turn, so it differs from
    ///   [`CompletionModel::model`] when a fallback served it.
    /// - [`FinishReason::Refusal`](ailoop_core::FinishReason::Refusal)
    ///   means the whole chain declined.
    ///
    /// A `fallbacks` key in
    /// [`ChatRequest::additional_params`](ailoop_core::ChatRequest::additional_params)
    /// replaces `"default"`, like every other key there. The array form
    /// (`[{"model": "..."}]`) needs the `server-side-fallback-2026-06-01`
    /// beta instead; add it with [`AnthropicClient::beta`] and leave this
    /// setting off.
    ///
    /// When a fallback happens mid-stream, the text the declined model
    /// already streamed stays in the turn, and the fallback model
    /// continues it.
    pub fn server_fallbacks(mut self, enabled: bool) -> Self {
        self.server_fallbacks = enabled;
        self
    }

    /// Body and `anthropic-beta` value for `req`. Split from
    /// [`CompletionModel::chat_stream`] so tests can read the wire
    /// shape without a server.
    fn wire_request(&self, req: &ChatRequest) -> (serde_json::Value, Option<String>) {
        let shape = Shape::resolve(&self.model, self.thinking_mode);
        let mut body = build_body(&self.model, req, shape);

        let mut betas: Vec<&str> = self
            .client
            .beta_features
            .iter()
            .map(String::as_str)
            .collect();

        if self.server_fallbacks {
            if let Some(obj) = body.as_object_mut() {
                obj.entry("fallbacks")
                    .or_insert_with(|| serde_json::json!("default"));
            }
            if !betas.contains(&SERVER_FALLBACK_BETA) {
                betas.push(SERVER_FALLBACK_BETA);
            }
        }

        let beta_header = (!betas.is_empty()).then(|| betas.join(","));
        (body, beta_header)
    }
}

#[async_trait::async_trait]
impl CompletionModel for AnthropicChatModel {
    type Error = AnthropicError;

    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, AnthropicError>>, AnthropicError> {
        let (body, beta_header) = self.wire_request(&req);

        let mut req_builder = self
            .client
            .http_client
            .post(&self.client.base_url)
            .header("x-api-key", &self.client.api_key)
            .header("anthropic-version", &self.client.anthropic_version)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");

        if let Some(betas) = beta_header {
            req_builder = req_builder.header("anthropic-beta", betas);
        }

        let response = req_builder.json(&body).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let retry_after = parse_retry_after(response.headers());
            let body = response.text().await.unwrap_or_default();
            return Err(classify_http_error(status, body, retry_after));
        }

        Ok(process_response(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req() -> ChatRequest {
        ChatRequest::new(vec![], 1024)
    }

    fn model() -> AnthropicChatModel {
        AnthropicClient::new("key".into()).model("claude-opus-5-5")
    }

    #[test]
    fn fallbacks_off_by_default() {
        let (body, betas) = model().wire_request(&req());
        assert!(body.get("fallbacks").is_none());
        assert!(betas.is_none());
    }

    #[test]
    fn server_fallbacks_send_default_and_beta() {
        let (body, betas) = model().server_fallbacks(true).wire_request(&req());
        assert_eq!(body["fallbacks"], json!("default"));
        assert_eq!(betas.as_deref(), Some(SERVER_FALLBACK_BETA));
    }

    #[test]
    fn server_fallbacks_are_sent_for_any_model() {
        // No model table: an id without a server-side fallback still
        // gets the parameter, so the setting is never dropped silently.
        let m = AnthropicClient::new("key".into())
            .model("claude-haiku-5-5")
            .server_fallbacks(true);
        let (body, betas) = m.wire_request(&req());
        assert_eq!(body["fallbacks"], json!("default"));
        assert_eq!(betas.as_deref(), Some(SERVER_FALLBACK_BETA));
    }

    #[test]
    fn server_fallbacks_keep_client_betas_without_duplicates() {
        let m = AnthropicClient::new("key".into())
            .beta("context-management-2025-06-27")
            .beta(SERVER_FALLBACK_BETA)
            .model("claude-opus-5-5")
            .server_fallbacks(true);
        let (_, betas) = m.wire_request(&req());
        assert_eq!(
            betas.as_deref(),
            Some("context-management-2025-06-27,server-side-fallback-2026-07-01")
        );
    }

    #[test]
    fn additional_params_fallbacks_win() {
        let mut r = req();
        r.additional_params = Some(json!({"fallbacks": [{"model": "claude-opus-4-8"}]}));
        let (body, _) = model().server_fallbacks(true).wire_request(&r);
        assert_eq!(body["fallbacks"], json!([{"model": "claude-opus-4-8"}]));
    }

    #[test]
    fn disabling_again_removes_both() {
        let m = model().server_fallbacks(true).server_fallbacks(false);
        let (body, betas) = m.wire_request(&req());
        assert!(body.get("fallbacks").is_none());
        assert!(betas.is_none());
    }
}
