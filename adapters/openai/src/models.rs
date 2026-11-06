use reqwest::Response;
use serde_json::{Value, json};

use crate::client::OpenAIClient;
use crate::error_body::{RequestFeatures, classify_http_error};
use crate::errors::OpenAIError;

/// What is known about a model's input. `None` means the server did not
/// say; the adapter never infers it from the model's name.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelCapabilities {
    /// Whether the model accepts images.
    pub images: Option<bool>,
    /// Whether the model accepts tools.
    pub tools: Option<bool>,
}

impl ModelCapabilities {
    /// Build a value from both fields.
    pub fn new(images: Option<bool>, tools: Option<bool>) -> Self {
        Self { images, tools }
    }

    /// Whether either field is still unknown.
    pub fn is_incomplete(&self) -> bool {
        self.images.is_none() || self.tools.is_none()
    }

    fn fill_from(&mut self, other: ModelCapabilities) {
        self.images = self.images.or(other.images);
        self.tools = self.tools.or(other.tools);
    }
}

/// One entry of `GET /models`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelInfo {
    /// The id to pass to [`OpenAIClient::model`].
    pub id: String,
    /// A display name, when the server sends one (OpenRouter).
    pub display_name: Option<String>,
    /// `owned_by` (OpenAI, Ollama: `library`, LM Studio: the publisher).
    pub owned_by: Option<String>,
    /// `created`, in seconds since the Unix epoch.
    pub created: Option<u64>,
    /// The context window in tokens, when the server sends it
    /// (OpenRouter).
    pub context_length: Option<u64>,
    /// Capabilities the listing itself states. Only OpenRouter's
    /// listing does (`architecture.input_modalities`,
    /// `supported_parameters`); for other servers both are `None` and
    /// [`OpenAIClient::model_capabilities`] can ask further.
    pub capabilities: ModelCapabilities,
}

impl ModelInfo {
    fn from_json(value: &Value) -> Option<Self> {
        let id = value.get("id")?.as_str()?.to_string();
        let string = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
        let images = value
            .pointer("/architecture/input_modalities")
            .and_then(Value::as_array)
            .map(|m| m.iter().any(|v| v.as_str() == Some("image")));
        let tools = value
            .get("supported_parameters")
            .and_then(Value::as_array)
            .map(|p| p.iter().any(|v| v.as_str() == Some("tools")));
        Some(Self {
            id,
            display_name: string("name"),
            owned_by: string("owned_by"),
            created: value.get("created").and_then(Value::as_u64),
            context_length: value.get("context_length").and_then(Value::as_u64),
            capabilities: ModelCapabilities { images, tools },
        })
    }
}

impl OpenAIClient {
    /// List the models the server offers, from `GET {base_url}/models`,
    /// in the server's order. On Ollama these are the installed models;
    /// on LM Studio the downloaded ones.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, OpenAIError> {
        let url = self.endpoint("models");
        let response = self.get_json_response(&url).await?;
        let body: Value = response.json().await.map_err(OpenAIError::Http)?;
        let data = body
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| OpenAIError::Config(format!("{url} did not return a model list")))?;
        Ok(data.iter().filter_map(ModelInfo::from_json).collect())
    }

    /// Find out whether `model` accepts images and tools, from what the
    /// server documents. In order:
    ///
    /// 1. The model's entry in [`list_models`](Self::list_models)
    ///    (OpenRouter states both).
    /// 2. Ollama's `POST /api/show`: `capabilities` lists `"vision"` and
    ///    `"tools"`.
    /// 3. LM Studio's `GET /api/v1/models`: `capabilities.vision`, and
    ///    `capabilities.trained_for_tool_use` when it is `true`. LM Studio
    ///    also runs tools on models not trained for them, so `false` stays
    ///    unknown. Older LM Studio: `GET /api/v0/models/{id}`, whose
    ///    `type` is `"vlm"` for vision models.
    ///
    /// Steps 2 and 3 are only tried against local servers. What none of
    /// them states stays `None`: OpenAI's own API publishes neither.
    /// Errors from the listing propagate; a failed native lookup only
    /// leaves its fields unknown.
    pub async fn model_capabilities(&self, model: &str) -> Result<ModelCapabilities, OpenAIError> {
        let mut caps = self
            .list_models()
            .await?
            .into_iter()
            .find(|m| m.id == model)
            .map(|m| m.capabilities)
            .unwrap_or_default();
        if !caps.is_incomplete() || !self.is_local() {
            return Ok(caps);
        }
        if let Some(found) = self.ollama_capabilities(model).await {
            caps.fill_from(found);
        }
        if caps.is_incomplete()
            && let Some(found) = self.lm_studio_capabilities(model).await
        {
            caps.fill_from(found);
        }
        Ok(caps)
    }

    async fn ollama_capabilities(&self, model: &str) -> Option<ModelCapabilities> {
        let url = format!("{}/api/show", self.server_root());
        let builder = self
            .authorize(self.http_client.post(&url).json(&json!({ "model": model })))
            .ok()?;
        let response = builder.send().await.ok()?;
        if !response.status().is_success() {
            return None;
        }
        let body: Value = response.json().await.ok()?;
        let list = body.get("capabilities")?.as_array()?;
        let has = |name: &str| list.iter().any(|v| v.as_str() == Some(name));
        Some(ModelCapabilities {
            images: Some(has("vision")),
            tools: Some(has("tools")),
        })
    }

    async fn lm_studio_capabilities(&self, model: &str) -> Option<ModelCapabilities> {
        let root = self.server_root();
        if let Some(caps) = self.lm_studio_v1(&root, model).await {
            return Some(caps);
        }
        let url = format!("{root}/api/v0/models/{model}");
        let response = self
            .authorize(self.http_client.get(&url))
            .ok()?
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            return None;
        }
        let body: Value = response.json().await.ok()?;
        let images = match body.get("type")?.as_str()? {
            "vlm" => Some(true),
            "llm" => Some(false),
            _ => None,
        };
        Some(ModelCapabilities {
            images,
            tools: None,
        })
    }

    async fn lm_studio_v1(&self, root: &str, model: &str) -> Option<ModelCapabilities> {
        let url = format!("{root}/api/v1/models");
        let response = self
            .authorize(self.http_client.get(&url))
            .ok()?
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            return None;
        }
        let body: Value = response.json().await.ok()?;
        let entries = body
            .get("models")
            .or_else(|| body.get("data"))
            .and_then(Value::as_array)?;
        let entry = entries.iter().find(|e| {
            e.get("key").and_then(Value::as_str) == Some(model)
                || e.get("id").and_then(Value::as_str) == Some(model)
        })?;
        let caps = entry.get("capabilities")?;
        Some(ModelCapabilities {
            images: caps.get("vision").and_then(Value::as_bool),
            tools: caps
                .get("trained_for_tool_use")
                .and_then(Value::as_bool)
                .filter(|trained| *trained),
        })
    }

    async fn get_json_response(&self, url: &str) -> Result<Response, OpenAIError> {
        let builder = self
            .authorize(self.http_client.get(url))?
            .header("accept", "application/json");
        let response = builder.send().await.map_err(|e| self.send_error(url, e))?;
        if !response.status().is_success() {
            let status = response.status();
            let headers = response.headers().clone();
            let body = response.text().await.unwrap_or_default();
            return Err(classify_http_error(
                status,
                &headers,
                body,
                RequestFeatures::default(),
            ));
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openrouter_entry() {
        let entry = json!({
            "id": "openai/gpt-4o",
            "name": "OpenAI: GPT-4o",
            "context_length": 128000,
            "architecture": { "input_modalities": ["text", "image"] },
            "supported_parameters": ["max_tokens", "tools", "temperature"]
        });
        let info = ModelInfo::from_json(&entry).unwrap();
        assert_eq!(info.display_name.as_deref(), Some("OpenAI: GPT-4o"));
        assert_eq!(info.context_length, Some(128000));
        assert_eq!(
            info.capabilities,
            ModelCapabilities::new(Some(true), Some(true))
        );
    }

    #[test]
    fn plain_entry_has_unknown_capabilities() {
        let entry = json!({ "id": "llama3.2:latest", "object": "model", "created": 1730000000, "owned_by": "library" });
        let info = ModelInfo::from_json(&entry).unwrap();
        assert_eq!(info.owned_by.as_deref(), Some("library"));
        assert_eq!(info.created, Some(1730000000));
        assert_eq!(info.capabilities, ModelCapabilities::default());
    }
}
