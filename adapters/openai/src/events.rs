// Some wire fields (e.g. `Choice.index`, `ToolCallDelta.kind`) are
// deserialized but not read by the state machine; they are kept so the
// shape stays documented and tests can build chunks. Serde's derive does
// not count toward the dead-code lint.
#![allow(dead_code)]

use serde::Deserialize;
use serde_json::Value;

/// One SSE chunk of a streaming chat completion. Many arrive per turn,
/// ending with `data: [DONE]`. With `stream_options.include_usage`, an
/// extra chunk before `[DONE]` carries `usage` and no choices.
#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionsChunk {
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    /// Set when the server fails after the response started and sends
    /// `data: {"error":...}`. Kept as raw JSON because its shape varies
    /// between servers (object, plain string). `null` is `None`.
    #[serde(default)]
    pub error: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Choice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// Visible reasoning as DeepSeek, vLLM and LM Studio send it.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Visible reasoning as Ollama and OpenRouter send it.
    #[serde(default)]
    pub reasoning: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolCallDelta {
    /// Missing on some servers (older Ollama), which send each call
    /// whole in one chunk.
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerates_nulls_and_unknown_fields() {
        // LM Studio and llama.cpp send `"tool_calls": null` and
        // `"content": null`; OpenRouter adds `provider` and `native_finish_reason`.
        let json = r#"{"id":"x","provider":"Groq","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":null},"finish_reason":null,"native_finish_reason":null,"logprobs":null}],"usage":null}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.choices[0].delta.content.is_none());
        assert!(chunk.choices[0].delta.tool_calls.is_none());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn parses_reasoning_fields() {
        let ollama = r#"{"choices":[{"index":0,"delta":{"reasoning":"hmm"}}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(ollama).unwrap();
        assert_eq!(chunk.choices[0].delta.reasoning.as_deref(), Some("hmm"));
        let deepseek = r#"{"choices":[{"index":0,"delta":{"reasoning_content":"hmm"}}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(deepseek).unwrap();
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("hmm")
        );
    }

    #[test]
    fn tool_call_without_index() {
        let json = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        let calls = chunk.choices[0].delta.tool_calls.as_ref().unwrap();
        assert!(calls[0].index.is_none());
    }
}
