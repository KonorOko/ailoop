// Some wire fields (e.g. `id`, `Choice.index`, `Delta.role`,
// `Delta.reasoning_content`) are intentionally deserialized but not read
// by the state machine — we keep them so the deserializer survives wire
// drift, and for ergonomic test construction. Serde's
// `#[derive(Deserialize)]` does not count toward the dead-code lint.
#![allow(dead_code)]

use serde::Deserialize;

/// One SSE chunk of a streaming chat completion. Many of these arrive per
/// turn, terminated by `data: [DONE]`. With `stream_options.include_usage`
/// set, an extra chunk before `[DONE]` carries `usage` and an empty
/// `choices` array.
#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionsChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Choice {
    pub index: u32,
    #[serde(default)]
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct Delta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallDelta>,
    /// o-series reasoning text emitted via Chat Completions. Tolerated for
    /// forward-compat but ignored by the state machine — Chat Completions
    /// does not round-trip reasoning with a signature. Use the Responses
    /// API model when you need that.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ToolCallDelta {
    pub index: u32,
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
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_delta_chunk() {
        let json = r#"{"id":"x","choices":[{"index":0,"delta":{"content":"Hi"}}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
        assert!(chunk.choices[0].finish_reason.is_none());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn parses_tool_call_first_chunk() {
        let json = r#"{"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("call_1"));
        assert_eq!(tc.kind.as_deref(), Some("function"));
        let f = tc.function.as_ref().unwrap();
        assert_eq!(f.name.as_deref(), Some("get_weather"));
        assert_eq!(f.arguments.as_deref(), Some(""));
    }

    #[test]
    fn parses_tool_call_continuation_chunk() {
        // Subsequent chunks for the same tool_call index only carry partial
        // arguments; id/type/name only appear on the first chunk.
        let json = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":"}}]}}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        let tc = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(tc.index, 0);
        assert!(tc.id.is_none());
        assert!(tc.kind.is_none());
        let f = tc.function.as_ref().unwrap();
        assert!(f.name.is_none());
        assert_eq!(f.arguments.as_deref(), Some("{\"x\":"));
    }

    #[test]
    fn parses_final_usage_chunk() {
        // Emitted before `data: [DONE]` when stream_options.include_usage=true:
        // empty choices, populated usage with prompt_tokens_details.cached_tokens.
        let json = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":5}}}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.choices.is_empty());
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, 5);
    }

    #[test]
    fn parses_finish_reason_tool_calls() {
        let json = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        assert_eq!(
            chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[test]
    fn parses_finish_reason_stop() {
        let json = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn tolerates_unknown_top_level_fields() {
        // Azure injects fields like `system_fingerprint` and
        // `prompt_filter_results` / `content_filter_results`. The deserializer
        // must not deny_unknown_fields — confirm explicitly.
        let json = r#"{"id":"x","system_fingerprint":"fp_123","content_filter_results":{},"prompt_filter_results":[],"choices":[{"index":0,"delta":{"content":"Hi"},"content_filter_results":{}}]}"#;
        let chunk: ChatCompletionsChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
    }
}
