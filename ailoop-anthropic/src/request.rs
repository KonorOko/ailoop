use ailoop_core::{
    AssistantBlock, ChatRequest, Message, ToolChoice, ToolDefinition, ToolResultContent, UserBlock,
};
use serde_json::json;

pub fn build_body(model: &str, req: &ChatRequest) -> serde_json::Value {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(model));
    body.insert("max_tokens".into(), json!(req.max_tokens));
    body.insert("stream".into(), json!(true));
    body.insert(
        "messages".into(),
        json!(to_anthropic_messages(&req.messages)),
    );

    if let Some(system) = &req.system_prompt {
        body.insert("system".into(), json!(system));
    }

    if let Some(tools) = &req.tools {
        if !tools.is_empty() {
            body.insert("tools".into(), json!(to_anthropic_tools(tools)));
        }
    }

    if let Some(t) = req.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if let Some(k) = req.top_k {
        body.insert("top_k".into(), json!(k));
    }
    if !req.stop_sequences.is_empty() {
        body.insert("stop_sequences".into(), json!(req.stop_sequences));
    }

    // Anthropic carries `disable_parallel_tool_use` *inside* the
    // tool_choice object, so emitting one without the other still has
    // to materialise a tool_choice (defaulting to `auto`).
    if req.tool_choice.is_some() || req.disable_parallel_tool_use.is_some() {
        body.insert(
            "tool_choice".into(),
            to_anthropic_tool_choice(req.tool_choice.as_ref(), req.disable_parallel_tool_use),
        );
    }

    if let Some(extra) = &req.additional_params {
        if let Some(map) = extra.as_object() {
            for (k, v) in map {
                body.insert(k.clone(), v.clone());
            }
        }
    }

    serde_json::Value::Object(body)
}

fn to_anthropic_messages(messages: &[Message]) -> Vec<serde_json::Value> {
    messages.iter().map(to_anthropic_message).collect()
}

fn to_anthropic_message(message: &Message) -> serde_json::Value {
    match message {
        Message::User { blocks } => json!({
            "role": "user",
            "content": blocks.iter().map(to_anthropic_user_block).collect::<Vec<_>>(),
        }),
        Message::Assistant { blocks } => json!({
            "role": "assistant",
            "content": blocks.iter().map(to_anthropic_assistant_block).collect::<Vec<_>>(),
        }),
    }
}

fn to_anthropic_user_block(block: &UserBlock) -> serde_json::Value {
    match block {
        UserBlock::Text(text) => json!({ "type": "text", "text": text }),
        UserBlock::ToolResult { call_id, content } => {
            let (text, is_error) = match content {
                ToolResultContent::Text(t) => (t.as_str(), false),
                ToolResultContent::Error(e) => (e.as_str(), true),
            };
            json!({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": text,
                "is_error": is_error,
            })
        }
    }
}

fn to_anthropic_assistant_block(block: &AssistantBlock) -> serde_json::Value {
    match block {
        AssistantBlock::Text(text) => json!({ "type": "text", "text": text }),
        AssistantBlock::ToolCall { id, name, args } => json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": args,
        }),
        AssistantBlock::Reasoning { text, signature } => {
            // Anthropic requires the signature verbatim when this block lives
            // in a turn that the next request continues with tool_result.
            // Default to empty when absent (e.g. block was constructed from a
            // snapshot or another provider) — the API will validate.
            let sig = signature.as_deref().unwrap_or("");
            json!({
                "type": "thinking",
                "thinking": text,
                "signature": sig,
            })
        }
        AssistantBlock::RedactedReasoning { data } => json!({
            "type": "redacted_thinking",
            "data": data,
        }),
    }
}

fn to_anthropic_tool_choice(
    choice: Option<&ToolChoice>,
    disable_parallel: Option<bool>,
) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    match choice.unwrap_or(&ToolChoice::Auto) {
        ToolChoice::Auto => {
            obj.insert("type".into(), json!("auto"));
        }
        ToolChoice::Any => {
            obj.insert("type".into(), json!("any"));
        }
        ToolChoice::Tool { name } => {
            obj.insert("type".into(), json!("tool"));
            obj.insert("name".into(), json!(name));
        }
        ToolChoice::None_ => {
            obj.insert("type".into(), json!("none"));
        }
    }
    if let Some(flag) = disable_parallel {
        obj.insert("disable_parallel_tool_use".into(), json!(flag));
    }
    serde_json::Value::Object(obj)
}

fn to_anthropic_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::ChatRequest;

    fn base_req() -> ChatRequest {
        ChatRequest {
            messages: vec![],
            system_prompt: None,
            tools: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: vec![],
            max_tokens: 1024,
            additional_params: None,
            tool_choice: None,
            disable_parallel_tool_use: None,
        }
    }

    #[test]
    fn omits_tool_choice_when_unset() {
        let body = build_body("claude", &base_req());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn maps_tool_choice_auto() {
        let req = ChatRequest {
            tool_choice: Some(ToolChoice::Auto),
            ..base_req()
        };
        let body = build_body("claude", &req);
        assert_eq!(body["tool_choice"], json!({ "type": "auto" }));
    }

    #[test]
    fn maps_tool_choice_any() {
        let req = ChatRequest {
            tool_choice: Some(ToolChoice::Any),
            ..base_req()
        };
        let body = build_body("claude", &req);
        assert_eq!(body["tool_choice"], json!({ "type": "any" }));
    }

    #[test]
    fn maps_tool_choice_specific_tool() {
        let req = ChatRequest {
            tool_choice: Some(ToolChoice::Tool {
                name: "get_weather".into(),
            }),
            ..base_req()
        };
        let body = build_body("claude", &req);
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "tool", "name": "get_weather" })
        );
    }

    #[test]
    fn maps_tool_choice_none() {
        let req = ChatRequest {
            tool_choice: Some(ToolChoice::None_),
            ..base_req()
        };
        let body = build_body("claude", &req);
        assert_eq!(body["tool_choice"], json!({ "type": "none" }));
    }

    /// `disable_parallel_tool_use` lives inside `tool_choice` on
    /// Anthropic, so setting it without an explicit choice must still
    /// emit a tool_choice object — defaulting to `auto`.
    #[test]
    fn disable_parallel_alone_emits_auto_tool_choice_with_flag() {
        let req = ChatRequest {
            disable_parallel_tool_use: Some(true),
            ..base_req()
        };
        let body = build_body("claude", &req);
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "auto", "disable_parallel_tool_use": true })
        );
    }

    #[test]
    fn disable_parallel_combines_with_explicit_tool_choice() {
        let req = ChatRequest {
            tool_choice: Some(ToolChoice::Any),
            disable_parallel_tool_use: Some(true),
            ..base_req()
        };
        let body = build_body("claude", &req);
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "any", "disable_parallel_tool_use": true })
        );
    }

    #[test]
    fn reasoning_block_serializes_with_signature() {
        let block = AssistantBlock::Reasoning {
            text: "step 1".into(),
            signature: Some("abc123".into()),
        };
        let json = to_anthropic_assistant_block(&block);
        assert_eq!(
            json,
            json!({
                "type": "thinking",
                "thinking": "step 1",
                "signature": "abc123",
            })
        );
    }

    #[test]
    fn reasoning_block_without_signature_serializes_with_empty_string() {
        let block = AssistantBlock::Reasoning {
            text: "step 1".into(),
            signature: None,
        };
        let json = to_anthropic_assistant_block(&block);
        assert_eq!(json["signature"], json!(""));
    }

    #[test]
    fn redacted_reasoning_block_serializes_as_redacted_thinking() {
        let block = AssistantBlock::RedactedReasoning {
            data: "opaque-blob".into(),
        };
        let json = to_anthropic_assistant_block(&block);
        assert_eq!(
            json,
            json!({
                "type": "redacted_thinking",
                "data": "opaque-blob",
            })
        );
    }
}
