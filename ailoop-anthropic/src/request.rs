use ailoop_core::{
    AssistantBlock, ChatRequest, Message, ToolDefinition, ToolResultContent, UserBlock,
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
