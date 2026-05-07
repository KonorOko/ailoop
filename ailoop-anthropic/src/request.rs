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
