use ailoop_core::{
    AssistantBlock, ChatRequest, Message, ToolChoice, ToolDefinition, ToolResultContent, UserBlock,
};
use serde_json::{Value, json};

/// Build the Chat Completions v1 request body. The deployment name is
/// passed as `model` (Azure v1 takes deployments via the `model` field
/// rather than the URL path).
pub fn build_body(deployment: &str, req: &ChatRequest) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(deployment));
    body.insert("stream".into(), json!(true));
    body.insert("stream_options".into(), json!({ "include_usage": true }));
    body.insert("max_tokens".into(), json!(req.max_tokens));
    let system_text = req.system_prompt.as_ref().map(|s| s.as_text());
    body.insert(
        "messages".into(),
        json!(to_messages(system_text.as_deref(), &req.messages)),
    );

    if let Some(tools) = &req.tools
        && !tools.is_empty()
    {
        body.insert("tools".into(), json!(to_tools(tools)));
    }

    if let Some(t) = req.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = req.top_p {
        body.insert("top_p".into(), json!(p));
    }
    // top_k is silently dropped: Chat Completions does not accept it. Models
    // that do can be reached by setting it via `additional_params`.
    if !req.stop_sequences.is_empty() {
        body.insert("stop".into(), json!(req.stop_sequences));
    }

    if let Some(choice) = &req.tool_choice {
        body.insert("tool_choice".into(), to_chat_tool_choice(choice));
    }
    // Chat Completions exposes the inverse: `parallel_tool_calls` (default true).
    if let Some(disable) = req.disable_parallel_tool_use {
        body.insert("parallel_tool_calls".into(), json!(!disable));
    }

    if let Some(extra) = &req.additional_params
        && let Some(map) = extra.as_object()
    {
        for (k, v) in map {
            body.insert(k.clone(), v.clone());
        }
    }

    Value::Object(body)
}

fn to_messages(system_prompt: Option<&str>, messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    // Azure auto-converts `system` to `developer` for o-series; we always
    // emit `system` and let the service do the right thing.
    if let Some(prompt) = system_prompt {
        out.push(json!({ "role": "system", "content": prompt }));
    }
    for msg in messages {
        match msg {
            Message::User { blocks } => append_user_blocks(&mut out, blocks),
            Message::Assistant { blocks } => append_assistant_blocks(&mut out, blocks),
            _ => {}
        }
    }
    out
}

fn append_user_blocks(out: &mut Vec<Value>, blocks: &[UserBlock]) {
    let mut text_buf: Vec<&str> = Vec::new();

    for block in blocks {
        match block {
            UserBlock::Text { text, .. } => text_buf.push(text.as_str()),
            UserBlock::ToolResult {
                call_id, content, ..
            } => {
                if !text_buf.is_empty() {
                    out.push(json!({
                        "role": "user",
                        "content": text_buf.join("\n"),
                    }));
                    text_buf.clear();
                }
                // Chat Completions has no `is_error` flag on tool results;
                // Error variant ships the error text as the content body.
                let text = match content {
                    ToolResultContent::Text(t) => t.as_str(),
                    ToolResultContent::Error(e) => e.as_str(),
                    _ => "",
                };
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": text,
                }));
            }
            _ => {}
        }
    }
    if !text_buf.is_empty() {
        out.push(json!({
            "role": "user",
            "content": text_buf.join("\n"),
        }));
    }
}

fn append_assistant_blocks(out: &mut Vec<Value>, blocks: &[AssistantBlock]) {
    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for block in blocks {
        match block {
            AssistantBlock::Text { text, .. } => text_parts.push(text.as_str()),
            AssistantBlock::ToolCall { id, name, args, .. } => {
                // Chat Completions requires `arguments` as a JSON-encoded
                // string, not an object.
                let arguments = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": arguments },
                }));
            }
            // Reasoning blocks have no slot in Chat Completions: the
            // signature would be lost across turns. Use
            // AzureOpenAIResponsesModel for reasoning round-trip.
            AssistantBlock::Reasoning { .. } | AssistantBlock::RedactedReasoning { .. } => {}
            _ => {}
        }
    }

    let mut msg = serde_json::Map::new();
    msg.insert("role".into(), json!("assistant"));
    msg.insert(
        "content".into(),
        if text_parts.is_empty() {
            Value::Null
        } else {
            json!(text_parts.join("\n"))
        },
    );
    if !tool_calls.is_empty() {
        msg.insert("tool_calls".into(), json!(tool_calls));
    }
    out.push(Value::Object(msg));
}

fn to_chat_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        // Chat Completions calls this `"required"`, not `"any"`.
        ToolChoice::Any => json!("required"),
        ToolChoice::Tool { name } => json!({
            "type": "function",
            "function": { "name": name },
        }),
        ToolChoice::None_ => json!("none"),
        // ToolChoice is `#[non_exhaustive]`; future variants fall back
        // to the provider default.
        _ => json!("auto"),
    }
}

fn to_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{ChatRequest, ToolTag};

    fn base_req() -> ChatRequest {
        ChatRequest::new(Vec::new(), 1024)
    }

    #[test]
    fn serializes_simple_text_turn() {
        let mut req = base_req();
        req.messages = vec![Message::user("hi"), Message::assistant_text("hello")];
        let body = build_body("gpt-4o-mini-deployment", &req);
        assert_eq!(body["model"], json!("gpt-4o-mini-deployment"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"], json!({ "include_usage": true }));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], json!("user"));
        assert_eq!(messages[0]["content"], json!("hi"));
        assert_eq!(messages[1]["role"], json!("assistant"));
        assert_eq!(messages[1]["content"], json!("hello"));
    }

    #[test]
    fn serializes_assistant_tool_call_with_stringified_arguments() {
        let mut req = base_req();
        req.messages = vec![Message::Assistant {
            blocks: vec![AssistantBlock::tool_call(
                "call_1",
                "get_weather",
                json!({ "location": "SF", "units": "C" }),
            )],
        }];
        let body = build_body("dep", &req);
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], json!("assistant"));
        assert_eq!(msg["content"], Value::Null);
        let tc = &msg["tool_calls"][0];
        assert_eq!(tc["id"], json!("call_1"));
        assert_eq!(tc["type"], json!("function"));
        assert_eq!(tc["function"]["name"], json!("get_weather"));
        // arguments is a STRING containing JSON, not an object.
        let args_str = tc["function"]["arguments"].as_str().unwrap();
        let round_trip: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(round_trip, json!({ "location": "SF", "units": "C" }));
    }

    #[test]
    fn serializes_tool_result_as_role_tool() {
        let mut req = base_req();
        req.messages = vec![Message::User {
            blocks: vec![UserBlock::tool_result(
                "call_1",
                ToolResultContent::Text("70F".into()),
            )],
        }];
        let body = build_body("dep", &req);
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], json!("tool"));
        assert_eq!(msg["tool_call_id"], json!("call_1"));
        assert_eq!(msg["content"], json!("70F"));
    }

    #[test]
    fn serializes_tool_result_error_as_role_tool_text() {
        let mut req = base_req();
        req.messages = vec![Message::User {
            blocks: vec![UserBlock::tool_result(
                "call_1",
                ToolResultContent::Error("API down".into()),
            )],
        }];
        let body = build_body("dep", &req);
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], json!("tool"));
        assert_eq!(msg["tool_call_id"], json!("call_1"));
        // Chat Completions has no is_error flag; the error text is the body.
        assert_eq!(msg["content"], json!("API down"));
    }

    #[test]
    fn splits_user_blocks_with_mixed_text_and_tool_result() {
        let mut req = base_req();
        req.messages = vec![Message::User {
            blocks: vec![
                UserBlock::text("before"),
                UserBlock::tool_result("c1", ToolResultContent::Text("ok".into())),
                UserBlock::text("after"),
            ],
        }];
        let body = build_body("dep", &req);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], json!("user"));
        assert_eq!(messages[0]["content"], json!("before"));
        assert_eq!(messages[1]["role"], json!("tool"));
        assert_eq!(messages[1]["tool_call_id"], json!("c1"));
        assert_eq!(messages[2]["role"], json!("user"));
        assert_eq!(messages[2]["content"], json!("after"));
    }

    #[test]
    fn serializes_tools_array_without_tags() {
        let mut req = base_req();
        req.tools = Some(vec![ToolDefinition::new(
            "get_weather",
            "Look up the weather",
            json!({ "type": "object", "properties": {} }),
            vec![ToolTag::ReadOnly, ToolTag::Network],
        )]);
        let body = build_body("dep", &req);
        let tool = &body["tools"][0];
        assert_eq!(tool["type"], json!("function"));
        assert_eq!(tool["function"]["name"], json!("get_weather"));
        assert_eq!(
            tool["function"]["description"],
            json!("Look up the weather")
        );
        assert_eq!(
            tool["function"]["parameters"],
            json!({ "type": "object", "properties": {} })
        );
        // tags must NOT leak to the wire.
        assert!(tool.get("tags").is_none());
        assert!(tool["function"].get("tags").is_none());
    }

    #[test]
    fn omits_top_k_silently() {
        let mut req = base_req();
        req.top_k = Some(50);
        let body = build_body("dep", &req);
        assert!(body.get("top_k").is_none());
    }

    #[test]
    fn merges_additional_params_last() {
        let mut req = base_req();
        req.temperature = Some(0.2);
        req.additional_params = Some(json!({
            "temperature": 0.9,
            "logit_bias": { "100": -100 }
        }));
        let body = build_body("dep", &req);
        // additional_params overrides the canonical temperature.
        assert_eq!(body["temperature"], json!(0.9));
        assert_eq!(body["logit_bias"], json!({ "100": -100 }));
    }

    #[test]
    fn system_prompt_becomes_first_message() {
        let mut req = base_req();
        req.system_prompt = Some("be helpful".into());
        req.messages = vec![Message::user("hi")];
        let body = build_body("dep", &req);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], json!("system"));
        assert_eq!(messages[0]["content"], json!("be helpful"));
        assert_eq!(messages[1]["role"], json!("user"));
    }

    #[test]
    fn omits_optional_sampling_when_none() {
        let mut req = base_req();
        req.messages = vec![Message::user("hi")];
        let body = build_body("dep", &req);
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("stop").is_none());
    }

    #[test]
    fn omits_tool_choice_and_parallel_when_unset() {
        let body = build_body("dep", &base_req());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn maps_tool_choice_auto_as_string() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::Auto);
        let body = build_body("dep", &req);
        assert_eq!(body["tool_choice"], json!("auto"));
    }

    /// Azure / OpenAI Chat Completions calls Anthropic's `"any"`
    /// `"required"` — the adapter is the right place for that
    /// translation.
    #[test]
    fn maps_tool_choice_any_as_required() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::Any);
        let body = build_body("dep", &req);
        assert_eq!(body["tool_choice"], json!("required"));
    }

    #[test]
    fn maps_tool_choice_specific_tool_as_function_object() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".into(),
        });
        let body = build_body("dep", &req);
        assert_eq!(
            body["tool_choice"],
            json!({
                "type": "function",
                "function": { "name": "get_weather" },
            })
        );
    }

    #[test]
    fn maps_tool_choice_none_as_string() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::None_);
        let body = build_body("dep", &req);
        assert_eq!(body["tool_choice"], json!("none"));
    }

    /// Anthropic carries the flag inside `tool_choice`; Chat Completions
    /// exposes its inverse (`parallel_tool_calls`) as a top-level field.
    /// Setting `disable_parallel_tool_use = true` must produce
    /// `parallel_tool_calls = false`, independent of `tool_choice`.
    #[test]
    fn disable_parallel_emits_parallel_tool_calls_false() {
        let mut req = base_req();
        req.disable_parallel_tool_use = Some(true);
        let body = build_body("dep", &req);
        assert_eq!(body["parallel_tool_calls"], json!(false));
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn disable_parallel_false_emits_parallel_tool_calls_true() {
        let mut req = base_req();
        req.disable_parallel_tool_use = Some(false);
        let body = build_body("dep", &req);
        assert_eq!(body["parallel_tool_calls"], json!(true));
    }

    #[test]
    fn ignores_assistant_reasoning_block_silently() {
        let mut req = base_req();
        req.messages = vec![Message::Assistant {
            blocks: vec![
                AssistantBlock::Reasoning {
                    text: "thinking...".into(),
                    signature: Some("sig".into()),
                },
                AssistantBlock::text("answer"),
            ],
        }];
        let body = build_body("dep", &req);
        let msg = &body["messages"][0];
        assert_eq!(msg["role"], json!("assistant"));
        // Reasoning is dropped; only text survives.
        assert_eq!(msg["content"], json!("answer"));
        assert!(msg.get("tool_calls").is_none());
        assert!(msg.get("thinking").is_none());
        assert!(msg.get("reasoning").is_none());
    }
}
