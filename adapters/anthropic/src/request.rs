use std::time::Duration;

use ailoop_core::{
    AssistantBlock, CacheControl, ChatRequest, Message, ReasoningEffort, Source, SystemBlock,
    SystemPrompt, ToolChoice, ToolDefinition, ToolResultBlock, UserBlock,
};
use serde_json::{Value, json};

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
        body.insert("system".into(), to_anthropic_system(system));
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

    if let Some(effort) = req.reasoning_effort
        && let Some(thinking) = to_anthropic_thinking(effort)
    {
        body.insert("thinking".into(), thinking);
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

/// Encode an [`ailoop_core::CacheControl`] into Anthropic's wire object.
/// Anthropic accepts only `5m` and `1h` as `ttl`; for any other duration
/// we round to the closer of the two and omit the field if the caller
/// asked for the default (`Ephemeral` without a TTL).
fn cache_control_value(cc: &CacheControl) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("ephemeral"));
    if let CacheControl::EphemeralWithTtl(d) = cc {
        let ttl = ttl_string_for(*d);
        obj.insert("ttl".into(), json!(ttl));
    }
    Value::Object(obj)
}

fn ttl_string_for(d: Duration) -> &'static str {
    // Anthropic supports exactly two TTL strings; round to the closer.
    let secs = d.as_secs();
    let five_min = 5 * 60;
    let one_hour = 60 * 60;
    if secs.abs_diff(five_min) <= secs.abs_diff(one_hour) {
        "5m"
    } else {
        "1h"
    }
}

fn insert_cache_control(obj: &mut serde_json::Map<String, Value>, cc: Option<&CacheControl>) {
    if let Some(cc) = cc {
        obj.insert("cache_control".into(), cache_control_value(cc));
    }
}

fn to_anthropic_system(prompt: &SystemPrompt) -> Value {
    match prompt {
        // Plain string — wire-compatible with the pre-caching shape so
        // existing fixtures and zero-cache-control callers see no diff.
        SystemPrompt::Plain(s) => json!(s),
        SystemPrompt::Blocks(blocks) => {
            json!(
                blocks
                    .iter()
                    .map(to_anthropic_system_block)
                    .collect::<Vec<_>>()
            )
        }
        // SystemPrompt is `#[non_exhaustive]`; future variants degrade
        // to an empty system on the wire rather than failing the build.
        _ => json!(""),
    }
}

fn to_anthropic_system_block(block: &SystemBlock) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("text"));
    obj.insert("text".into(), json!(block.text));
    insert_cache_control(&mut obj, block.cache_control.as_ref());
    Value::Object(obj)
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
        _ => json!({ "role": "user", "content": [] }),
    }
}

fn to_anthropic_user_block(block: &UserBlock) -> serde_json::Value {
    match block {
        UserBlock::Text {
            text,
            cache_control,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("text"));
            obj.insert("text".into(), json!(text));
            insert_cache_control(&mut obj, cache_control.as_ref());
            Value::Object(obj)
        }
        UserBlock::ToolResult {
            call_id,
            content,
            cache_control,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("tool_result"));
            obj.insert("tool_use_id".into(), json!(call_id));
            obj.insert(
                "content".into(),
                json!(
                    content
                        .blocks
                        .iter()
                        .map(to_anthropic_tool_result_block)
                        .collect::<Vec<_>>()
                ),
            );
            if content.is_error {
                obj.insert("is_error".into(), json!(true));
            }
            insert_cache_control(&mut obj, cache_control.as_ref());
            Value::Object(obj)
        }
        UserBlock::Image {
            source,
            cache_control,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("image"));
            obj.insert("source".into(), to_anthropic_source(source));
            insert_cache_control(&mut obj, cache_control.as_ref());
            Value::Object(obj)
        }
        UserBlock::Document {
            source,
            cache_control,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("document"));
            obj.insert("source".into(), to_anthropic_source(source));
            insert_cache_control(&mut obj, cache_control.as_ref());
            Value::Object(obj)
        }
        // UserBlock is `#[non_exhaustive]`; future variants are dropped
        // until the adapter learns to translate them.
        _ => Value::Null,
    }
}

/// Encode a [`Source`] into Anthropic's content-block `source` object.
/// Anthropic uses different wire `type` values per form
/// (`base64`, `url`, `file`).
fn to_anthropic_source(source: &Source) -> Value {
    match source {
        Source::Base64 { media_type, data } => json!({
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }),
        Source::Url { url } => json!({
            "type": "url",
            "url": url,
        }),
        Source::FileId { id } => json!({
            "type": "file",
            "file_id": id,
        }),
        // Source is `#[non_exhaustive]`; future variants are dropped
        // until the adapter learns to translate them.
        _ => Value::Null,
    }
}

/// Encode one [`ToolResultBlock`] into the matching entry inside
/// `tool_result.content[]`. Anthropic accepts the same `text` /
/// `image` shapes here as in a top-level user message.
fn to_anthropic_tool_result_block(block: &ToolResultBlock) -> Value {
    match block {
        ToolResultBlock::Text { text } => json!({
            "type": "text",
            "text": text,
        }),
        ToolResultBlock::Image { source } => json!({
            "type": "image",
            "source": to_anthropic_source(source),
        }),
        // ToolResultBlock is `#[non_exhaustive]`; future variants are
        // dropped until the adapter learns to translate them.
        _ => Value::Null,
    }
}

fn to_anthropic_assistant_block(block: &AssistantBlock) -> serde_json::Value {
    match block {
        AssistantBlock::Text {
            text,
            cache_control,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("text"));
            obj.insert("text".into(), json!(text));
            insert_cache_control(&mut obj, cache_control.as_ref());
            Value::Object(obj)
        }
        AssistantBlock::ToolCall {
            id,
            name,
            args,
            cache_control,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("tool_use"));
            obj.insert("id".into(), json!(id));
            obj.insert("name".into(), json!(name));
            obj.insert("input".into(), args.clone());
            insert_cache_control(&mut obj, cache_control.as_ref());
            Value::Object(obj)
        }
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
        // AssistantBlock is `#[non_exhaustive]`; future variants are
        // dropped until the adapter learns to translate them.
        _ => Value::Null,
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
        // ToolChoice is `#[non_exhaustive]`; future variants degrade to
        // `auto` so the request does not error out at adapter level.
        _ => {
            obj.insert("type".into(), json!("auto"));
        }
    }
    if let Some(flag) = disable_parallel {
        obj.insert("disable_parallel_tool_use".into(), json!(flag));
    }
    serde_json::Value::Object(obj)
}

/// Lower a [`ReasoningEffort`] into Anthropic's `thinking` object.
/// `Minimal` emits explicit `{"type":"disabled"}` so the wire is
/// unambiguous; the other variants map to `{"type":"enabled","budget_tokens":N}`
/// using the budgets documented on [`ReasoningEffort`]. Returns `None`
/// for unknown future variants so the field is dropped from the body
/// rather than emitted as `null`.
fn to_anthropic_thinking(effort: ReasoningEffort) -> Option<Value> {
    match effort {
        ReasoningEffort::Minimal => Some(json!({ "type": "disabled" })),
        ReasoningEffort::Low => Some(json!({ "type": "enabled", "budget_tokens": 1024 })),
        ReasoningEffort::Medium => Some(json!({ "type": "enabled", "budget_tokens": 4096 })),
        ReasoningEffort::High => Some(json!({ "type": "enabled", "budget_tokens": 16384 })),
        ReasoningEffort::Budget(n) => Some(json!({ "type": "enabled", "budget_tokens": n })),
        // Future variants degrade to no `thinking` field on the wire so
        // the request still validates instead of failing the build.
        _ => None,
    }
}

fn to_anthropic_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            let mut obj = serde_json::Map::new();
            obj.insert("name".into(), json!(t.name));
            obj.insert("description".into(), json!(t.description));
            obj.insert("input_schema".into(), t.input_schema.clone());
            insert_cache_control(&mut obj, t.cache_control.as_ref());
            Value::Object(obj)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::ChatRequest;

    fn base_req() -> ChatRequest {
        ChatRequest::new(vec![], 1024)
    }

    #[test]
    fn omits_tool_choice_when_unset() {
        let body = build_body("claude", &base_req());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn maps_tool_choice_auto() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::Auto);
        let body = build_body("claude", &req);
        assert_eq!(body["tool_choice"], json!({ "type": "auto" }));
    }

    #[test]
    fn maps_tool_choice_any() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::Any);
        let body = build_body("claude", &req);
        assert_eq!(body["tool_choice"], json!({ "type": "any" }));
    }

    #[test]
    fn maps_tool_choice_specific_tool() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".into(),
        });
        let body = build_body("claude", &req);
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "tool", "name": "get_weather" })
        );
    }

    #[test]
    fn maps_tool_choice_none() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::None_);
        let body = build_body("claude", &req);
        assert_eq!(body["tool_choice"], json!({ "type": "none" }));
    }

    /// `disable_parallel_tool_use` lives inside `tool_choice` on
    /// Anthropic, so setting it without an explicit choice must still
    /// emit a tool_choice object — defaulting to `auto`.
    #[test]
    fn disable_parallel_alone_emits_auto_tool_choice_with_flag() {
        let mut req = base_req();
        req.disable_parallel_tool_use = Some(true);
        let body = build_body("claude", &req);
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "auto", "disable_parallel_tool_use": true })
        );
    }

    #[test]
    fn disable_parallel_combines_with_explicit_tool_choice() {
        let mut req = base_req();
        req.tool_choice = Some(ToolChoice::Any);
        req.disable_parallel_tool_use = Some(true);
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

    /// Plain system prompts must continue to wire as a string so the
    /// pre-caching call shape is preserved for callers that did not opt
    /// in to cache breakpoints.
    #[test]
    fn plain_system_prompt_wires_as_string() {
        let mut req = base_req();
        req.system_prompt = Some(SystemPrompt::from("be helpful"));
        let body = build_body("claude", &req);
        assert_eq!(body["system"], json!("be helpful"));
    }

    /// A multi-block system prompt must wire as an array of typed blocks
    /// with `cache_control` only on blocks that asked for one.
    #[test]
    fn block_system_prompt_emits_cache_control_per_block() {
        let mut req = base_req();
        req.system_prompt = Some(SystemPrompt::Blocks(vec![
            SystemBlock::new("static prelude").with_cache_control(CacheControl::Ephemeral),
            SystemBlock::new("dynamic suffix"),
        ]));
        let body = build_body("claude", &req);
        assert_eq!(
            body["system"],
            json!([
                {
                    "type": "text",
                    "text": "static prelude",
                    "cache_control": { "type": "ephemeral" },
                },
                {
                    "type": "text",
                    "text": "dynamic suffix",
                },
            ])
        );
    }

    /// `EphemeralWithTtl` must surface `ttl: "1h"` on the wire when the
    /// caller asked for the long TTL.
    #[test]
    fn ephemeral_with_one_hour_ttl_emits_ttl_string() {
        let block = UserBlock::text("cached").with_cache_control(Some(
            CacheControl::EphemeralWithTtl(Duration::from_secs(60 * 60)),
        ));
        let json = to_anthropic_user_block(&block);
        assert_eq!(
            json,
            json!({
                "type": "text",
                "text": "cached",
                "cache_control": { "type": "ephemeral", "ttl": "1h" },
            })
        );
    }

    #[test]
    fn user_text_without_cache_control_omits_field() {
        let block = UserBlock::text("plain");
        let json = to_anthropic_user_block(&block);
        assert!(json.get("cache_control").is_none());
    }

    #[test]
    fn tool_result_carries_cache_control() {
        use ailoop_core::ToolResultContent;
        let block = UserBlock::tool_result("call_1", ToolResultContent::text("ok"))
            .with_cache_control(Some(CacheControl::Ephemeral));
        let json = to_anthropic_user_block(&block);
        assert_eq!(
            json["cache_control"],
            json!({ "type": "ephemeral" }),
            "tool_result blocks must support cache_control",
        );
    }

    #[test]
    fn tool_result_text_wires_as_array_of_text_blocks() {
        use ailoop_core::ToolResultContent;
        let block = UserBlock::tool_result("call_1", ToolResultContent::text("70F"));
        let json = to_anthropic_user_block(&block);
        assert_eq!(json["type"], json!("tool_result"));
        assert_eq!(json["tool_use_id"], json!("call_1"));
        assert_eq!(
            json["content"],
            json!([{ "type": "text", "text": "70F" }]),
            "tool_result content must always be an array of typed blocks",
        );
        // is_error is omitted when false to match Anthropic's default.
        assert!(json.get("is_error").is_none());
    }

    #[test]
    fn tool_result_error_emits_is_error_true() {
        use ailoop_core::ToolResultContent;
        let block = UserBlock::tool_result("call_1", ToolResultContent::error("API down"));
        let json = to_anthropic_user_block(&block);
        assert_eq!(json["is_error"], json!(true));
        assert_eq!(
            json["content"],
            json!([{ "type": "text", "text": "API down" }]),
        );
    }

    #[test]
    fn tool_result_with_image_wires_mixed_blocks() {
        use ailoop_core::{Source, ToolResultBlock, ToolResultContent};
        let content = ToolResultContent::from_blocks(vec![
            ToolResultBlock::text("see chart"),
            ToolResultBlock::image(Source::Base64 {
                media_type: "image/png".into(),
                data: "AAAA".into(),
            }),
        ]);
        let block = UserBlock::tool_result("call_1", content);
        let json = to_anthropic_user_block(&block);
        assert_eq!(
            json["content"],
            json!([
                { "type": "text", "text": "see chart" },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA",
                    },
                },
            ])
        );
    }

    #[test]
    fn user_image_block_maps_to_image_with_base64_source() {
        use ailoop_core::Source;
        let block = UserBlock::image(Source::Base64 {
            media_type: "image/png".into(),
            data: "iVBOR".into(),
        });
        let json = to_anthropic_user_block(&block);
        assert_eq!(
            json,
            json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBOR",
                },
            })
        );
    }

    #[test]
    fn user_image_block_maps_to_image_with_url_source() {
        use ailoop_core::Source;
        let block = UserBlock::image(Source::Url {
            url: "https://example.com/img.png".into(),
        });
        let json = to_anthropic_user_block(&block);
        assert_eq!(
            json,
            json!({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/img.png",
                },
            })
        );
    }

    #[test]
    fn user_image_block_maps_to_image_with_file_id_source() {
        use ailoop_core::Source;
        let block = UserBlock::image(Source::FileId {
            id: "file_abc".into(),
        });
        let json = to_anthropic_user_block(&block);
        assert_eq!(
            json,
            json!({
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": "file_abc",
                },
            })
        );
    }

    #[test]
    fn user_document_block_maps_to_document_with_base64_source() {
        use ailoop_core::Source;
        let block = UserBlock::document(Source::Base64 {
            media_type: "application/pdf".into(),
            data: "JVBERi0".into(),
        });
        let json = to_anthropic_user_block(&block);
        assert_eq!(json["type"], json!("document"));
        assert_eq!(
            json["source"],
            json!({
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0",
            })
        );
    }

    #[test]
    fn image_block_carries_cache_control() {
        use ailoop_core::Source;
        let block = UserBlock::image(Source::Url {
            url: "https://example.com/img.png".into(),
        })
        .with_cache_control(Some(CacheControl::Ephemeral));
        let json = to_anthropic_user_block(&block);
        assert_eq!(json["cache_control"], json!({ "type": "ephemeral" }));
    }

    #[test]
    fn assistant_text_carries_cache_control() {
        let block =
            AssistantBlock::text("answer").with_cache_control(Some(CacheControl::Ephemeral));
        let json = to_anthropic_assistant_block(&block);
        assert_eq!(json["cache_control"], json!({ "type": "ephemeral" }));
    }

    #[test]
    fn assistant_tool_call_carries_cache_control() {
        let block = AssistantBlock::tool_call("toolu_1", "search", json!({"q": "x"}))
            .with_cache_control(Some(CacheControl::Ephemeral));
        let json = to_anthropic_assistant_block(&block);
        assert_eq!(json["cache_control"], json!({ "type": "ephemeral" }));
    }

    /// Tool definitions can carry their own cache breakpoint so the
    /// model sees a cacheable tool prefix when enough tools share an
    /// identical (or stable-prefix) schema.
    #[test]
    fn tool_definition_emits_cache_control_when_set() {
        let mut req = base_req();
        req.tools = Some(vec![
            ToolDefinition::new(
                "get_weather",
                "Look up the weather",
                json!({ "type": "object", "properties": {} }),
                vec![],
            )
            .with_cache_control(CacheControl::Ephemeral),
        ]);
        let body = build_body("claude", &req);
        let tool = &body["tools"][0];
        assert_eq!(tool["cache_control"], json!({ "type": "ephemeral" }));
    }

    /// Categorical variants map to `thinking.budget_tokens` using the
    /// budgets documented on `ReasoningEffort`.
    #[test]
    fn reasoning_effort_low_medium_high_emit_documented_budgets() {
        for (variant, expected_budget) in [
            (ReasoningEffort::Low, 1024),
            (ReasoningEffort::Medium, 4096),
            (ReasoningEffort::High, 16384),
        ] {
            let mut req = base_req();
            req.reasoning_effort = Some(variant);
            let body = build_body("claude", &req);
            assert_eq!(
                body["thinking"],
                json!({ "type": "enabled", "budget_tokens": expected_budget }),
                "{variant:?} should map to budget {expected_budget}"
            );
        }
    }

    /// `Minimal` emits explicit `{"type":"disabled"}` so the model is
    /// told *not* to think — distinct from `None` which omits the field.
    #[test]
    fn reasoning_effort_minimal_emits_disabled() {
        let mut req = base_req();
        req.reasoning_effort = Some(ReasoningEffort::Minimal);
        let body = build_body("claude", &req);
        assert_eq!(body["thinking"], json!({ "type": "disabled" }));
    }

    /// `Budget(n)` forwards the exact budget verbatim — including
    /// out-of-spec values, so the API returns a caller-actionable error
    /// rather than the adapter silently rounding.
    #[test]
    fn reasoning_effort_budget_forwards_verbatim() {
        let mut req = base_req();
        req.reasoning_effort = Some(ReasoningEffort::Budget(7777));
        let body = build_body("claude", &req);
        assert_eq!(
            body["thinking"],
            json!({ "type": "enabled", "budget_tokens": 7777 })
        );
    }

    /// `None` (the default) omits the `thinking` field entirely so the
    /// wire stays compatible with non-thinking deployments.
    #[test]
    fn reasoning_effort_none_omits_thinking_field() {
        let body = build_body("claude", &base_req());
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn tool_definition_omits_cache_control_when_unset() {
        let mut req = base_req();
        req.tools = Some(vec![ToolDefinition::new(
            "get_weather",
            "Look up the weather",
            json!({ "type": "object", "properties": {} }),
            vec![],
        )]);
        let body = build_body("claude", &req);
        let tool = &body["tools"][0];
        assert!(tool.get("cache_control").is_none());
    }
}
