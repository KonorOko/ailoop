//! Conversation message model: [`Message`], its block enums, and the
//! [`SystemPrompt`] / [`CacheControl`] support types.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Cache breakpoint placed on a content block, system prompt block, or
/// tool definition. Providers that support prompt caching (Anthropic
/// today) read these to decide which prefix is cacheable and at what
/// TTL; providers without explicit caching ignore the field. The
/// presence of `cache_control` only declares intent — the actual cache
/// hit/miss is reported via [`crate::Usage::cached_input_tokens`] and
/// the cache-creation counters.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheControl {
    /// Ephemeral cache with the provider's default TTL (5 minutes on
    /// Anthropic). Equivalent to omitting `ttl` on the wire.
    Ephemeral,
    /// Ephemeral cache with an explicit TTL. Anthropic accepts only
    /// `5m` and `1h`; the adapter rounds to the nearest supported value
    /// and warns if neither fits cleanly.
    EphemeralWithTtl(Duration),
}

/// One turn in the conversation history exchanged with a provider.
///
/// `Message` is the wire shape: every provider adapter maps this enum
/// to its own block model (Anthropic Messages, OpenAI Chat
/// Completions, etc.). Only the user and assistant roles live here —
/// system instructions are passed separately through
/// [`crate::ChatRequest::system_prompt`] because most providers
/// represent them out-of-band.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Message {
    /// A user-authored turn: free text, tool results from the previous
    /// step, or a mix.
    User {
        /// Blocks rendered in order. Tool results live here because, on
        /// the wire, they are sent back to the model as user content.
        blocks: Vec<UserBlock>,
    },
    /// A turn produced by the model: visible text, tool calls,
    /// reasoning. Tool results are in the *next* `User` turn.
    Assistant {
        /// Blocks rendered in order. Ordering is provider-significant
        /// for reasoning + tool-use chains (Anthropic extended
        /// thinking).
        blocks: Vec<AssistantBlock>,
    },
}

impl Message {
    /// Shorthand for a user turn containing a single text block.
    pub fn user(text: impl Into<String>) -> Message {
        Message::User {
            blocks: vec![UserBlock::text(text)],
        }
    }

    /// Shorthand for an assistant turn containing a single text block.
    /// Use the [`Message::Assistant`] variant directly when seeding
    /// history with tool calls or reasoning.
    pub fn assistant_text(text: impl Into<String>) -> Message {
        Message::Assistant {
            blocks: vec![AssistantBlock::text(text)],
        }
    }
}

/// One block inside a [`Message::User`] turn.
///
/// Free user text, tool results, and inline media (images, documents)
/// all live here because providers route tool results — and the rest
/// of the multimodal surface — back through the user role on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UserBlock {
    /// Free user text.
    Text {
        /// Text content.
        text: String,
        /// Per-request cache hint. `#[serde(skip)]` — the cache
        /// breakpoint is a per-call directive to the provider, not
        /// part of the persisted conversation state, so it is dropped
        /// on snapshot round-trip and restored as `None`.
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
    /// Result of a tool invocation paired with the assistant
    /// [`AssistantBlock::ToolCall`] of the previous turn (matched by
    /// id).
    ToolResult {
        /// Matches the `id` on the originating [`AssistantBlock::ToolCall`].
        call_id: String,
        /// The tool's reply. `content.is_error` flags tool-reported
        /// failures separately from the block list, so an error reply
        /// can still carry images.
        content: ToolResultContent,
        /// See [`UserBlock::Text::cache_control`].
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
    /// Image content rendered inline. Adapters map this to the
    /// provider's image content type (Anthropic `image`, Chat
    /// Completions `image_url`). Adapters that cannot represent the
    /// chosen [`Source`] (e.g. a Chat Completions deployment with no
    /// vision support) surface a typed error.
    Image {
        /// Image source: base64, URL, or provider-side file ID.
        source: Source,
        /// See [`UserBlock::Text::cache_control`].
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
    /// Document content rendered inline (PDF and similar). Anthropic
    /// has a dedicated `document` content type; Chat Completions does
    /// not, so the Azure adapter surfaces a typed
    /// `UnsupportedContent` error and callers downgrade via
    /// `ChatMiddleware::on_chat_request` if they want a text
    /// substitute.
    Document {
        /// Document source: base64, URL, or provider-side file ID.
        source: Source,
        /// See [`UserBlock::Text::cache_control`].
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
}

impl UserBlock {
    /// Build a [`UserBlock::Text`] with no cache breakpoint.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Build a [`UserBlock::ToolResult`] with no cache breakpoint. The
    /// `content` argument is anything that converts into
    /// [`ToolResultContent`] — `String` and `&str` produce a single
    /// text block with `is_error = false`. Use
    /// [`ToolResultContent::error`] to flag a tool-reported failure or
    /// [`ToolResultContent::from_blocks`] to build a multi-block reply.
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<ToolResultContent>) -> Self {
        Self::ToolResult {
            call_id: call_id.into(),
            content: content.into(),
            cache_control: None,
        }
    }

    /// Build a [`UserBlock::Image`] with no cache breakpoint.
    pub fn image(source: Source) -> Self {
        Self::Image {
            source,
            cache_control: None,
        }
    }

    /// Build a [`UserBlock::Document`] with no cache breakpoint.
    pub fn document(source: Source) -> Self {
        Self::Document {
            source,
            cache_control: None,
        }
    }

    /// Builder-style helper: set or replace the cache breakpoint on this
    /// block. Use `None` to clear.
    pub fn with_cache_control(mut self, cache_control: Option<CacheControl>) -> Self {
        match &mut self {
            Self::Text {
                cache_control: cc, ..
            }
            | Self::ToolResult {
                cache_control: cc, ..
            }
            | Self::Image {
                cache_control: cc, ..
            }
            | Self::Document {
                cache_control: cc, ..
            } => *cc = cache_control,
        }
        self
    }

    /// Read the current cache breakpoint, if any.
    pub fn cache_control(&self) -> Option<&CacheControl> {
        match self {
            Self::Text { cache_control, .. }
            | Self::ToolResult { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::Document { cache_control, .. } => cache_control.as_ref(),
        }
    }
}

/// One block inside a [`Message::Assistant`] turn.
///
/// Block ordering is preserved on replay because some providers
/// (Anthropic extended thinking) require the original sequence on
/// every subsequent request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AssistantBlock {
    /// Visible model-authored text.
    Text {
        /// Text content.
        text: String,
        /// See [`UserBlock::Text::cache_control`].
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
    /// A tool invocation request from the model. Pair with a
    /// [`UserBlock::ToolResult`] in the next user turn that matches
    /// `id` to `call_id`.
    ToolCall {
        /// Provider-assigned id; mirrors back as `call_id` on the
        /// matching [`UserBlock::ToolResult`].
        id: String,
        /// Tool name as registered in the [`crate::ChatRequest::tools`]
        /// list.
        name: String,
        /// JSON arguments. Adapters serialize this through to the
        /// provider verbatim; the engine does not validate the schema.
        args: serde_json::Value,
        /// See [`UserBlock::Text::cache_control`].
        #[serde(skip, default)]
        cache_control: Option<CacheControl>,
    },
    /// Visible reasoning emitted by the model. `signature` is provider-issued
    /// material that must be replayed verbatim on subsequent turns when tools
    /// are involved (Anthropic extended thinking). Providers without a
    /// signature concept (e.g. OpenAI reasoning) leave it `None`.
    ///
    /// Reasoning blocks intentionally have no `cache_control` slot:
    /// Anthropic does not accept the field on `thinking` /
    /// `redacted_thinking` blocks. Place breakpoints on adjacent text or
    /// tool blocks instead.
    Reasoning {
        /// Visible reasoning text.
        text: String,
        /// Provider signature (Anthropic extended thinking). Replay
        /// verbatim on subsequent turns when tools are involved;
        /// `None` for providers without a signature concept.
        signature: Option<String>,
    },
    /// Opaque reasoning block whose content the provider chose to hide.
    /// `data` is verbatim provider material — store it untouched and replay
    /// it back when the next request continues a tool-use chain.
    RedactedReasoning {
        /// Verbatim provider payload; treat as opaque bytes.
        data: String,
    },
}

impl AssistantBlock {
    /// Build an [`AssistantBlock::Text`] with no cache breakpoint.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Build an [`AssistantBlock::ToolCall`] with no cache breakpoint.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        args: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            args,
            cache_control: None,
        }
    }

    /// Builder-style helper: set or replace the cache breakpoint on this
    /// block. No-op for reasoning variants (they do not carry cache
    /// breakpoints on the wire).
    pub fn with_cache_control(mut self, cache_control: Option<CacheControl>) -> Self {
        match &mut self {
            Self::Text {
                cache_control: cc, ..
            } => *cc = cache_control,
            Self::ToolCall {
                cache_control: cc, ..
            } => *cc = cache_control,
            Self::Reasoning { .. } | Self::RedactedReasoning { .. } => {}
        }
        self
    }

    /// Read the current cache breakpoint. Always `None` for the
    /// reasoning variants (they do not carry breakpoints on the wire).
    pub fn cache_control(&self) -> Option<&CacheControl> {
        match self {
            Self::Text { cache_control, .. } | Self::ToolCall { cache_control, .. } => {
                cache_control.as_ref()
            }
            Self::Reasoning { .. } | Self::RedactedReasoning { .. } => None,
        }
    }
}

/// Source of an image or document content block.
///
/// Three forms cover the providers we ship adapters for today:
/// - `Base64` carries the binary inline. Always works but inflates the
///   request body and any persisted snapshot — prefer `Url` or
///   `FileId` for large media.
/// - `Url` points the provider at an external resource. Subject to
///   the provider's own fetch limits and accessibility rules.
/// - `FileId` references a provider-side file (Anthropic Files Beta,
///   OpenAI Files). Adapters that do not understand the id surface a
///   typed error rather than degrading silently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Source {
    /// Inline base64-encoded bytes.
    Base64 {
        /// MIME type of the payload (e.g. `image/png`, `application/pdf`).
        media_type: String,
        /// Base64-encoded data.
        data: String,
    },
    /// External URL the provider fetches.
    Url {
        /// HTTP(S) URL.
        url: String,
    },
    /// Provider-side file ID (Anthropic Files Beta, OpenAI Files). The
    /// id is opaque to the adapter — the provider resolves it on its
    /// side.
    FileId {
        /// Provider-issued identifier.
        id: String,
    },
}

/// One block inside a [`ToolResultContent::blocks`] list.
///
/// Several blocks can be interleaved inside a single tool reply
/// (e.g. text + a rendered chart), so a tool that generates an image
/// alongside an explanation does not have to choose. Error semantics
/// live on the parent [`ToolResultContent::is_error`], not per-block,
/// so a failed reply can still carry images.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    /// Plain text segment of the tool reply.
    Text {
        /// Text content.
        text: String,
    },
    /// Image segment of the tool reply. Adapters that cannot represent
    /// images inside tool results surface a typed error.
    Image {
        /// Image source.
        source: Source,
    },
}

impl ToolResultBlock {
    /// Build a [`ToolResultBlock::Text`].
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Build a [`ToolResultBlock::Image`] from the given source.
    pub fn image(source: Source) -> Self {
        Self::Image { source }
    }
}

/// Body of a tool reply sent back to the model in a
/// [`UserBlock::ToolResult`].
///
/// The body is a list of [`ToolResultBlock`]s (text, image, …) plus an
/// `is_error` flag. The flag is the wire-level error signal — Anthropic
/// emits it as `tool_result.is_error`; Chat Completions has no field
/// for it and treats the body as the error message.
///
/// `is_error` is **not** a Rust [`Result::Err`] — both forms represent
/// successful tool calls whose outcome the engine relays to the model.
/// `is_error = true` flags the reply as a failure the model should
/// account for (e.g. "the API returned 404"). Engine-level errors
/// (panic in the handler, arguments that don't deserialize, registry
/// lookup miss) are converted to a synthesized error reply so the
/// loop can continue; transport errors propagate through `Result`
/// channels instead.
///
/// Most callers build replies through the [`Self::text`] /
/// [`Self::error`] constructors; multi-block replies use
/// [`Self::from_blocks`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolResultContent {
    /// Content blocks in order. A "normal" text reply has a single
    /// [`ToolResultBlock::Text`] here.
    pub blocks: Vec<ToolResultBlock>,
    /// `true` flags the reply as a tool-reported failure (Anthropic
    /// `is_error: true`). Adapters that don't speak the flag on the
    /// wire just emit the text body.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl ToolResultContent {
    /// Build a successful text-only tool reply (`is_error = false`).
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![ToolResultBlock::text(text)],
            is_error: false,
        }
    }

    /// Build a failing text-only tool reply (`is_error = true`).
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![ToolResultBlock::text(text)],
            is_error: true,
        }
    }

    /// Build a successful image-only tool reply (`is_error = false`).
    pub fn image(source: Source) -> Self {
        Self {
            blocks: vec![ToolResultBlock::image(source)],
            is_error: false,
        }
    }

    /// Build a reply from arbitrary blocks. Defaults `is_error: false`;
    /// chain [`Self::with_is_error`] to flag failure.
    pub fn from_blocks(blocks: Vec<ToolResultBlock>) -> Self {
        Self {
            blocks,
            is_error: false,
        }
    }

    /// Builder-style helper: set the `is_error` flag.
    pub fn with_is_error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }

    /// First [`ToolResultBlock::Text`] body, if any. Useful when the
    /// caller only cares about the text portion of a reply.
    pub fn as_text(&self) -> Option<&str> {
        self.blocks.iter().find_map(|b| match b {
            ToolResultBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
    }

    /// Concatenate every [`ToolResultBlock::Text`] body in order,
    /// joined by newlines. Returns an empty string when there are no
    /// text blocks (the reply was image-only).
    pub fn collect_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                ToolResultBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl From<String> for ToolResultContent {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

impl From<&str> for ToolResultContent {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}

/// System prompt passed to the provider. `Plain(String)` matches the
/// pre-caching API and is what `From<String>` / `From<&str>` produce —
/// callers that don't care about prompt caching keep using strings.
/// `Blocks(...)` opts in to per-block cache breakpoints (Anthropic emits
/// the `system` field as an array; other providers concatenate the
/// block texts).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SystemPrompt {
    /// Single string passed to the provider as-is. The default for
    /// callers that don't care about prompt caching; matches `From<&str>`
    /// / `From<String>`.
    Plain(String),
    /// Sequence of blocks with optional per-block cache breakpoints.
    /// Anthropic emits this as the wire `system` array; providers
    /// without per-block caching concatenate the texts.
    Blocks(Vec<SystemBlock>),
}

/// One entry inside a [`SystemPrompt::Blocks`] sequence.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SystemBlock {
    /// Text content of this block.
    pub text: String,
    /// Optional cache breakpoint for this block.
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    /// Build a block with the given text and no cache breakpoint.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache_control: None,
        }
    }

    /// Builder-style helper: attach a cache breakpoint to this block.
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

impl From<String> for SystemPrompt {
    fn from(value: String) -> Self {
        Self::Plain(value)
    }
}

impl From<&str> for SystemPrompt {
    fn from(value: &str) -> Self {
        Self::Plain(value.to_string())
    }
}

impl SystemPrompt {
    /// Concatenate all blocks into a single string. Used by adapters
    /// without per-block caching (Chat Completions) and as a debug aid.
    pub fn as_text(&self) -> String {
        match self {
            Self::Plain(s) => s.clone(),
            Self::Blocks(bs) => bs
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn round_trip(msg: &Message) -> Message {
        let json = serde_json::to_string(msg).expect("serialize");
        serde_json::from_str(&json).expect("deserialize")
    }

    fn assert_user_text_eq(msg: &Message, expected: &str) {
        match msg {
            Message::User { blocks } => match &blocks[0] {
                UserBlock::Text {
                    text,
                    cache_control,
                } => {
                    assert_eq!(text, expected);
                    assert!(cache_control.is_none(), "cache_control must not round-trip");
                }
                other => panic!("expected UserBlock::Text, got {other:?}"),
            },
            other => panic!("expected Message::User, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_user_text_drops_cache_control() {
        let msg = Message::User {
            blocks: vec![
                UserBlock::text("hello").with_cache_control(Some(CacheControl::Ephemeral)),
            ],
        };
        let restored = round_trip(&msg);
        assert_user_text_eq(&restored, "hello");
    }

    #[test]
    fn round_trip_user_tool_result() {
        let msg = Message::User {
            blocks: vec![UserBlock::tool_result(
                "call-1",
                ToolResultContent::text("ok"),
            )],
        };
        let restored = round_trip(&msg);
        match &restored {
            Message::User { blocks } => match &blocks[0] {
                UserBlock::ToolResult {
                    call_id,
                    content,
                    cache_control,
                } => {
                    assert_eq!(call_id, "call-1");
                    assert_eq!(content.as_text(), Some("ok"));
                    assert!(!content.is_error);
                    assert!(cache_control.is_none());
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_assistant_text_and_tool_call() {
        let msg = Message::Assistant {
            blocks: vec![
                AssistantBlock::text("thinking out loud"),
                AssistantBlock::tool_call("c1", "fetch", json!({"q": "x"})),
            ],
        };
        let restored = round_trip(&msg);
        match &restored {
            Message::Assistant { blocks } => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    AssistantBlock::Text { text, .. } => assert_eq!(text, "thinking out loud"),
                    other => panic!("expected Text, got {other:?}"),
                }
                match &blocks[1] {
                    AssistantBlock::ToolCall { id, name, args, .. } => {
                        assert_eq!(id, "c1");
                        assert_eq!(name, "fetch");
                        assert_eq!(args, &json!({"q": "x"}));
                    }
                    other => panic!("expected ToolCall, got {other:?}"),
                }
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_assistant_reasoning_variants() {
        let msg = Message::Assistant {
            blocks: vec![
                AssistantBlock::Reasoning {
                    text: "consider X".into(),
                    signature: Some("sig-1".into()),
                },
                AssistantBlock::RedactedReasoning {
                    data: "opaque".into(),
                },
            ],
        };
        let restored = round_trip(&msg);
        match &restored {
            Message::Assistant { blocks } => match (&blocks[0], &blocks[1]) {
                (
                    AssistantBlock::Reasoning { text, signature },
                    AssistantBlock::RedactedReasoning { data },
                ) => {
                    assert_eq!(text, "consider X");
                    assert_eq!(signature.as_deref(), Some("sig-1"));
                    assert_eq!(data, "opaque");
                }
                other => panic!("unexpected blocks: {other:?}"),
            },
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_tool_result_error_variant() {
        let content = ToolResultContent::error("boom");
        let json = serde_json::to_string(&content).unwrap();
        let back: ToolResultContent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.as_text(), Some("boom"));
        assert!(back.is_error);
    }

    #[test]
    fn tool_result_content_is_error_omitted_when_false() {
        let content = ToolResultContent::text("ok");
        let json = serde_json::to_value(&content).unwrap();
        assert!(
            json.get("is_error").is_none(),
            "is_error must be skipped when false, got {json}"
        );
    }

    #[test]
    fn tool_result_content_multi_block_round_trip() {
        let content = ToolResultContent::from_blocks(vec![
            ToolResultBlock::text("see chart"),
            ToolResultBlock::image(Source::Url {
                url: "https://example.com/chart.png".into(),
            }),
        ]);
        let json = serde_json::to_string(&content).unwrap();
        let back: ToolResultContent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.blocks.len(), 2);
        assert!(!back.is_error);
    }

    #[test]
    fn round_trip_user_image_block() {
        let msg = Message::User {
            blocks: vec![UserBlock::image(Source::Base64 {
                media_type: "image/png".into(),
                data: "AAAA".into(),
            })],
        };
        let restored = round_trip(&msg);
        match &restored {
            Message::User { blocks } => match &blocks[0] {
                UserBlock::Image {
                    source,
                    cache_control,
                } => {
                    assert!(matches!(
                        source,
                        Source::Base64 { media_type, data }
                            if media_type == "image/png" && data == "AAAA"
                    ));
                    assert!(cache_control.is_none());
                }
                other => panic!("expected Image, got {other:?}"),
            },
            other => panic!("expected User, got {other:?}"),
        }
    }
}
