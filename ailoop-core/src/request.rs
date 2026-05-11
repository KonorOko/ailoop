//! Per-turn request shape: [`ChatRequest`] plus [`ToolChoice`],
//! [`ToolDefinition`], and [`ToolTag`].

use serde::Serialize;
use serde_json::Value;

use crate::{CacheControl, Message, SystemPrompt};

/// Per-turn request the engine assembles and the adapter lowers to
/// the provider's wire format.
///
/// The struct is `#[non_exhaustive]`, so external construction goes
/// through [`Default`] / [`ChatRequest::new`] rather than struct
/// literals. Adapters that don't natively support a control (e.g.
/// `top_k` on Chat Completions) ignore it; provider-specific knobs
/// outside the typed surface ride [`Self::additional_params`].
#[derive(Clone)]
#[non_exhaustive]
pub struct ChatRequest {
    /// Conversation history sent to the provider, in order. Adapters
    /// translate each [`Message`] block to the provider's content
    /// shape.
    pub messages: Vec<Message>,
    /// System instructions sent out-of-band from `messages`. `None`
    /// lets the provider use its default behaviour.
    pub system_prompt: Option<SystemPrompt>,
    /// Tools the model may call this turn. `None` disables tool use
    /// entirely and lets adapters omit the `tools` field on the wire,
    /// which some providers require when no tools are configured.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Sampling temperature in the provider's native range
    /// (typically `[0.0, 1.0]` or `[0.0, 2.0]`). `None` keeps the
    /// provider default.
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff. `None` keeps the provider default.
    pub top_p: Option<f32>,
    /// Top-k sampling cutoff. Only honoured by providers that expose
    /// the parameter (Anthropic). `None` keeps the provider default;
    /// adapters without `top_k` ignore the field.
    pub top_k: Option<u32>,
    /// Strings that, if produced, terminate the response with
    /// [`crate::FinishReason::StopSequence`].
    pub stop_sequences: Vec<String>,
    /// Maximum output tokens the provider should generate this turn.
    pub max_tokens: u32,
    /// How the model should pick among the available tools. `None`
    /// leaves the provider default (typically `Auto`). Mapped to each
    /// provider's wire format by the adapter.
    pub tool_choice: Option<ToolChoice>,
    /// Forbid the model from emitting more than one `tool_use` block
    /// per turn. `None` leaves the provider default (parallel allowed).
    /// Adapters lower this to the field their API expects:
    /// `tool_choice.disable_parallel_tool_use` for Anthropic,
    /// `parallel_tool_calls` (negated) for Chat Completions.
    pub disable_parallel_tool_use: Option<bool>,

    /// How hard the model should think before answering. `None` keeps
    /// the provider default (no thinking on Anthropic; the deployment's
    /// configured default on OpenAI). Adapters translate to their
    /// native shape — see [`ReasoningEffort`] for the per-variant
    /// mapping. Providers that do not surface a reasoning control
    /// ignore the field.
    pub reasoning_effort: Option<ReasoningEffort>,

    /// Free-form JSON merged into the provider's request body.
    /// Escape hatch for provider-specific knobs not yet typed in the
    /// shared shape (Anthropic `thinking`, OpenAI `reasoning_effort`,
    /// future betas). Adapters merge this last so it can override
    /// any field the typed surface set.
    pub additional_params: Option<Value>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            system_prompt: None,
            tools: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            max_tokens: 4096,
            tool_choice: None,
            disable_parallel_tool_use: None,
            reasoning_effort: None,
            additional_params: None,
        }
    }
}

impl ChatRequest {
    /// Build a request with the given history and token cap; every
    /// other field takes its [`Default`] value. Use `..Default::default()`
    /// in struct-update form when more than these two fields are set.
    pub fn new(messages: Vec<Message>, max_tokens: u32) -> Self {
        Self {
            messages,
            max_tokens,
            ..Default::default()
        }
    }
}

/// Constraint placed on the model's tool selection for a single
/// request. Variant naming follows Anthropic's wire vocabulary; the
/// Chat Completions adapter translates `Any` → `"required"` and
/// `None_` → `"none"`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolChoice {
    /// Model decides whether to call a tool. Provider default.
    Auto,
    /// Model must call **some** tool, but may pick which one.
    /// Translates to `"required"` on Chat Completions.
    Any,
    /// Model must call this specific tool.
    Tool {
        /// Name of the required tool; must appear in the request's
        /// `tools` list.
        name: String,
    },
    /// Model is forbidden from calling any tool. Trailing underscore
    /// avoids collision with the keyword `None`.
    None_,
}

/// Tool description sent to the provider so the model can decide when
/// and how to call it.
///
/// `ToolDefinition` is the wire-level shape; tool *implementations*
/// live behind the `ailoop-tools` `Tool` / `ToolDyn` traits and only
/// produce a `ToolDefinition` when they are mounted on a
/// [`ChatRequest`].
#[derive(Debug, Serialize, Clone)]
#[non_exhaustive]
pub struct ToolDefinition {
    /// Tool name; must match the `name` the model emits on
    /// [`crate::AssistantBlock::ToolCall`].
    pub name: String,
    /// Human-readable description used by the model to decide when to
    /// invoke the tool.
    pub description: String,
    /// JSON Schema (2020-12) for the tool's input arguments.
    pub input_schema: serde_json::Value,
    /// Tags consumed by capability gating and approval middleware.
    pub tags: Vec<ToolTag>,
    /// Cache breakpoint for this tool entry on providers that support
    /// per-tool prompt caching (Anthropic). Adapters without per-tool
    /// caching ignore the field.
    #[serde(skip)]
    pub cache_control: Option<CacheControl>,
}

impl ToolDefinition {
    /// Build a definition with the given identity, schema, and tags
    /// and no cache breakpoint.
    pub fn new(
        name: &str,
        description: &str,
        input_schema: serde_json::Value,
        tags: Vec<ToolTag>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            tags,
            cache_control: None,
        }
    }

    /// Builder-style helper: attach a cache breakpoint to this tool
    /// entry.
    pub fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }
}

/// Cross-provider knob for how hard the model should think before
/// answering.
///
/// OpenAI/Azure exposes this as a categorical `reasoning_effort` string
/// (`"minimal" | "low" | "medium" | "high"`); Anthropic exposes a
/// `thinking.budget_tokens` integer. The variants below cover both
/// shapes, with the per-adapter mapping documented inline so the
/// translation is predictable rather than magic:
///
/// | Variant       | OpenAI / Azure          | Anthropic (`budget_tokens`) |
/// |---------------|-------------------------|------------------------------|
/// | `Minimal`     | `"minimal"`             | `thinking.type = "disabled"` |
/// | `Low`         | `"low"`                 | `1024` (Anthropic minimum)   |
/// | `Medium`      | `"medium"`              | `4096`                       |
/// | `High`        | `"high"`                | `16384`                      |
/// | `Budget(n)`   | `< 2048 → "low"`,<br/>`< 8192 → "medium"`,<br/>`≥ 8192 → "high"` | `n` (verbatim) |
///
/// Use `Budget(n)` when you want exact control over Anthropic's
/// `budget_tokens`. The OpenAI side then bucketises into the closest
/// categorical effort — a deliberately opinionated mapping so the same
/// `ChatRequest` works against either adapter without falling back to
/// [`ChatRequest::additional_params`]. If you need a budget outside
/// the bucketing, the escape hatch is still available.
///
/// Providers that do not surface a reasoning control ignore the field
/// entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningEffort {
    /// OpenAI: `"minimal"`. Anthropic: explicit `thinking.type =
    /// "disabled"` on the wire.
    Minimal,
    /// OpenAI: `"low"`. Anthropic: `budget_tokens = 1024` (the API
    /// minimum for enabled thinking).
    Low,
    /// OpenAI: `"medium"`. Anthropic: `budget_tokens = 4096`.
    Medium,
    /// OpenAI: `"high"`. Anthropic: `budget_tokens = 16384`.
    High,
    /// Anthropic-native exact budget. OpenAI bucketises: values below
    /// `2048` map to `"low"`, below `8192` to `"medium"`, otherwise to
    /// `"high"`. Anthropic requires at least `1024`; values below that
    /// are still forwarded verbatim so the API returns a
    /// caller-actionable error rather than silently rounding.
    Budget(u32),
}

/// Capability tag attached to a [`ToolDefinition`].
///
/// Tags are the input to the façade's capability filter
/// (`ConversationBuilder::with_capabilities`) and to the built-in
/// approval middleware. The `ailoop_derive::ailoop_tool` proc-macro
/// emits these via the `tags(...)` argument.
#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolTag {
    /// Tool mutates state outside the conversation (delete files,
    /// send a payment, ...). The default approval middleware gates
    /// these by default.
    Destructive,
    /// Tool only reads state; safe to run unattended.
    ReadOnly,
    /// Tool reaches the network. Useful for sandboxed deployments.
    Network,
    /// Tool writes to the local filesystem. Approval-gated by the
    /// default middleware alongside [`Self::Destructive`].
    WritesFiles,
    /// Free-form tag for caller-defined capability schemes.
    Custom(String),
}
