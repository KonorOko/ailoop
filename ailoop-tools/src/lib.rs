//! Tool registry and trait surface for ailoop.
//!
//! Two parallel traits cover the two ways tools come into existence:
//!
//! - [`Tool`] — typed, compile-time tools. Derive
//!   [`#[ailoop_tool]`](https://docs.rs/ailoop) on a Rust function and
//!   register the resulting type with
//!   [`ConversationBuilder::tool`](https://docs.rs/ailoop). Args are
//!   typed, deserialized from the model's JSON, and the return value is
//!   serialized back out.
//! - [`ToolDyn`] — object-safe sibling of [`Tool`]. Same semantics, but
//!   reachable behind `Arc<dyn ToolDyn>` for plugins, MCP servers, and
//!   anything else where the tool list is built at runtime. Register
//!   with
//!   [`ConversationBuilder::tool_dyn`](https://docs.rs/ailoop). The
//!   blanket `impl<T: Tool> ToolDyn for T` means every static tool is
//!   automatically a dynamic one too.
//!
//! [`ToolRegistry`] owns the active set and dispatches model-issued
//! tool calls. The façade builds one inside its
//! [`Conversation`](https://docs.rs/ailoop) and exposes its surface
//! via builder methods (capabilities, approval gating, runtime
//! activate/deactivate); plug it in directly only when driving
//! [`advanced::run_chat`](https://docs.rs/ailoop).
//!
//! ## Mini-index
//!
//! - [`Tool`], [`ToolDyn`] — the two trait shapes.
//! - [`ToolRegistry`] — registration + activation + dispatch.
//! - [`ToolJsonType`] — per-type JSON Schema fragments. The
//!   [`#[ailoop_tool]`](https://docs.rs/ailoop) macro falls back to
//!   this trait for unknown parameter types; derive it with
//!   [`#[derive(ToolJsonType)]`](https://docs.rs/ailoop) on C-style
//!   enums.
//! - [`ToolRegistryError`] — failure surface of registry mutations.

#![deny(missing_docs)]

pub mod errors;
pub mod registry;
pub mod schema;

pub use errors::ToolRegistryError;
pub use registry::{Tool, ToolDyn, ToolRegistry};
pub use schema::ToolJsonType;
