//! Failure surface of [`ToolRegistry`](crate::ToolRegistry) mutations
//! and lookups.

use thiserror::Error;

/// Errors surfaced by [`ToolRegistry`](crate::ToolRegistry) operations.
///
/// The engine treats [`Self::NotFound`] in-band — when the model calls
/// a tool that is not registered, the engine synthesizes a
/// [`ToolResultContent::Error`] reply rather than aborting the run, so
/// the model can recover by trying a different tool. Every other
/// variant escalates as
/// [`EngineError::Tool`](https://docs.rs/ailoop) (or [`BuildError::ToolRegistry`](https://docs.rs/ailoop)
/// during builder setup) since they indicate a configuration bug
/// rather than a model recoverable state.
///
/// [`ToolResultContent::Error`]: ailoop_core::ToolResultContent::Error
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolRegistryError {
    /// No tool with the requested name is registered. Returned by
    /// [`ToolRegistry::tool_call`](crate::ToolRegistry::tool_call) and
    /// [`ToolRegistry::activate_tool`](crate::ToolRegistry::activate_tool).
    #[error("Tool '{0}' not found")]
    NotFound(String),

    /// A second
    /// [`register`](crate::ToolRegistry::register) call was issued for
    /// a tool whose name is already in the registry. Names are
    /// globally unique within one registry; collisions across
    /// `tool(...)` / `tool_dyn(...)` calls (or across MCP servers
    /// after sanitization) surface here at builder time.
    #[error("Tool {0} already registered")]
    AlreadyRegistered(String),
}
