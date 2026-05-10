//! Run and step identifiers used by the engine to correlate
//! [`crate::StreamChunk`]s and [`crate::ChatMiddleware`] hooks.

use std::fmt;

use uuid::Uuid;

/// Unique identifier for a single run of the engine.
///
/// Carried on every [`crate::StreamChunk`] variant the engine emits and on
/// every [`crate::ChatMiddleware`] hook so observability code can correlate
/// events across concurrent runs. The engine mints a fresh
/// [`Uuid::new_v4`] when [`crate::RunConfig::run_id`] is `None`; callers
/// pass an existing `RunId` when an outer system already has its own
/// trace identifier to bind to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunId(pub Uuid);

impl RunId {
    /// Create a fresh `RunId` backed by a new v4 UUID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Unique identifier for one provider turn (one model call) within a
/// run.
///
/// A run is composed of one or more steps: each step is a
/// `chat_stream` call plus any tool execution that follows. The engine
/// mints a fresh `StepId` at the start of every step and surfaces it on
/// [`crate::StreamChunk::StepStarted`] / [`crate::StreamChunk::StepFinished`]
/// so middlewares can scope per-step state (e.g. token counts, retries)
/// without tracking iteration numbers themselves.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StepId(pub Uuid);

impl StepId {
    /// Create a fresh `StepId` backed by a new v4 UUID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for StepId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
