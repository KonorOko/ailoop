//! Per-run configuration: see [`RunConfig`].

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::ids::RunId;
use crate::message::SystemPrompt;
use crate::middleware::ChatMiddleware;

/// Per-run configuration consumed by the engine entry point.
///
/// The defaults shipped via [`RunConfig::default`] are tuned for
/// short interactive turns (10 iterations, 4096 max output tokens, no
/// timeout). Use struct-update syntax to override what you need:
/// `RunConfig { max_iterations: 20, ..Default::default() }`. The
/// struct is `#[non_exhaustive]`, so external callers must always go
/// through `Default` (or [`RunConfig::new`]) to construct it.
#[non_exhaustive]
pub struct RunConfig {
    /// System prompt prepended to the conversation. `None` lets the
    /// provider use its own default behaviour. Use
    /// [`SystemPrompt::Blocks`] to opt in to per-block cache breakpoints.
    pub system_prompt: Option<SystemPrompt>,
    /// Maximum number of provider turns before the engine aborts the
    /// run with [`crate::FinishReason::Aborted`]. One iteration covers a
    /// `chat_stream` call plus the tool calls it triggers. The cap
    /// prevents runaway tool-use loops; pair with an [`crate::ChatMiddleware`]
    /// such as `AntiLoop` for content-aware loop detection.
    pub max_iterations: usize,
    /// Default `max_tokens` for every per-turn [`crate::ChatRequest`]
    /// the engine builds. User-supplied middlewares can override this
    /// per request via [`crate::ChatMiddleware::on_chat_request`].
    pub max_tokens: u32,
    /// Middlewares the engine invokes in registration order. The
    /// façade prepends an internal middleware that injects per-request
    /// defaults; entries supplied here run after it and can override
    /// any field.
    pub middlewares: Vec<Arc<dyn ChatMiddleware>>,
    /// Caller-supplied id for the run. When `None`, the engine mints a
    /// fresh UUID v4. Set this when an outer system needs to correlate
    /// the run with its own trace id.
    pub run_id: Option<RunId>,
    /// Wall-clock deadline for the entire run, including tool calls and
    /// any retry backoff inside `RetryingModel`. `None` disables the
    /// timeout. The engine checks this at await boundaries (HTTP setup,
    /// SSE chunks, tool execution, approval middleware) — synchronous
    /// work is not preempted. Sleeps inside `RetryingModel`'s backoff
    /// race against this deadline because they run under the engine's
    /// `select!`, so retry attempts are interruptible without the
    /// decorator knowing about cancellation.
    pub timeout: Option<Duration>,
    /// External cancellation handle. Calling `cancel()` from another
    /// task aborts the in-flight run at the next await boundary, with
    /// the same persistence discipline as the timeout (partial
    /// `tools_result` preserved, `on_run_finished` fired). Pass
    /// `parent.child_token()` if you want to cancel this run without
    /// affecting siblings sharing the parent.
    pub cancellation: Option<CancellationToken>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_iterations: 10,
            max_tokens: 4096,
            middlewares: vec![],
            run_id: None,
            timeout: None,
            cancellation: None,
        }
    }
}

impl RunConfig {
    /// Build a config with the given iteration cap and otherwise
    /// default values. Equivalent to
    /// `RunConfig { max_iterations, ..Default::default() }`; provided
    /// because capping iterations is the most common single-field
    /// override.
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..Default::default()
        }
    }
}
