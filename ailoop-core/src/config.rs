//! Per-run configuration: see [`RunConfig`].

use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::ids::RunId;
use crate::message::SystemPrompt;
use crate::middleware::ChatMiddleware;

/// Default for [`RunConfig::max_iterations`]: the number of provider
/// turns a run may take before the engine aborts it with
/// [`crate::AbortReason::MaxIterations`]. A safety brake, not a budget;
/// see the field docs.
pub const DEFAULT_MAX_ITERATIONS: usize = 25;

/// Default for [`RunConfig::max_tokens`] (and
/// [`crate::ChatRequest::max_tokens`]): the output-token cap sent with
/// every request when neither the conversation nor the run sets one.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Per-run configuration consumed by the engine entry point.
///
/// The defaults shipped via [`RunConfig::default`] are 25 iterations
/// ([`DEFAULT_MAX_ITERATIONS`]), 4096 max output tokens
/// ([`DEFAULT_MAX_TOKENS`]) and no timeout. They are safety brakes, not a
/// budget: in production, set `max_iterations` and `timeout`
/// explicitly and add a tool-call cap (`MaxToolCalls` in the `ailoop`
/// crate). Use struct-update syntax to override what you need:
/// `RunConfig { max_iterations: 10, ..Default::default() }`. The
/// struct is `#[non_exhaustive]`, so external callers must always go
/// through `Default` (or [`RunConfig::new`]) to construct it.
#[non_exhaustive]
pub struct RunConfig {
    /// System prompt prepended to the conversation. `None` lets the
    /// provider use its own default behaviour. Use
    /// [`SystemPrompt::Blocks`] to opt in to per-block cache breakpoints.
    pub system_prompt: Option<SystemPrompt>,
    /// Maximum number of provider turns before the engine aborts the
    /// run with [`crate::FinishReason::Aborted`] carrying
    /// [`crate::AbortReason::MaxIterations`]. Like every abort, this is
    /// not an error: the partial `new_messages` are kept and
    /// `on_run_finished` fires. One iteration covers a
    /// `chat_stream` call plus the tool calls it triggers. The cap
    /// prevents runaway tool-use loops; pair with an [`crate::ChatMiddleware`]
    /// such as `AntiLoop` for content-aware loop detection.
    ///
    /// Defaults to [`DEFAULT_MAX_ITERATIONS`] (25). This is a safety brake, not a budget: it bounds
    /// how long a runaway loop can go on, not what a run costs (a turn
    /// can spend any number of tokens). Every
    /// [`crate::ContinueDecision::Continue`] from `on_turn_end` counts as
    /// an iteration. In production, set it explicitly for your workload,
    /// together with [`Self::timeout`] and a tool-call cap
    /// (`MaxToolCalls` in the `ailoop` crate).
    ///
    /// The default also applies to sub-agents: a `SubAgentTool` child
    /// with no explicit cap (via `SubAgentConfig::max_iterations` or
    /// its own conversation) runs with 25 iterations too.
    pub max_iterations: usize,
    /// Default `max_tokens` for every per-turn [`crate::ChatRequest`]
    /// the engine builds. User-supplied middlewares can override this
    /// per request via [`crate::ChatMiddleware::on_chat_request`].
    /// Defaults to [`DEFAULT_MAX_TOKENS`] (4096).
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
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_tokens: DEFAULT_MAX_TOKENS,
            middlewares: vec![],
            run_id: None,
            timeout: None,
            cancellation: None,
        }
    }
}

impl std::fmt::Debug for RunConfig {
    /// Middlewares are trait objects, so only their count is shown.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunConfig")
            .field("system_prompt", &self.system_prompt)
            .field("max_iterations", &self.max_iterations)
            .field("max_tokens", &self.max_tokens)
            .field("middlewares", &self.middlewares.len())
            .field("run_id", &self.run_id)
            .field("timeout", &self.timeout)
            .field("cancellation", &self.cancellation)
            .finish()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchors the pre-1.0 default: raising or lowering it changes the
    /// per-run cost ceiling for everyone who does not set it, so it
    /// must be a deliberate, changelogged change.
    #[test]
    fn run_config_default_max_iterations_is_25() {
        assert_eq!(RunConfig::default().max_iterations, 25);
    }

    #[test]
    fn run_config_default_uses_the_named_consts() {
        let config = RunConfig::default();
        assert_eq!(config.max_iterations, DEFAULT_MAX_ITERATIONS);
        assert_eq!(config.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(DEFAULT_MAX_TOKENS, 4096);
        assert_eq!(crate::ChatRequest::default().max_tokens, DEFAULT_MAX_TOKENS);
    }
}
