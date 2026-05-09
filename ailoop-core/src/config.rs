use std::sync::Arc;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::ids::RunId;
use crate::message::SystemPrompt;
use crate::middleware::ChatMiddleware;

pub struct RunConfig {
    pub system_prompt: Option<SystemPrompt>,
    pub max_iterations: usize,
    pub max_tokens: u32,
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
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..Default::default()
        }
    }
}
