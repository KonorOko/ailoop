//! [`TimeoutTool`]: per-tool wall-clock cap that wraps any [`ToolDyn`].

use std::time::Duration;

use ailoop_core::{ToolDefinition, ToolResultContent};
use async_trait::async_trait;

use crate::context::ToolContext;
use crate::registry::ToolDyn;

/// [`ToolDyn`] wrapper that caps a single tool invocation at
/// `timeout`. When the inner future does not resolve in time the
/// wrapper returns a [`ToolResultContent`] with `is_error: true` so
/// the engine keeps feeding the model — the run does not abort.
///
/// Per-tool granularity matters because different tools have wildly
/// different expected latencies: a `get_weather` should not have the
/// same cap as a `run_terraform_apply`. The run-wide
/// [`RunConfig::timeout`](ailoop_core::RunConfig::timeout) stays the
/// right knob for the *overall* run; this wrapper is the right knob
/// for an individual MCP/HTTP tool that can hang.
///
/// `name()` and `tool_definition()` delegate to the inner tool so
/// the model sees the wrapper transparently. Wrap once at
/// registration time:
///
/// ```ignore
/// use std::time::Duration;
/// use ailoop_tools::TimeoutTool;
///
/// Conversation::builder(model)
///     .tool(TimeoutTool::new(SlowMcpTool, Duration::from_secs(30)))
///     .build()?;
/// ```
pub struct TimeoutTool<T: ToolDyn> {
    inner: T,
    timeout: Duration,
}

impl<T: ToolDyn> TimeoutTool<T> {
    /// Wrap `inner` so that any single call exceeding `timeout`
    /// returns an `is_error: true` [`ToolResultContent`] instead of
    /// running to completion.
    pub fn new(inner: T, timeout: Duration) -> Self {
        Self { inner, timeout }
    }
}

#[async_trait]
impl<T: ToolDyn> ToolDyn for TimeoutTool<T> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn tool_definition(&self) -> ToolDefinition {
        self.inner.tool_definition()
    }

    fn concurrency_safe(&self) -> bool {
        self.inner.concurrency_safe()
    }

    /// Forwarded: a moved call still stops at `timeout`, counted from
    /// when it started.
    fn backgroundable(&self) -> bool {
        self.inner.backgroundable()
    }

    async fn call(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolResultContent {
        match tokio::time::timeout(self.timeout, self.inner.call(args, ctx)).await {
            Ok(content) => content,
            Err(_) => ToolResultContent::error(format!(
                "Tool '{}' timed out after {:?}",
                self.inner.name(),
                self.timeout,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::ToolTag;
    use serde_json::json;

    struct InstantTool;

    #[async_trait]
    impl ToolDyn for InstantTool {
        fn name(&self) -> &str {
            "instant"
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "instant",
                "returns immediately",
                json!({ "type": "object", "properties": {}, "required": [] }),
                vec![ToolTag::ReadOnly],
            )
        }
        fn concurrency_safe(&self) -> bool {
            true
        }
        async fn call(&self, _: serde_json::Value, _: &ToolContext) -> ToolResultContent {
            ToolResultContent::text("ok")
        }
    }

    struct SlowTool;

    #[async_trait]
    impl ToolDyn for SlowTool {
        fn name(&self) -> &str {
            "slow"
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "slow",
                "sleeps forever",
                json!({ "type": "object", "properties": {}, "required": [] }),
                vec![],
            )
        }
        // The opposite of its concurrency safety, so a wrapper that
        // forwards one flag for the other fails the tests below.
        fn backgroundable(&self) -> bool {
            true
        }
        async fn call(&self, _: serde_json::Value, _: &ToolContext) -> ToolResultContent {
            tokio::time::sleep(Duration::from_secs(60)).await;
            ToolResultContent::text("never")
        }
    }

    /// Definition and name pass through unchanged so the model sees the
    /// inner tool's schema — wrapping is transparent on the wire.
    #[tokio::test]
    async fn delegates_name_and_definition() {
        let wrapped = TimeoutTool::new(InstantTool, Duration::from_millis(10));
        assert_eq!(wrapped.name(), "instant");
        let def = wrapped.tool_definition();
        assert_eq!(def.name, "instant");
        assert_eq!(def.tags, vec![ToolTag::ReadOnly]);
    }

    /// The wrapper keeps the inner tool's concurrency safety, so
    /// wrapping a safe tool does not make its calls run one at a time.
    #[test]
    fn delegates_concurrency_safety() {
        let safe = TimeoutTool::new(InstantTool, Duration::from_millis(10));
        assert!(safe.concurrency_safe());
        let unsafe_tool = TimeoutTool::new(SlowTool, Duration::from_millis(10));
        assert!(!unsafe_tool.concurrency_safe());
    }

    /// The wrapper keeps the inner tool's opt-in to being moved to the
    /// background, so wrapping it does not make its calls stick to the
    /// run.
    #[test]
    fn delegates_backgroundable() {
        assert!(TimeoutTool::new(SlowTool, Duration::from_millis(10)).backgroundable());
        assert!(!TimeoutTool::new(InstantTool, Duration::from_millis(10)).backgroundable());
    }

    /// A call that resolves inside the budget passes through untouched.
    #[tokio::test]
    async fn fast_tool_returns_inner_result() {
        let wrapped = TimeoutTool::new(InstantTool, Duration::from_secs(1));
        let ctx = ToolContext::detached();
        let result = wrapped.call(json!({}), &ctx).await;
        assert!(!result.is_error);
        assert_eq!(result.blocks.len(), 1);
    }

    /// A call that overshoots the budget surfaces an `is_error: true`
    /// result. The run keeps going — the engine feeds the error back to
    /// the model on the next turn.
    #[tokio::test]
    async fn slow_tool_returns_error_after_timeout() {
        let wrapped = TimeoutTool::new(SlowTool, Duration::from_millis(50));
        let ctx = ToolContext::detached();
        let result = wrapped.call(json!({}), &ctx).await;
        assert!(result.is_error, "timed-out tool must report as error");
        // The error message names the tool so logs are diagnosable.
        let text = result
            .blocks
            .first()
            .and_then(|b| match b {
                ailoop_core::ToolResultBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or("");
        assert!(
            text.contains("slow") && text.contains("timed out"),
            "unexpected error text: {text}"
        );
    }
}
