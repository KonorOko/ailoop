//! [`MaxToolCalls`]: middleware that caps the total number of tool
//! invocations across an entire run.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use ailoop_core::{
    ChatMiddleware, FinishReason, Message, RunId, ToolCallInfo, ToolDecision, Usage,
};
use serde_json::Value;

/// Middleware that aborts a run once the total tool-call count
/// reaches a fixed cap.
///
/// [`crate::Conversation`]'s [`ailoop_core::RunConfig::max_iterations`]
/// only counts *steps* (one provider turn plus its tool calls), so a
/// model that emits 30 parallel tool calls in a single turn still
/// burns just one iteration. With MCP / dynamic tool sets, that
/// shape is common — a malformed prompt can drive runaway tool
/// spam without ever tripping the iteration cap. `MaxToolCalls`
/// gives you a flat cap on the absolute number of executions.
///
/// On the (N+1)-th call the middleware returns
/// [`ToolDecision::Terminate`], which the engine surfaces as
/// [`FinishReason::Aborted`](ailoop_core::FinishReason::Aborted)
/// carrying [`AbortReason::ToolTerminated`](ailoop_core::AbortReason::ToolTerminated) while
/// preserving any tool results already produced in the current step.
/// Pair with [`crate::AntiLoop`] for content-aware repetition
/// detection — the two are independent and compose.
///
/// State is keyed by [`RunId`], so a single middleware instance
/// can be shared across concurrent conversations without
/// cross-attributing calls between runs.
pub struct MaxToolCalls {
    max: usize,
    counts: Mutex<HashMap<RunId, usize>>,
}

impl MaxToolCalls {
    /// Cap the run at `max` tool calls. `max = 0` means *no* tool
    /// calls — the first invocation already terminates the run. There
    /// is no default; you opt in explicitly with a number that
    /// matches your worst-case acceptable budget.
    pub fn new(max: usize) -> Self {
        Self {
            max,
            counts: Mutex::new(HashMap::new()),
        }
    }

    fn counts(&self) -> MutexGuard<'_, HashMap<RunId, usize>> {
        // Poisoning only means another thread panicked mid-update; the
        // map itself is still consistent.
        self.counts.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(test)]
    pub(crate) fn tracked_runs(&self) -> usize {
        self.counts().len()
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for MaxToolCalls {
    async fn on_before_tool_call(&self, call: &ToolCallInfo, _args: &Value) -> ToolDecision {
        let mut guard = self.counts();
        let counter = guard.entry(call.run_id.clone()).or_insert(0);
        *counter += 1;
        if *counter > self.max {
            let n = *counter;
            let max = self.max;
            return ToolDecision::Terminate {
                reason: format!("max-tool-calls: tool call #{n} exceeds cap of {max} per run"),
            };
        }
        ToolDecision::Continue
    }

    async fn on_run_finished(
        &self,
        run_id: &RunId,
        _reason: &FinishReason,
        _usage: &Usage,
        _new_messages: &[Message],
    ) {
        self.counts().remove(run_id);
    }

    async fn on_run_error(
        &self,
        run_id: &RunId,
        _err: &(dyn std::error::Error + Send + Sync),
        _usage: &Usage,
        _partial_messages: &[Message],
    ) {
        self.counts().remove(run_id);
    }

    fn on_run_dropped(&self, run_id: &RunId) {
        self.counts().remove(run_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::StepId;
    use serde_json::json;
    use std::sync::Arc;

    /// Calls below the cap pass through unchanged.
    #[tokio::test]
    async fn allows_calls_under_cap() {
        let mw = MaxToolCalls::new(3);
        let run_id = RunId::new();
        let step_id = StepId::new();
        for _ in 0..3 {
            let decision = mw
                .on_before_tool_call(
                    &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "any"),
                    &json!({}),
                )
                .await;
            assert!(
                matches!(decision, ToolDecision::Continue),
                "expected Continue under cap"
            );
        }
    }

    /// The (N+1)-th call trips the gate with a descriptive reason.
    #[tokio::test]
    async fn terminates_on_call_over_cap() {
        let mw = MaxToolCalls::new(2);
        let run_id = RunId::new();
        let step_id = StepId::new();
        for _ in 0..2 {
            assert!(matches!(
                mw.on_before_tool_call(
                    &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "t"),
                    &json!({})
                )
                .await,
                ToolDecision::Continue
            ));
        }
        match mw
            .on_before_tool_call(
                &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "t"),
                &json!({}),
            )
            .await
        {
            ToolDecision::Terminate { reason } => {
                assert!(
                    reason.contains("max-tool-calls") && reason.contains("cap of 2"),
                    "unexpected reason: {reason}"
                );
            }
            _ => panic!("expected Terminate"),
        }
    }

    /// `max = 0` blocks the very first call.
    #[tokio::test]
    async fn zero_cap_blocks_first_call() {
        let mw = MaxToolCalls::new(0);
        let run_id = RunId::new();
        let step_id = StepId::new();
        match mw
            .on_before_tool_call(
                &ToolCallInfo::new(run_id.clone(), step_id.clone(), "toolu_test", "t"),
                &json!({}),
            )
            .await
        {
            ToolDecision::Terminate { .. } => {}
            _ => panic!("expected Terminate"),
        }
    }

    /// Counters are keyed by `RunId`, so a shared middleware does not
    /// cross-attribute calls between concurrent conversations.
    #[tokio::test]
    async fn counts_are_isolated_per_run() {
        let mw = Arc::new(MaxToolCalls::new(2));
        let run_a = RunId::new();
        let run_b = RunId::new();
        let step = StepId::new();

        // Saturate run A.
        for _ in 0..2 {
            assert!(matches!(
                mw.on_before_tool_call(
                    &ToolCallInfo::new(run_a.clone(), step.clone(), "toolu_test", "t"),
                    &json!({})
                )
                .await,
                ToolDecision::Continue
            ));
        }

        // Run B is unaffected.
        assert!(matches!(
            mw.on_before_tool_call(
                &ToolCallInfo::new(run_b.clone(), step.clone(), "toolu_test", "t"),
                &json!({})
            )
            .await,
            ToolDecision::Continue
        ));

        // Run A's next call still trips.
        assert!(matches!(
            mw.on_before_tool_call(
                &ToolCallInfo::new(run_a.clone(), step.clone(), "toolu_test", "t"),
                &json!({})
            )
            .await,
            ToolDecision::Terminate { .. }
        ));
    }

    /// `on_run_finished` clears the per-run counter, so a long-lived
    /// middleware does not leak memory across many runs.
    #[tokio::test]
    async fn finished_run_clears_state() {
        let mw = MaxToolCalls::new(5);
        let run_id = RunId::new();
        let step = StepId::new();
        mw.on_before_tool_call(
            &ToolCallInfo::new(run_id.clone(), step.clone(), "toolu_test", "t"),
            &json!({}),
        )
        .await;
        assert_eq!(mw.counts().len(), 1);
        mw.on_run_finished(&run_id, &FinishReason::EndTurn, &Usage::default(), &[])
            .await;
        assert!(mw.counts().is_empty());
    }
}
