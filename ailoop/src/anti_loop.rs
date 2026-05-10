use std::collections::HashMap;
use std::sync::Arc;

use ailoop_core::{
    ChatMiddleware, FinishReason, Message, RunId, StepId, StreamChunk, ToolDecision, Usage,
};
use serde_json::Value;
use tokio::sync::Mutex;

/// Predicate used to compare two assistant text turns. Receives
/// `(previous, current)` and returns `true` when the texts should count
/// as a repeated turn for the loop detector. Default trims both sides
/// and compares for equality, which absorbs the trailing newline that
/// some providers emit at the end of a turn.
pub type TextPredicate = Arc<dyn Fn(&str, &str) -> bool + Send + Sync>;

#[derive(Default)]
struct RunState {
    last_tool_call: Option<(String, Value)>,
    tool_call_streak: usize,
    last_text: Option<String>,
    text_streak: usize,
    text_buffer: String,
    last_step_processed: Option<StepId>,
}

#[derive(Default)]
struct Inner {
    runs: HashMap<RunId, RunState>,
    /// The run whose step is currently being streamed. Set on
    /// `StepStarted`, used by `TextDelta` to know which run's buffer
    /// to append to (since `TextDelta` itself carries no `RunId`).
    active_text_run: Option<RunId>,
}

/// Middleware that aborts a run when the model gets stuck repeating
/// itself. Two independent detectors run side by side:
///
/// - **Tool-call loop**: trips when the same tool is invoked
///   `max_repeated_tool_calls` times in a row with structurally equal
///   arguments (`serde_json::Value` `PartialEq`, so key ordering does
///   not matter).
/// - **Looped text**: trips when the assistant's `Text` block is
///   considered identical (per the configured [`TextPredicate`]) across
///   `max_repeated_texts` consecutive turns.
///
/// Both default to a threshold of `3`. Setting either to `0` disables
/// that detector individually.
///
/// On detection the middleware returns
/// [`ToolDecision::Terminate`](ailoop_core::ToolDecision::Terminate)
/// from `on_before_tool_call`, which the engine surfaces as
/// `FinishReason::Aborted(_)` while preserving any tool results already
/// produced in the current step. `Skip` is deliberately not used: a
/// skip would feed an error back to the model, leaving the loop intact.
///
/// The text detector relies on `on_before_tool_call` as its
/// termination point, so a pure text-only run with no tool calls cannot
/// be aborted mid-stream. That is intentional and not a limitation in
/// practice: the engine breaks out of its iteration loop on
/// `FinishReason::EndTurn`, so a text-only turn already ends the run on
/// its own.
pub struct AntiLoop {
    max_repeated_tool_calls: usize,
    max_repeated_texts: usize,
    text_predicate: TextPredicate,
    inner: Mutex<Inner>,
}

impl AntiLoop {
    /// Build with the default thresholds (`3` for both detectors) and
    /// the default text predicate (trim + equality).
    pub fn new() -> Self {
        Self {
            max_repeated_tool_calls: 3,
            max_repeated_texts: 3,
            text_predicate: Arc::new(|a, b| a.trim() == b.trim()),
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Override the threshold for consecutive identical tool calls.
    /// Pass `0` to disable the tool-call detector entirely.
    pub fn with_max_repeated_tool_calls(mut self, n: usize) -> Self {
        self.max_repeated_tool_calls = n;
        self
    }

    /// Override the threshold for consecutive identical assistant text
    /// turns. Pass `0` to disable the text detector entirely.
    pub fn with_max_repeated_texts(mut self, n: usize) -> Self {
        self.max_repeated_texts = n;
        self
    }

    /// Replace the equality predicate used by the text detector. The
    /// callback receives `(previous_turn, current_turn)` and should
    /// return `true` when the two should be treated as identical for
    /// streak-counting purposes. Useful for fuzzy comparisons (case
    /// folding, prefix matching, edit distance via an external crate).
    pub fn with_text_predicate<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&str, &str) -> bool + Send + Sync + 'static,
    {
        self.text_predicate = Arc::new(predicate);
        self
    }
}

impl Default for AntiLoop {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for AntiLoop {
    async fn on_run_started(
        &self,
        run_id: &RunId,
        _messages: &[Message],
        _config: &ailoop_core::RunConfig,
    ) -> ailoop_core::HookAction {
        self.inner
            .lock()
            .await
            .runs
            .insert(run_id.clone(), RunState::default());
        ailoop_core::HookAction::Continue
    }

    async fn on_chunk(&self, chunk: &StreamChunk) {
        match chunk {
            StreamChunk::StepStarted { run_id, .. } => {
                let mut guard = self.inner.lock().await;
                guard.active_text_run = Some(run_id.clone());
                let state = guard.runs.entry(run_id.clone()).or_default();
                state.text_buffer.clear();
            }
            StreamChunk::TextDelta { delta } => {
                // `TextDelta` does not carry a `RunId`. Append to the
                // run whose step is currently in flight, tracked by
                // `active_text_run`. A single `AntiLoop` shared across
                // concurrent conversations may cross-attribute text
                // here — wire one instance per `Conversation` when
                // concurrent runs matter.
                let mut guard = self.inner.lock().await;
                if let Some(run_id) = guard.active_text_run.clone()
                    && let Some(state) = guard.runs.get_mut(&run_id)
                {
                    state.text_buffer.push_str(delta);
                }
            }
            _ => {}
        }
    }

    async fn on_before_tool_call(
        &self,
        run_id: &RunId,
        step_id: &StepId,
        name: &str,
        args: &Value,
    ) -> ToolDecision {
        let mut guard = self.inner.lock().await;
        let state = guard.runs.entry(run_id.clone()).or_default();

        // First tool call of this step closes out the assistant's text
        // turn and updates the text streak. Subsequent tool calls in
        // the same step skip this work.
        if state.last_step_processed.as_ref() != Some(step_id) {
            state.last_step_processed = Some(step_id.clone());
            let current_text = std::mem::take(&mut state.text_buffer);
            if current_text.is_empty() {
                state.last_text = None;
                state.text_streak = 0;
            } else {
                let same = match &state.last_text {
                    Some(prev) => (self.text_predicate)(prev, &current_text),
                    None => false,
                };
                if same {
                    state.text_streak += 1;
                } else {
                    state.last_text = Some(current_text);
                    state.text_streak = 1;
                }
            }
        }

        if self.max_repeated_texts > 0 && state.text_streak >= self.max_repeated_texts {
            let n = state.text_streak;
            return ToolDecision::Terminate {
                reason: format!("anti-loop: assistant text repeated identically across {n} turns"),
            };
        }

        if self.max_repeated_tool_calls > 0 {
            let same = matches!(
                &state.last_tool_call,
                Some((prev_name, prev_args)) if prev_name == name && prev_args == args
            );
            if same {
                state.tool_call_streak += 1;
            } else {
                state.last_tool_call = Some((name.to_string(), args.clone()));
                state.tool_call_streak = 1;
            }
            if state.tool_call_streak >= self.max_repeated_tool_calls {
                let n = state.tool_call_streak;
                return ToolDecision::Terminate {
                    reason: format!(
                        "anti-loop: tool '{name}' called {n} times in a row with identical args"
                    ),
                };
            }
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
        let mut guard = self.inner.lock().await;
        guard.runs.remove(run_id);
        if guard.active_text_run.as_ref() == Some(run_id) {
            guard.active_text_run = None;
        }
    }

    async fn on_run_error(&self, run_id: &RunId, _err: &(dyn std::error::Error + Send + Sync)) {
        let mut guard = self.inner.lock().await;
        guard.runs.remove(run_id);
        if guard.active_text_run.as_ref() == Some(run_id) {
            guard.active_text_run = None;
        }
    }
}
