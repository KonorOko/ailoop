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

/// Computes the equivalence key for a tool call. Receives the tool
/// name and its arguments; the returned string is what the loop
/// detector compares across consecutive invocations — two calls count
/// as "the same call" iff their identity strings are byte-equal.
///
/// Plug a custom identity via [`AntiLoop::with_tool_call_identity`] to
/// absorb cosmetic argument variation that the default structural
/// `serde_json::Value` comparison treats as distinct (whitespace
/// tweaks inside string fields, ignored auxiliary fields, etc.).
pub type ToolCallIdentity = Arc<dyn Fn(&str, &Value) -> String + Send + Sync>;

#[derive(Default)]
struct RunState {
    /// Tracks the previous call when no custom identity is configured.
    last_tool_call: Option<(String, Value)>,
    /// Tracks the previous call's identity string when a custom
    /// identity callback is configured. Only one of `last_tool_call`
    /// and `last_tool_call_id` is ever populated per run, depending on
    /// whether `AntiLoop::tool_call_identity` is set.
    last_tool_call_id: Option<String>,
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
///   not matter). Plug a custom equivalence with
///   [`Self::with_tool_call_identity`] when callers re-issue calls
///   with cosmetic argument variation that should still count as
///   repeats.
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
    tool_call_identity: Option<ToolCallIdentity>,
    inner: Mutex<Inner>,
}

impl AntiLoop {
    /// Build with the default thresholds (`3` for both detectors), the
    /// default text predicate (trim + equality), and structural
    /// `serde_json::Value` `PartialEq` for tool-call equivalence.
    pub fn new() -> Self {
        Self {
            max_repeated_tool_calls: 3,
            max_repeated_texts: 3,
            text_predicate: Arc::new(|a, b| a.trim() == b.trim()),
            tool_call_identity: None,
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

    /// Replace the default tool-call equivalence with a custom
    /// identity function. The callback maps `(name, args)` to a
    /// string; the loop detector counts consecutive calls whose
    /// identities are byte-equal.
    ///
    /// Use this to catch calls that the model is repeating with
    /// cosmetic argument variation (string whitespace, ignored
    /// auxiliary fields, key reordering inside an embedded JSON
    /// string) which the default `serde_json::Value` `PartialEq` would
    /// treat as distinct and thus reset the streak. When the detector
    /// fires under a custom identity, the terminate reason includes
    /// the identity string for diagnostics.
    ///
    /// Mirrors [`Self::with_text_predicate`] for the text detector,
    /// with one intentional asymmetry: text takes a predicate,
    /// tool-call takes an identity. Identity strictly subsumes
    /// predicate (the predicate `id(a) == id(b)` is recoverable from
    /// any identity), allows lighter per-run state (an
    /// `Option<String>` instead of `Option<(String, Value)>`), and
    /// gives a useful key in the terminate reason for free.
    pub fn with_tool_call_identity<F>(mut self, identity: F) -> Self
    where
        F: Fn(&str, &Value) -> String + Send + Sync + 'static,
    {
        self.tool_call_identity = Some(Arc::new(identity));
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
            match &self.tool_call_identity {
                None => {
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
                Some(identity) => {
                    let id = identity(name, args);
                    let same = matches!(&state.last_tool_call_id, Some(prev) if prev == &id);
                    if same {
                        state.tool_call_streak += 1;
                    } else {
                        state.last_tool_call_id = Some(id.clone());
                        state.tool_call_streak = 1;
                    }
                    if state.tool_call_streak >= self.max_repeated_tool_calls {
                        let n = state.tool_call_streak;
                        return ToolDecision::Terminate {
                            reason: format!(
                                "anti-loop: tool '{name}' with identity '{id}' called {n} times in a row"
                            ),
                        };
                    }
                }
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
