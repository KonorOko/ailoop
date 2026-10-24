//! Moving a running tool call to the background: see
//! [`crate::RunControl::background_tool`] and [`BackgroundConfig`].

use std::sync::Arc;

use serde_json::Value;

use crate::ids::{RunId, StepId};
use crate::message::ToolResultContent;
use crate::middleware::ToolCallInfo;

type NoticeFn = dyn Fn(&ToolCallInfo) -> String + Send + Sync;
type ResultFn = dyn Fn(BackgroundToolResult) + Send + Sync;

/// How a run handles the tool calls moved to the background with
/// [`crate::RunControl::background_tool`]: the provisional result the
/// model gets when a call is moved, and where its real result goes.
///
/// Set it on `RunConfig::background` (or `RunOptions::background` on a
/// `Conversation`). The default answers a moved call with
/// [`Self::default_notice`] and has no callback: the real result then
/// reaches only the run that moved the call, and only if that run still
/// takes it (see [`BackgroundToolResult::delivered`]); otherwise it is
/// dropped.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct BackgroundConfig {
    notice: Option<Arc<NoticeFn>>,
    on_result: Option<Arc<ResultFn>>,
}

impl BackgroundConfig {
    /// No callback and the default notice.
    pub fn new() -> Self {
        Self::default()
    }

    /// The text of the provisional result a call gets when it is moved,
    /// built from the call. It is a successful result (`is_error:
    /// false`) and goes through the run's `on_after_tool_call*` hooks
    /// like any other.
    pub fn notice(
        mut self,
        notice: impl Fn(&ToolCallInfo) -> String + Send + Sync + 'static,
    ) -> Self {
        self.notice = Some(Arc::new(notice));
        self
    }

    /// Called once for every call this run moves to the background,
    /// with its real result, whether or not the run took that result
    /// (see [`BackgroundToolResult::delivered`]). It may be called from
    /// the run's stream, from the call's own task, or when the run ends
    /// (its stream dropped included), possibly after the run is over,
    /// so it must not block: hand the result over (a channel send, say)
    /// and return.
    pub fn on_result(
        mut self,
        on_result: impl Fn(BackgroundToolResult) + Send + Sync + 'static,
    ) -> Self {
        self.on_result = Some(Arc::new(on_result));
        self
    }

    /// The text a moved call gets when no [`Self::notice`] is set.
    pub fn default_notice(call: &ToolCallInfo) -> String {
        format!(
            "Moved to the background while running (call id {}). Its result will arrive \
             later in a user message; do not call the tool again for it.",
            call.call_id
        )
    }

    /// The provisional result for `call`.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn notice_for(&self, call: &ToolCallInfo) -> ToolResultContent {
        let text = match &self.notice {
            Some(notice) => notice(call),
            None => Self::default_notice(call),
        };
        ToolResultContent::text(text)
    }

    /// Hand `result` to the callback, if there is one.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn report(&self, result: BackgroundToolResult) {
        if let Some(on_result) = &self.on_result {
            on_result(result);
        }
    }
}

impl std::fmt::Debug for BackgroundConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundConfig")
            .field("notice", &self.notice.is_some())
            .field("on_result", &self.on_result.is_some())
            .finish()
    }
}

/// The real result of a tool call moved to the background, handed to
/// [`BackgroundConfig::on_result`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BackgroundToolResult {
    /// The run that moved the call.
    pub run_id: RunId,
    /// The step the call belongs to.
    pub step_id: StepId,
    /// The call, as on its `ToolCallStarted`.
    pub call_id: String,
    /// The tool's name.
    pub name: String,
    /// The arguments the tool ran with (after the run's
    /// `on_before_tool_call_mut` hooks).
    pub args: Value,
    /// What the tool returned, after the run's `on_after_tool_call_mut`
    /// hooks (so a `Sanitize` applies). An error result when the call
    /// was cancelled ([`Self::cancelled`]) or the tool panicked.
    pub content: ToolResultContent,
    /// The call was stopped by
    /// [`crate::RunControl::cancel_background`].
    pub cancelled: bool,
    /// Whether the run that moved the call took this result: it is
    /// then part of that run's messages, as an [`crate::Inbound`] whose
    /// id is [`Self::call_id`], reported on
    /// [`crate::StreamChunk::InboundDelivered`]. When `false` that run
    /// had stopped taking results (it ended, or its last turn closed
    /// before the result arrived) and nothing delivered it: it is the
    /// caller's to deliver, for instance by pushing it on the next
    /// run's [`crate::RunControl`].
    pub delivered: bool,
}

impl BackgroundToolResult {
    /// A result not cancelled and not delivered.
    pub fn new(
        run_id: RunId,
        step_id: StepId,
        call_id: impl Into<String>,
        name: impl Into<String>,
        args: Value,
        content: ToolResultContent,
    ) -> Self {
        Self {
            run_id,
            step_id,
            call_id: call_id.into(),
            name: name.into(),
            args,
            content,
            cancelled: false,
            delivered: false,
        }
    }
}
