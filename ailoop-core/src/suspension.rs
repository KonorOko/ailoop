//! A run paused on deferred tool calls, and how to resume it: see
//! [`Suspension`] and [`Resume`].

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::{RunId, StepId};
use crate::message::ToolResultContent;

/// The state a run leaves behind when a gate returns
/// [`crate::ToolDecision::Defer`]: the step it stopped in, with every
/// tool call of that step in the model's order.
///
/// The run ends with [`crate::FinishReason::Suspended`]. The step's
/// assistant message (with its `tool_use` blocks) is in the run's
/// messages, but its tool results are not: they live here until the
/// step is resumed (or abandoned), because providers need every result
/// of a step in the one user message that follows it. This is the only
/// case where the run's messages end on a `tool_use` without its
/// `tool_result`, and only while this value is kept with them.
///
/// Serializable so it can be persisted (a `ConversationSnapshot` carries
/// it) and resumed in another process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Suspension {
    /// The run that was suspended.
    pub run_id: RunId,
    /// The step it stopped in.
    pub step_id: StepId,
    /// Every tool call of the step, in the model's order.
    pub calls: Vec<SuspendedCall>,
}

impl Suspension {
    /// Build a suspension. The engine creates these; this is for tests
    /// and for callers that persist the parts themselves.
    pub fn new(run_id: RunId, step_id: StepId, calls: Vec<SuspendedCall>) -> Self {
        Self {
            run_id,
            step_id,
            calls,
        }
    }

    /// The calls waiting for a decision, in the model's order.
    pub fn pending(&self) -> impl Iterator<Item = &SuspendedCall> {
        self.calls
            .iter()
            .filter(|call| matches!(call.state, CallState::Deferred { .. }))
    }
}

/// One tool call of a suspended step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SuspendedCall {
    /// Provider-assigned id of the call.
    pub call_id: String,
    /// Wire name of the tool.
    pub name: String,
    /// Arguments the tool runs with if the call is approved. For a
    /// deferred call these are the arguments after every
    /// `on_before_tool_call_mut` hook, i.e. the payload the gate saw;
    /// resuming runs exactly these.
    pub args: Value,
    /// Where the call stands.
    pub state: CallState,
}

impl SuspendedCall {
    /// Build one call of a suspension.
    pub fn new(
        call_id: impl Into<String>,
        name: impl Into<String>,
        args: Value,
        state: CallState,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            args,
            state,
        }
    }
}

/// Where one call of a suspended step stands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CallState {
    /// Already has its result: it ran (and went through its `after`
    /// hooks), was answered without running, or, in a suspension put
    /// back after a failed or dropped resume, was interrupted while it
    /// ran (an error result saying it may have partly completed).
    /// Resuming keeps the result and never runs it again.
    Answered {
        /// The recorded result.
        content: ToolResultContent,
    },
    /// A gate deferred it; it needs a [`ResumeDecision`].
    Deferred {
        /// The reason the gate gave.
        reason: String,
    },
    /// Came after the call that suspended the step, so its gates have
    /// not run yet. Resuming takes it through them as usual; it may be
    /// deferred again.
    Waiting,
}

/// The decision for one deferred call when a run is resumed.
///
/// Whatever the decision, the call's `before` hooks do not run again
/// (they ran in the suspended run); its `after` hooks run on its
/// result, which goes out as a `ToolResult` like any other.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResumeDecision {
    /// Run the call with the arguments stored in the suspension.
    Approve,
    /// Do not run it; the model gets an error result carrying `reason`.
    Deny {
        /// Why it was denied, shown to the model.
        reason: String,
    },
    /// Do not run it; the model gets `content` as the call's result,
    /// as given (`is_error` included), after the call's
    /// `on_after_tool_call_mut` hooks (so a `Sanitize` covers it too).
    /// For a result that comes from outside the run: a person's answer,
    /// or one computed elsewhere. The tool is never invoked, even if the
    /// call was cancelled before its turn came.
    Answer {
        /// The call's result.
        content: ToolResultContent,
    },
}

impl ResumeDecision {
    /// Shorthand for [`Self::Deny`].
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// Shorthand for [`Self::Answer`].
    pub fn answer(content: impl Into<ToolResultContent>) -> Self {
        Self::Answer {
            content: content.into(),
        }
    }
}

/// A suspension together with a decision for each of its deferred
/// calls: what the engine needs to resume a run. Built with
/// [`Resume::new`], which checks the decisions match the deferred calls
/// exactly, so a resumed run never starts with a call left undecided.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Resume {
    /// The suspension being resumed.
    pub suspension: Suspension,
    /// One decision per deferred call, by `call_id`.
    pub decisions: HashMap<String, ResumeDecision>,
}

impl Resume {
    /// Pair `suspension` with `decisions`. Fails if a deferred call has
    /// no decision, or a decision names a call that is not deferred.
    pub fn new(
        suspension: Suspension,
        decisions: impl IntoIterator<Item = (String, ResumeDecision)>,
    ) -> Result<Self, ResumeError> {
        let decisions: HashMap<String, ResumeDecision> = decisions.into_iter().collect();
        for call_id in decisions.keys() {
            if !suspension.pending().any(|call| &call.call_id == call_id) {
                return Err(ResumeError::NotDeferred(call_id.clone()));
            }
        }
        if let Some(call) = suspension
            .pending()
            .find(|call| !decisions.contains_key(&call.call_id))
        {
            return Err(ResumeError::MissingDecision(call.call_id.clone()));
        }
        Ok(Self {
            suspension,
            decisions,
        })
    }
}

/// Why a set of decisions cannot resume a suspension.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ResumeError {
    /// A deferred call has no decision.
    #[error("no decision for deferred tool call `{0}`")]
    MissingDecision(String),
    /// A decision names a call that is not deferred in the suspension.
    #[error("tool call `{0}` is not deferred in this suspension")]
    NotDeferred(String),
    /// There is no suspension to resume.
    #[error("there is no suspended run to resume")]
    NotSuspended,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suspension() -> Suspension {
        Suspension::new(
            RunId::new(),
            StepId::new(),
            vec![
                SuspendedCall::new(
                    "a",
                    "read",
                    Value::Null,
                    CallState::Answered {
                        content: ToolResultContent::text("ok"),
                    },
                ),
                SuspendedCall::new(
                    "b",
                    "delete",
                    Value::Null,
                    CallState::Deferred {
                        reason: "needs approval".into(),
                    },
                ),
                SuspendedCall::new("c", "write", Value::Null, CallState::Waiting),
            ],
        )
    }

    #[test]
    fn pending_lists_only_deferred_calls() {
        let ids: Vec<_> = suspension().pending().map(|c| c.call_id.clone()).collect();
        assert_eq!(ids, ["b"]);
    }

    #[test]
    fn resume_requires_exactly_the_deferred_calls() {
        assert_eq!(
            Resume::new(suspension(), []).unwrap_err(),
            ResumeError::MissingDecision("b".into())
        );
        assert_eq!(
            Resume::new(
                suspension(),
                [
                    ("b".into(), ResumeDecision::Approve),
                    ("c".into(), ResumeDecision::Approve)
                ]
            )
            .unwrap_err(),
            ResumeError::NotDeferred("c".into())
        );
        assert!(Resume::new(suspension(), [("b".into(), ResumeDecision::deny("no"))]).is_ok());
    }

    #[test]
    fn answer_decision_round_trips_through_json() {
        for decision in [
            ResumeDecision::answer("42"),
            ResumeDecision::answer(ToolResultContent::error("nope")),
        ] {
            let json = serde_json::to_string(&decision).unwrap();
            let back: ResumeDecision = serde_json::from_str(&json).unwrap();
            assert_eq!(back, decision);
        }
    }

    #[test]
    fn suspension_round_trips_through_json() {
        let original = suspension();
        let json = serde_json::to_string(&original).unwrap();
        let back: Suspension = serde_json::from_str(&json).unwrap();
        assert_eq!(back, original);
    }
}
