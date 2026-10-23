//! Controlling a run from outside while it is in flight: see
//! [`RunControl`].

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use tokio_util::sync::CancellationToken;

use crate::message::UserBlock;

/// A message from outside the run, delivered into it between turns
/// through a [`RunControl`].
///
/// `id` is the caller's own: the engine never interprets it, only
/// reports it back on [`crate::StreamChunk::InboundDelivered`] so the
/// caller can tell which messages entered the run and at which step.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Inbound {
    /// Caller-assigned id, reported back on delivery.
    pub id: String,
    /// Blocks appended to the run's next user message, in order.
    pub blocks: Vec<UserBlock>,
}

impl Inbound {
    /// A message made of `blocks`.
    pub fn new(id: impl Into<String>, blocks: Vec<UserBlock>) -> Self {
        Self {
            id: id.into(),
            blocks,
        }
    }

    /// A message made of one text block.
    pub fn text(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self::new(id, vec![UserBlock::text(text)])
    }
}

/// Handle for controlling a run from outside while it is in flight.
///
/// It carries an inbox — [`Self::push`] queues an [`Inbound`] message
/// from any task, and the engine delivers what is queued when a turn
/// closes — and per-call cancellation: [`Self::cancel_tool`] stops one
/// tool call and lets the run go on. Pass it to a run with
/// `RunConfig::control` (or `RunOptions::control` on a
/// `Conversation`). Cheap to clone; every clone shares the same state.
///
/// # Delivery
///
/// - The engine drains the queue only when a turn that did not abort
///   closes, after every tool call of that turn has finished (so
///   never mid-tool or while an approval is pending). The blocks go,
///   in arrival order, after the turn's tool results in the same user
///   message.
/// - If the turn did not end in tool calls and something was
///   delivered, the run continues instead of finishing, without
///   consulting `ChatMiddleware::on_turn_end` for that turn. The
///   continuation counts against `RunConfig::max_iterations`.
/// - A message queued before the run starts is delivered when the
///   first turn closes, not in the first request; put it in the
///   kickoff message if it must be there.
/// - Each delivery emits [`crate::StreamChunk::InboundDelivered`]. An
///   [`Inbound`] with no blocks is reported there but does not, by
///   itself, keep the run going.
///
/// # What is never lost
///
/// The queue belongs to the caller; the engine only takes from it.
/// What is still queued when the run ends stays queued (read it back
/// with [`Self::take_pending`], or hand the handle to the next run).
/// Delivered messages are part of the run's messages, so they are in
/// `new_messages` on `RunFinished`, aborts included. If the run's
/// stream is dropped before the run ends, the messages it took are put
/// back at the front of the queue. If the run fails with an error they
/// are not: they are in the error's partial messages, like every other
/// message of the run.
///
/// Delivery is in memory and best-effort across process restarts: a
/// caller that must not lose a message keeps its own copy until it
/// sees the matching `InboundDelivered`.
///
/// One run at a time should use a given handle; two runs sharing one
/// would split the queue between them.
#[derive(Clone, Default)]
pub struct RunControl {
    inbox: Arc<Mutex<VecDeque<Inbound>>>,
    /// Token of each tool call of the step in flight, by `call_id`.
    calls: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl RunControl {
    /// A handle with an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue `message` for delivery at the next turn close.
    pub fn push(&self, message: Inbound) {
        self.lock().push_back(message);
    }

    /// Number of queued, undelivered messages.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no message is queued.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// Remove and return every queued message, oldest first. The engine
    /// uses this to drain the queue; a caller can use it to reclaim
    /// what a finished run did not deliver.
    pub fn take_pending(&self) -> Vec<Inbound> {
        self.lock().drain(..).collect()
    }

    /// Put `messages` back at the front of the queue, keeping their
    /// order, ahead of anything queued since. The engine uses this when
    /// a run's stream is dropped after it took them.
    pub fn restore(&self, messages: Vec<Inbound>) {
        let mut inbox = self.lock();
        for message in messages.into_iter().rev() {
            inbox.push_front(message);
        }
    }

    /// Cancel one tool call of the step in flight, by the `call_id`
    /// its `ToolCallStarted` chunk carried. The run is not aborted: the
    /// call gets an error result and the run goes on. Returns `false`,
    /// doing nothing, when no call of the step in flight has that id
    /// (unknown, already answered, or from a run that ended).
    ///
    /// What the model sees depends on when the cancel lands:
    ///
    /// - Before the call's turn comes, the tool is never invoked and no
    ///   tool hook fires for it; its result reads
    ///   `"Tool not run: cancelled by the caller"`.
    /// - While its `on_before_tool_call*` hooks run (an approval being
    ///   asked, say), those hooks are not interrupted. If they let the
    ///   call through, the tool is not invoked and gets the same
    ///   result; the `on_after_tool_call*` hooks fire with it, since
    ///   the `before` hooks did. A `Skip` or `Terminate` from them
    ///   still wins.
    /// - While the tool runs, its [`CancellationToken`] fires and the
    ///   engine drops the tool future, so the call stops even if the
    ///   tool ignores the token. The result reads `"Tool cancelled by
    ///   the caller while running; it may have partly completed"`: the
    ///   tool may already have had its effect. The `after` hooks fire
    ///   with that result.
    pub fn cancel_tool(&self, call_id: &str) -> bool {
        match self.calls().get(call_id) {
            Some(token) => {
                token.cancel();
                true
            }
            None => false,
        }
    }

    /// Make `call_id` cancellable through [`Self::cancel_tool`].
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn register_call(&self, call_id: String, token: CancellationToken) {
        self.calls().insert(call_id, token);
    }

    /// Stop tracking `call_id` once it has its result.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn unregister_call(&self, call_id: &str) {
        self.calls().remove(call_id);
    }

    /// Stop tracking every call, when a run ends.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn clear_calls(&self) {
        self.calls().clear();
    }

    fn lock(&self) -> MutexGuard<'_, VecDeque<Inbound>> {
        // A panic while holding the lock cannot leave the queue half
        // updated (every operation is one `VecDeque` call), so a
        // poisoned lock is safe to keep using.
        self.inbox.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn calls(&self) -> MutexGuard<'_, HashMap<String, CancellationToken>> {
        // Same reasoning as `lock`: one map call per operation.
        self.calls.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl std::fmt::Debug for RunControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunControl")
            .field("queued", &self.len())
            .field("calls", &self.calls().len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(messages: &[Inbound]) -> Vec<&str> {
        messages.iter().map(|m| m.id.as_str()).collect()
    }

    #[test]
    fn take_pending_drains_in_arrival_order() {
        let control = RunControl::new();
        control.push(Inbound::text("a", "1"));
        control.clone().push(Inbound::text("b", "2"));
        assert_eq!(control.len(), 2);
        assert_eq!(ids(&control.take_pending()), ["a", "b"]);
        assert!(control.is_empty());
    }

    #[test]
    fn cancel_tool_fires_only_the_registered_token() {
        let control = RunControl::new();
        let (a, b) = (CancellationToken::new(), CancellationToken::new());
        control.register_call("a".into(), a.clone());
        control.register_call("b".into(), b.clone());
        assert!(control.cancel_tool("a"));
        assert!(a.is_cancelled() && !b.is_cancelled());
        assert!(!control.cancel_tool("unknown"));
        control.unregister_call("b");
        assert!(!control.cancel_tool("b"));
        control.clear_calls();
        assert!(!control.cancel_tool("a"));
    }

    #[test]
    fn restore_puts_messages_ahead_of_newer_ones() {
        let control = RunControl::new();
        control.push(Inbound::text("c", "3"));
        control.restore(vec![Inbound::text("a", "1"), Inbound::text("b", "2")]);
        assert_eq!(ids(&control.take_pending()), ["a", "b", "c"]);
    }
}
