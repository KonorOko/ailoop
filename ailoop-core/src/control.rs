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
/// - A turn that suspends on a deferred tool call
///   ([`crate::FinishReason::Suspended`]) is not closed, so nothing is
///   delivered then. The queue stays with this handle: the run that
///   resumes the suspension delivers it only if it is given the same
///   handle, when it closes that turn.
/// - Until it is delivered, a queued message can be withdrawn with
///   [`Self::retract`] or edited in place with [`Self::replace`], both
///   by its id. Each takes the queue in one step, so a drain running at
///   the same time sees either the queue before the edit or the queue
///   after it, never a gap. Do not edit by [`Self::take_pending`]
///   followed by [`Self::restore`]: a drain between the two calls finds
///   the queue empty, and every other message waits one more turn.
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
    /// a run's stream is dropped after it took them. To withdraw or
    /// edit one queued message, use [`Self::retract`] or
    /// [`Self::replace`] instead of pairing this with
    /// [`Self::take_pending`]: see "Delivery" above.
    pub fn restore(&self, messages: Vec<Inbound>) {
        let mut inbox = self.lock();
        for message in messages.into_iter().rev() {
            inbox.push_front(message);
        }
    }

    /// Remove the queued message with this `id` and return it, leaving
    /// the rest of the queue in its order. With several queued under
    /// the same `id`, only the oldest is removed.
    ///
    /// Returns `None`, changing nothing, when no queued message has
    /// that `id`: it was already delivered (its
    /// [`crate::StreamChunk::InboundDelivered`] says so), already
    /// retracted, or never queued. It is atomic with respect to the
    /// engine draining the queue.
    pub fn retract(&self, id: &str) -> Option<Inbound> {
        let mut inbox = self.lock();
        let index = inbox.iter().position(|m| m.id == id)?;
        inbox.remove(index)
    }

    /// Swap the queued message with this `id` for `message`, in the same
    /// place in the queue. With several queued under the same `id`,
    /// only the oldest is replaced. `message` may carry a different id;
    /// its delivery reports that one.
    ///
    /// Returns `Err(message)`, changing nothing, when no queued message
    /// has that `id` (already delivered, retracted, or never queued),
    /// so the caller can decide what to do with it, such as
    /// [`Self::push`] it. It is atomic with respect to the engine
    /// draining the queue.
    pub fn replace(&self, id: &str, message: Inbound) -> Result<(), Inbound> {
        let mut inbox = self.lock();
        match inbox.iter_mut().find(|m| m.id == id) {
            Some(slot) => {
                *slot = message;
                Ok(())
            }
            None => Err(message),
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
        // updated (every operation changes it in one `VecDeque` call or
        // slot write), so a poisoned lock is safe to keep using.
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

    fn queue(control: &RunControl, ids: &[&str]) {
        for id in ids {
            control.push(Inbound::text(*id, *id));
        }
    }

    #[test]
    fn retract_removes_only_that_message_keeping_order() {
        let control = RunControl::new();
        queue(&control, &["a", "b", "c"]);
        assert_eq!(control.retract("b").unwrap().id, "b");
        assert_eq!(ids(&control.take_pending()), ["a", "c"]);
    }

    #[test]
    fn replace_keeps_the_message_position() {
        let control = RunControl::new();
        queue(&control, &["a", "b", "c"]);
        assert_eq!(control.replace("b", Inbound::text("b2", "edited")), Ok(()));
        let pending = control.take_pending();
        assert_eq!(ids(&pending), ["a", "b2", "c"]);
        assert_eq!(pending[1], Inbound::text("b2", "edited"));
    }

    #[test]
    fn retract_and_replace_on_absent_id_change_nothing() {
        let control = RunControl::new();
        queue(&control, &["a", "b"]);
        assert!(control.retract("b").is_some());
        assert_eq!(control.retract("b"), None);
        assert_eq!(control.retract("unknown"), None);
        let message = Inbound::text("x", "kept");
        assert_eq!(control.replace("b", message.clone()), Err(message));
        assert_eq!(ids(&control.take_pending()), ["a"]);
        assert_eq!(control.retract("a"), None, "drained ids are gone too");
    }

    #[test]
    fn repeated_ids_act_on_the_oldest() {
        let control = RunControl::new();
        control.push(Inbound::text("a", "a"));
        control.push(Inbound::text("x", "1"));
        control.push(Inbound::text("x", "2"));
        control.push(Inbound::text("x", "3"));
        assert_eq!(control.retract("x"), Some(Inbound::text("x", "1")));
        assert_eq!(control.replace("x", Inbound::text("y", "new")), Ok(()));
        assert_eq!(
            control.take_pending(),
            [
                Inbound::text("a", "a"),
                Inbound::text("y", "new"),
                Inbound::text("x", "3"),
            ]
        );
    }

    /// The case `take_pending` + `restore` got wrong: a drain racing an
    /// edit must take every other queued message in that one drain,
    /// never find the queue empty or partial and leave them a turn
    /// behind.
    #[test]
    fn retract_and_replace_are_atomic_against_a_concurrent_drain() {
        use std::sync::Barrier;
        use std::thread;

        for round in 0..5000 {
            let control = RunControl::new();
            queue(&control, &["o1", "t", "o2", "o3"]);
            let barrier = Arc::new(Barrier::new(2));
            let drainer = {
                let (control, barrier) = (control.clone(), barrier.clone());
                thread::spawn(move || {
                    barrier.wait();
                    // Aim the drain at the moment the edit touches the
                    // queue; bounded, since `replace` keeps its length.
                    let mut spins = 0;
                    while control.len() == 4 && spins < 10_000 {
                        spins += 1;
                        std::hint::spin_loop();
                    }
                    control.take_pending()
                })
            };
            barrier.wait();
            let edited = if round % 2 == 0 {
                control.retract("t").is_some()
            } else {
                control.replace("t", Inbound::text("t2", "t2")).is_ok()
            };
            let drained = drainer.join().unwrap();

            let expected: &[&str] = match (edited, round % 2 == 0) {
                (true, true) => &["o1", "o2", "o3"],
                (true, false) => &["o1", "t2", "o2", "o3"],
                (false, _) => &["o1", "t", "o2", "o3"],
            };
            assert_eq!(ids(&drained), expected, "round {round}");
            assert!(control.is_empty(), "round {round}: left behind");
        }
    }
}
