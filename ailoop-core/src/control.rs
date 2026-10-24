//! Controlling a run from outside while it is in flight: see
//! [`RunControl`].

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::background::BackgroundToolResult;
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
/// closes — per-call cancellation: [`Self::cancel_tool`] stops one
/// tool call and lets the run go on — and [`Self::background_tool`],
/// which moves a running call out of the run's way while the tool goes
/// on running (see [`crate::BackgroundConfig`] for its settings). Pass it to a
/// run with
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
    inbox: Arc<Mutex<Inbox>>,
    /// Token of each tool call of the step in flight, by `call_id`.
    /// When both are held, this lock is taken before `background`.
    calls: Arc<Mutex<HashMap<String, CancellationToken>>>,
    /// Calls to backgroundable tools that are running, moved or not.
    background: Arc<Mutex<BackgroundCalls>>,
}

/// The background calls a handle tracks, each under a key of its own:
/// a `call_id` alone is not enough, since a later run may reuse the id
/// of a call an earlier run moved and that is still running.
#[derive(Default)]
struct BackgroundCalls {
    /// The key the next call gets; keys only grow, so the newest call
    /// with a given `call_id` has the highest key.
    next: u64,
    calls: BTreeMap<u64, BackgroundCall>,
}

impl BackgroundCalls {
    /// Key of the newest tracked call with `call_id` that `matches`.
    fn newest(&self, call_id: &str, matches: impl Fn(&BackgroundState) -> bool) -> Option<u64> {
        self.calls
            .iter()
            .rev()
            .find(|(_, call)| call.call_id == call_id && matches(&call.state))
            .map(|(key, _)| *key)
    }
}

/// The queue and what the run in flight still takes besides it, under
/// one lock: deciding that a run takes no more background results and
/// draining what it took are one step.
#[derive(Default)]
struct Inbox {
    queue: VecDeque<Inbound>,
    /// The run taking background results, if any (see [`RunControl::open_run`]).
    open: Option<u64>,
    /// Real results of background calls, waiting for the open run's
    /// next drain.
    results: Vec<(Inbound, BackgroundToolResult)>,
}

/// A call to a backgroundable tool, tracked from when it starts until
/// the engine lets go of it (or, once moved, until it finishes).
struct BackgroundCall {
    call_id: String,
    /// The token its tool sees, not tied to the run's.
    token: CancellationToken,
    state: BackgroundState,
}

enum BackgroundState {
    /// Running, the engine waiting for it; the sender fires when it is
    /// moved.
    Waiting(oneshot::Sender<()>),
    /// Moved to the background and still running.
    Moved,
    /// Finished before it was moved; the engine takes its result.
    Done,
}

/// Mints the ids [`RunControl::open_run`] hands out.
static RUN_SEQ: AtomicU64 = AtomicU64::new(1);

/// How a background call's tool ended, seen from its task: see
/// [`RunControl::finish_background`].
///
/// Engine plumbing, hidden from the docs and not covered by semver.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundEnd {
    /// Not moved: the engine is waiting for its result.
    Engine,
    /// Moved: the task delivers its result.
    Moved,
    /// The engine gave up on it (abort, `cancel_tool`): nobody wants
    /// its result.
    Abandoned,
}

/// What a drain took: see [`RunControl::drain`].
///
/// Engine plumbing, hidden from the docs and not covered by semver.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct Drained {
    /// Queued messages, oldest first.
    pub messages: Vec<Inbound>,
    /// Background results, in arrival order, each with the message that
    /// delivers it.
    pub results: Vec<(Inbound, BackgroundToolResult)>,
}

impl RunControl {
    /// A handle with an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue `message` for delivery at the next turn close.
    pub fn push(&self, message: Inbound) {
        self.lock().queue.push_back(message);
    }

    /// Number of queued, undelivered messages. Background results
    /// waiting for the run are not counted: they are not in the queue.
    pub fn len(&self) -> usize {
        self.lock().queue.len()
    }

    /// Whether no message is queued.
    pub fn is_empty(&self) -> bool {
        self.lock().queue.is_empty()
    }

    /// Remove and return every queued message, oldest first. The engine
    /// uses this to drain the queue; a caller can use it to reclaim
    /// what a finished run did not deliver.
    pub fn take_pending(&self) -> Vec<Inbound> {
        self.lock().queue.drain(..).collect()
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
            inbox.queue.push_front(message);
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
        let index = inbox.queue.iter().position(|m| m.id == id)?;
        inbox.queue.remove(index)
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
        match inbox.queue.iter_mut().find(|m| m.id == id) {
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

    /// Move a running tool call to the background, by the `call_id` its
    /// `ToolCallStarted` chunk carried. The call is answered at once
    /// with a provisional result (`BackgroundConfig::notice`) and the
    /// step goes on without waiting for it, while the tool keeps
    /// running in a task of its own. Returns `false`, doing nothing,
    /// unless the call is to a tool that opts in
    /// (`ToolDyn::backgroundable`), is running (past its
    /// `on_before_tool_call*` hooks, tool invoked) and has no result
    /// yet; a call already moved, or already stopped with
    /// [`Self::cancel_tool`], returns `false` too.
    ///
    /// Once moved, the call is no longer the run's:
    ///
    /// - Ending, aborting or cancelling the run does not stop it, and
    ///   [`Self::cancel_tool`] returns `false` for it; stop it with
    ///   [`Self::cancel_background`]. The token its tool sees
    ///   (`ToolContext::cancellation`) fires only then.
    /// - Its real result goes through the run's
    ///   `on_after_tool_call_mut` hooks (so a `Sanitize` applies), then
    ///   to the run, as an [`Inbound`] whose id is the `call_id`, if the
    ///   run still takes it: delivered at the next turn close like any
    ///   queued message, ahead of them, with a
    ///   [`crate::StreamChunk::BackgroundToolFinished`]. A run takes
    ///   results until it ends, or until a turn that would end it
    ///   closes with nothing to deliver. Either way the result also
    ///   goes to `BackgroundConfig::on_result`, whose
    ///   [`BackgroundToolResult::delivered`] says whether the run took
    ///   it; a result is never left in a queue no run will drain.
    /// - Its progress updates are dropped, and usage it reports counts
    ///   towards the run only until the run finishes.
    ///
    /// The run's history keeps the provisional result as the call's
    /// tool result; the real one, if the run takes it, enters later as
    /// a user message. Nothing caps how many calls run in the
    /// background: that is the caller's to limit.
    pub fn background_tool(&self, call_id: &str) -> bool {
        // Holding `calls` throughout makes this exclusive with
        // `cancel_tool`: a call it already cancelled is not moved, and
        // a moved call leaves `calls` before `cancel_tool` can find it.
        let mut calls = self.calls();
        if calls
            .get(call_id)
            .is_some_and(CancellationToken::is_cancelled)
        {
            return false;
        }
        let moved = {
            let mut background = self.background();
            let key = background.newest(call_id, |state| {
                matches!(state, BackgroundState::Waiting(_))
            });
            match key.and_then(|key| background.calls.get_mut(&key)) {
                Some(call) => {
                    let BackgroundState::Waiting(moved) =
                        std::mem::replace(&mut call.state, BackgroundState::Moved)
                    else {
                        unreachable!("matched as waiting above");
                    };
                    // The engine may have stopped waiting already (its
                    // future dropped); its `release_background` then
                    // finds the call moved and leaves it running.
                    let _ = moved.send(());
                    true
                }
                None => false,
            }
        };
        if moved {
            calls.remove(call_id);
        }
        moved
    }

    /// Stop a call moved to the background with
    /// [`Self::background_tool`]: its token fires and its future is
    /// dropped, so it stops even if the tool ignores the token. Its
    /// result becomes an error saying it was cancelled and may have
    /// partly completed (with [`BackgroundToolResult::cancelled`] set),
    /// delivered like any background result. Returns `false`, doing
    /// nothing, when no moved call with that id is still running
    /// (unknown, not moved, or already finished). If several moved
    /// calls with that id are running (a provider reusing call ids
    /// across runs), it stops the most recently started one.
    pub fn cancel_background(&self, call_id: &str) -> bool {
        let background = self.background();
        let key = background.newest(call_id, |state| matches!(state, BackgroundState::Moved));
        match key.and_then(|key| background.calls.get(&key)) {
            Some(call) => {
                call.token.cancel();
                true
            }
            None => false,
        }
    }

    /// Track a running call to a backgroundable tool, whose tool sees
    /// `token`. Returns the key that [`Self::finish_background`] and
    /// [`Self::release_background`] take, and a receiver that fires if
    /// the call is moved.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn register_background(
        &self,
        call_id: String,
        token: CancellationToken,
    ) -> (u64, oneshot::Receiver<()>) {
        let (moved, on_moved) = oneshot::channel();
        let mut background = self.background();
        let key = background.next;
        background.next += 1;
        background.calls.insert(
            key,
            BackgroundCall {
                call_id,
                token,
                state: BackgroundState::Waiting(moved),
            },
        );
        (key, on_moved)
    }

    /// Called by a background call's task once its tool is done; says
    /// who takes its result. Exclusive with [`Self::background_tool`]:
    /// a call is either moved or finished first, never both.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn finish_background(&self, key: u64) -> BackgroundEnd {
        let mut background = self.background();
        match background.calls.get(&key).map(|call| &call.state) {
            Some(BackgroundState::Waiting(_)) => {
                // Kept until the engine releases it, so a release
                // finding no entry can only mean the call was moved.
                if let Some(call) = background.calls.get_mut(&key) {
                    call.state = BackgroundState::Done;
                }
                BackgroundEnd::Engine
            }
            Some(BackgroundState::Moved) => {
                background.calls.remove(&key);
                BackgroundEnd::Moved
            }
            Some(BackgroundState::Done) | None => BackgroundEnd::Abandoned,
        }
    }

    /// Called when the engine lets go of a call, whether it saw its
    /// result, saw it moved, or stopped waiting (abort, `cancel_tool`,
    /// a dropped stream). Returns whether the call was moved; if it was
    /// not, it is no longer tracked, and if it was still running its
    /// token fires (a finished call's token is left alone).
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn release_background(&self, key: u64) -> bool {
        let mut background = self.background();
        match background.calls.get(&key).map(|call| &call.state) {
            Some(BackgroundState::Moved) => true,
            // Moved, and already finished.
            None => true,
            Some(BackgroundState::Done) => {
                // Like any finished call, its token is left alone.
                background.calls.remove(&key);
                false
            }
            Some(BackgroundState::Waiting(_)) => {
                if let Some(call) = background.calls.remove(&key) {
                    call.token.cancel();
                }
                false
            }
        }
    }

    /// Start taking background results for a new run; returns the id
    /// that [`Self::offer_background_result`], [`Self::drain`] and
    /// [`Self::close_run`] take.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn open_run(&self) -> u64 {
        let run = RUN_SEQ.fetch_add(1, Ordering::Relaxed);
        self.lock().open = Some(run);
        run
    }

    /// Hand the real result of a background call to `run`, with the
    /// message that delivers it. Gives the result back when `run` no
    /// longer takes results.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn offer_background_result(
        &self,
        run: u64,
        message: Inbound,
        result: BackgroundToolResult,
    ) -> Option<BackgroundToolResult> {
        let mut inbox = self.lock();
        if inbox.open != Some(run) {
            return Some(result);
        }
        inbox.results.push((message, result));
        None
    }

    /// Take the queue and `run`'s background results in one step. With
    /// `close_if_empty`, a drain that finds neither also stops `run`
    /// taking results, in the same step, so a result arriving after it
    /// is refused rather than left behind.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn drain(&self, run: u64, close_if_empty: bool) -> Drained {
        let mut inbox = self.lock();
        let open = inbox.open == Some(run);
        let results = if open {
            std::mem::take(&mut inbox.results)
        } else {
            Vec::new()
        };
        let messages: Vec<Inbound> = inbox.queue.drain(..).collect();
        if open && close_if_empty && messages.is_empty() && results.is_empty() {
            inbox.open = None;
        }
        Drained { messages, results }
    }

    /// Stop `run` taking background results; returns those it took but
    /// never drained, for the caller to report as not delivered.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn close_run(&self, run: u64) -> Vec<BackgroundToolResult> {
        let mut inbox = self.lock();
        if inbox.open != Some(run) {
            return Vec::new();
        }
        inbox.open = None;
        std::mem::take(&mut inbox.results)
            .into_iter()
            .map(|(_, result)| result)
            .collect()
    }

    fn lock(&self) -> MutexGuard<'_, Inbox> {
        // A panic while holding the lock cannot leave the inbox half
        // updated (every operation changes each field in one call or
        // slot write), so a poisoned lock is safe to keep using.
        self.inbox.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn calls(&self) -> MutexGuard<'_, HashMap<String, CancellationToken>> {
        // Same reasoning as `lock`: one map call per operation.
        self.calls.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn background(&self) -> MutexGuard<'_, BackgroundCalls> {
        // Same reasoning as `lock`.
        self.background.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl std::fmt::Debug for RunControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunControl")
            .field("queued", &self.len())
            .field("calls", &self.calls().len())
            .field("background", &self.background().calls.len())
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

    fn background_result(call_id: &str) -> BackgroundToolResult {
        BackgroundToolResult::new(
            crate::RunId::new(),
            crate::StepId::new(),
            call_id,
            "job",
            serde_json::Value::Null,
            crate::ToolResultContent::text("done"),
        )
    }

    #[test]
    fn a_call_is_either_moved_or_finished_first() {
        let control = RunControl::new();
        // Moved first: the task delivers, the engine sees it moved.
        let token = CancellationToken::new();
        let (key, mut on_moved) = control.register_background("a".into(), token.clone());
        assert!(!control.cancel_background("a"), "not moved yet");
        assert!(control.background_tool("a"));
        assert!(on_moved.try_recv().is_ok());
        assert!(!control.background_tool("a"));
        assert!(control.release_background(key), "the engine sees it moved");
        assert!(!token.is_cancelled());
        assert!(control.cancel_background("a"));
        assert!(token.is_cancelled());
        assert_eq!(control.finish_background(key), BackgroundEnd::Moved);
        assert!(!control.cancel_background("a"), "finished");
        assert!(control.release_background(key), "moved and finished");

        // Finished first: the engine takes the result, a move is refused.
        let token = CancellationToken::new();
        let (key, _on_moved) = control.register_background("b".into(), token.clone());
        assert_eq!(control.finish_background(key), BackgroundEnd::Engine);
        assert!(!control.background_tool("b"));
        assert!(!control.release_background(key));
        assert!(
            !token.is_cancelled(),
            "a finished call's token is left alone"
        );

        // Let go of while running: its token fires, its task finds no taker.
        let token = CancellationToken::new();
        let (key, _on_moved) = control.register_background("c".into(), token.clone());
        assert!(!control.release_background(key));
        assert!(token.is_cancelled());
        assert_eq!(control.finish_background(key), BackgroundEnd::Abandoned);
        assert!(!control.background_tool("c"));
    }

    #[test]
    fn calls_sharing_an_id_are_tracked_apart() {
        let control = RunControl::new();
        // An earlier run's call, moved and still running.
        let old_token = CancellationToken::new();
        let (old, _on_moved) = control.register_background("t1".into(), old_token.clone());
        assert!(control.background_tool("t1"));
        assert!(control.release_background(old));
        // A later run reuses the id.
        let new_token = CancellationToken::new();
        let (new, _on_moved) = control.register_background("t1".into(), new_token.clone());
        assert_ne!(old, new);
        assert_eq!(
            control.finish_background(old),
            BackgroundEnd::Moved,
            "the moved call still delivers its own result"
        );
        assert_eq!(control.finish_background(new), BackgroundEnd::Engine);
        assert!(!control.release_background(new));

        // Both moved: `cancel_background` stops the newest.
        let (first, _on_moved) = control.register_background("t2".into(), old_token.clone());
        assert!(control.background_tool("t2"));
        let (_, _on_moved) = control.register_background("t2".into(), new_token.clone());
        assert!(control.background_tool("t2"));
        assert!(control.cancel_background("t2"));
        assert!(new_token.is_cancelled() && !old_token.is_cancelled());
        assert_eq!(control.finish_background(first), BackgroundEnd::Moved);
    }

    #[test]
    fn moving_a_call_makes_cancel_tool_refuse_it() {
        let control = RunControl::new();
        control.register_call("a".into(), CancellationToken::new());
        let _on_moved = control.register_background("a".into(), CancellationToken::new());
        assert!(control.background_tool("a"));
        assert!(!control.cancel_tool("a"));
    }

    #[test]
    fn cancelling_a_call_makes_background_tool_refuse_it() {
        let control = RunControl::new();
        control.register_call("a".into(), CancellationToken::new());
        let _on_moved = control.register_background("a".into(), CancellationToken::new());
        assert!(control.cancel_tool("a"));
        assert!(!control.background_tool("a"));
    }

    #[test]
    fn background_results_go_to_the_open_run_only() {
        let control = RunControl::new();
        let run = control.open_run();
        assert!(
            control
                .offer_background_result(
                    run + 1000,
                    Inbound::text("x", "x"),
                    background_result("x")
                )
                .is_some()
        );
        assert!(
            control
                .offer_background_result(run, Inbound::text("a", "a"), background_result("a"))
                .is_none()
        );
        control.push(Inbound::text("m", "m"));
        assert_eq!(control.len(), 1, "results are not queued messages");
        assert_eq!(ids(&control.take_pending()), ["m"]);
        // A drain that takes something keeps the run open.
        let drained = control.drain(run, true);
        assert_eq!(drained.results.len(), 1);
        assert!(
            control
                .offer_background_result(run, Inbound::text("b", "b"), background_result("b"))
                .is_none()
        );
        // Closing hands back what was not drained.
        let left = control.close_run(run);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].call_id, "b");
        assert!(
            control
                .offer_background_result(run, Inbound::text("c", "c"), background_result("c"))
                .is_some()
        );
        assert!(control.close_run(run).is_empty());
        // An empty closing drain stops taking results.
        let run = control.open_run();
        assert!(control.drain(run, true).results.is_empty());
        assert!(
            control
                .offer_background_result(run, Inbound::text("d", "d"), background_result("d"))
                .is_some()
        );
    }

    /// A result offered while the run's last drain closes it is either
    /// in that drain or refused: never left where nothing will take it.
    #[test]
    fn closing_drain_and_offer_never_strand_a_result() {
        use std::sync::Barrier;
        use std::thread;

        let (mut drained_count, mut refused_count) = (0, 0);
        for round in 0..5000 {
            let control = RunControl::new();
            let run = control.open_run();
            let barrier = Arc::new(Barrier::new(2));
            let offerer = {
                let (control, barrier) = (control.clone(), barrier.clone());
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..(round % 50) {
                        std::hint::spin_loop();
                    }
                    control.offer_background_result(
                        run,
                        Inbound::text("a", "a"),
                        background_result("a"),
                    )
                })
            };
            barrier.wait();
            let drained = control.drain(run, true);
            let refused = offerer.join().unwrap();
            let left = control.close_run(run);
            assert!(
                left.is_empty(),
                "round {round}: stranded after the closing drain"
            );
            match (drained.results.len(), refused.is_some()) {
                (1, false) => drained_count += 1,
                (0, true) => refused_count += 1,
                other => panic!("round {round}: {other:?}"),
            }
        }
        assert!(
            drained_count > 0 && refused_count > 0,
            "the race was never exercised"
        );
    }
}
