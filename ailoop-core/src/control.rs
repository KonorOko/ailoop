//! Controlling a run from outside while it is in flight: see
//! [`RunControl`].

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

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
/// Today it carries an inbox: [`Self::push`] queues an [`Inbound`]
/// message from any task, and the engine delivers what is queued when
/// a turn closes. Pass it to a run with `RunConfig::control` (or
/// `RunOptions::control` on a `Conversation`). Cheap to clone; every
/// clone shares the same queue.
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

    fn lock(&self) -> MutexGuard<'_, VecDeque<Inbound>> {
        // A panic while holding the lock cannot leave the queue half
        // updated (every operation is one `VecDeque` call), so a
        // poisoned lock is safe to keep using.
        self.inbox.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl std::fmt::Debug for RunControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunControl")
            .field("queued", &self.len())
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
    fn restore_puts_messages_ahead_of_newer_ones() {
        let control = RunControl::new();
        control.push(Inbound::text("c", "3"));
        control.restore(vec![Inbound::text("a", "1"), Inbound::text("b", "2")]);
        assert_eq!(ids(&control.take_pending()), ["a", "b", "c"]);
    }
}
