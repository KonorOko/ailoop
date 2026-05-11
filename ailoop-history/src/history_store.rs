//! [`HistoryStore`] trait + [`InMemoryHistoryStore`] /
//! [`JsonFileHistoryStore`] backends.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;

use crate::snapshot::ConversationSnapshot;

/// Persistence backend for conversation snapshots.
///
/// Implement this trait to durable-back a conversation in any store
/// you like (Redis, Postgres, S3, sled, …). The crate ships
/// [`InMemoryHistoryStore`] for tests and [`JsonFileHistoryStore`] for
/// single-file local persistence.
///
/// ## Concurrency
///
/// The trait is `async`; impls are free to use any backend. The
/// façade [`Conversation`](https://docs.rs/ailoop) does not serialize
/// access to a single store across runs — a process running multiple
/// conversations against the same backend should use one store per
/// `RunId` (or implement its own locking). A single `Conversation`
/// only writes from one task at a time, so per-conversation impls do
/// not need internal mutexes for that reason alone.
#[async_trait]
pub trait HistoryStore: Send + Sync {
    /// Error type surfaced from [`save`](Self::save) /
    /// [`load`](Self::load). Use [`std::convert::Infallible`] when the
    /// implementation cannot fail (see [`InMemoryHistoryStore`]).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Persist `snapshot` so a later [`load`](Self::load) returns it.
    /// Implementations should treat this as an overwrite — the façade
    /// passes the full updated [`ConversationSnapshot`] every time
    /// rather than diffing.
    async fn save(&self, snapshot: &ConversationSnapshot) -> Result<(), Self::Error>;

    /// Return the most recently persisted snapshot, or `Ok(None)` when
    /// the store has never been written to. Returning `Ok(None)` lets
    /// callers default-init the conversation on first run instead of
    /// special-casing a missing payload.
    async fn load(&self) -> Result<Option<ConversationSnapshot>, Self::Error>;
}

/// Volatile [`HistoryStore`] kept entirely in process memory. Intended
/// for tests and short-lived sessions where durable persistence is not
/// required.
///
/// # Panics
///
/// `save` / `load` `.expect()` on an internal [`Mutex`] — they only
/// panic if the lock is poisoned by a previous panic in another
/// task that held the guard.
#[derive(Default)]
pub struct InMemoryHistoryStore {
    inner: Mutex<Option<ConversationSnapshot>>,
}

impl InMemoryHistoryStore {
    /// Build an empty store. Equivalent to
    /// [`InMemoryHistoryStore::default`].
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl HistoryStore for InMemoryHistoryStore {
    type Error = std::convert::Infallible;

    async fn save(&self, snapshot: &ConversationSnapshot) -> Result<(), Self::Error> {
        *self
            .inner
            .lock()
            .expect("InMemoryHistoryStore mutex poisoned") = Some(snapshot.clone());
        Ok(())
    }

    async fn load(&self) -> Result<Option<ConversationSnapshot>, Self::Error> {
        Ok(self
            .inner
            .lock()
            .expect("InMemoryHistoryStore mutex poisoned")
            .clone())
    }
}

/// JSON-backed [`HistoryStore`] that writes the full snapshot to a
/// single file. `save` pretty-prints; `load` returns `Ok(None)` when
/// the file does not exist (first run) so callers can seed an empty
/// conversation without branching on the missing-file case.
pub struct JsonFileHistoryStore {
    path: PathBuf,
}

impl JsonFileHistoryStore {
    /// Bind the store to `path`. The file is not opened until
    /// [`save`](HistoryStore::save) or [`load`](HistoryStore::load)
    /// runs; constructing the store cannot fail.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Path the store reads from / writes to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// [`HistoryStore::Error`] variant for [`JsonFileHistoryStore`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JsonFileHistoryStoreError {
    /// Filesystem I/O failed (permission denied, disk full, EOF mid-
    /// read, …). Wraps the underlying [`std::io::Error`].
    #[error("failed to read/write snapshot file: {0}")]
    Io(#[from] std::io::Error),

    /// JSON encode/decode failed. Wraps the underlying
    /// [`serde_json::Error`] — the most common cause is a snapshot
    /// file written by a future incompatible version (e.g. an
    /// unsupported [`ConversationSnapshot::VERSION`]) or hand-edited
    /// to break the [`Message`](ailoop_core::Message) shape.
    ///
    /// [`ConversationSnapshot::VERSION`]: crate::ConversationSnapshot::VERSION
    #[error("failed to encode/decode snapshot JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[async_trait]
impl HistoryStore for JsonFileHistoryStore {
    type Error = JsonFileHistoryStoreError;

    async fn save(&self, snapshot: &ConversationSnapshot) -> Result<(), Self::Error> {
        let bytes = serde_json::to_vec_pretty(snapshot)?;
        tokio::fs::write(&self.path, bytes).await?;
        Ok(())
    }

    async fn load(&self) -> Result<Option<ConversationSnapshot>, Self::Error> {
        match tokio::fs::read(&self.path).await {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::Message;
    use tempfile::tempdir;

    fn sample_snapshot() -> ConversationSnapshot {
        ConversationSnapshot::new(
            vec![Message::user("hi"), Message::assistant_text("hello")],
            vec![true, false],
        )
        .expect("valid lengths")
    }

    #[tokio::test]
    async fn in_memory_store_round_trip() {
        let store = InMemoryHistoryStore::new();
        assert!(store.load().await.unwrap().is_none());

        let snap = sample_snapshot();
        store.save(&snap).await.unwrap();
        let restored = store.load().await.unwrap().expect("load after save");
        assert_eq!(restored, snap);
    }

    #[tokio::test]
    async fn json_file_store_returns_none_when_missing() {
        let dir = tempdir().unwrap();
        let store = JsonFileHistoryStore::new(dir.path().join("missing.json"));
        let restored = store.load().await.unwrap();
        assert!(restored.is_none(), "missing file must yield Ok(None)");
    }

    #[tokio::test]
    async fn json_file_store_round_trip() {
        let dir = tempdir().unwrap();
        let store = JsonFileHistoryStore::new(dir.path().join("snap.json"));
        let snap = sample_snapshot();
        store.save(&snap).await.unwrap();
        let restored = store.load().await.unwrap().expect("load after save");
        assert_eq!(restored, snap);
    }
}
