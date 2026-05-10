use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;

use crate::snapshot::ConversationSnapshot;

/// Persistence backend for conversation snapshots.
///
/// `save` writes the snapshot to whatever durable medium the impl
/// chooses; `load` returns the most recently persisted snapshot, or
/// `Ok(None)` when the store has never been written to.
#[async_trait]
pub trait HistoryStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn save(&self, snapshot: &ConversationSnapshot) -> Result<(), Self::Error>;
    async fn load(&self) -> Result<Option<ConversationSnapshot>, Self::Error>;
}

/// Volatile [`HistoryStore`] kept entirely in process memory. Intended
/// for tests and short-lived sessions where durable persistence is not
/// required.
#[derive(Default)]
pub struct InMemoryHistoryStore {
    inner: Mutex<Option<ConversationSnapshot>>,
}

impl InMemoryHistoryStore {
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
/// the file does not exist (first run).
pub struct JsonFileHistoryStore {
    path: PathBuf,
}

impl JsonFileHistoryStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JsonFileHistoryStoreError {
    #[error("failed to read/write snapshot file: {0}")]
    Io(#[from] std::io::Error),

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
