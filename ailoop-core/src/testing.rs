//! Test helpers for `ailoop-core` consumers.
//!
//! `ScriptedModel` is the canonical mock `CompletionModel`: pre-load it
//! with a queue of "what to return on the Nth turn" entries and the
//! engine drives it just like a real provider. Each turn entry can be
//! either a successful list of chunks or a setup-time error (which the
//! engine will surface through `EngineError::Model`). This is enough to
//! exercise both the happy path and retryability of a future
//! `RetryingModel<M>` decorator.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream::{BoxStream, self};

use crate::request::ChatRequest;
use crate::stream::StreamChunk;
use crate::traits::CompletionModel;

/// Concrete error type used by `ScriptedModel`. Implements
/// `std::error::Error` so it satisfies the `CompletionModel::Error`
/// bound (`Send + Sync + 'static`) without pulling new deps into
/// `ailoop-core`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptedError(pub String);

impl std::fmt::Display for ScriptedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "scripted error: {}", self.0)
    }
}

impl std::error::Error for ScriptedError {}

/// One scripted "turn". `Ok(chunks)` is replayed to the engine on the
/// next `chat_stream` call. `Err(e)` causes the call itself to fail,
/// matching an HTTP-level error path.
pub type ScriptedTurn = Result<Vec<StreamChunk>, ScriptedError>;

/// Replays a queue of pre-canned turns. Each `chat_stream` call pops
/// the next entry. An exhausted queue yields an empty stream — that
/// terminates a run cleanly when the test does not care about the
/// final turn.
pub struct ScriptedModel {
    name: String,
    model: String,
    scripts: Mutex<VecDeque<ScriptedTurn>>,
}

impl ScriptedModel {
    /// Build with a sequence of successful turns. Equivalent to
    /// `with_turns(turns.into_iter().map(Ok))`.
    pub fn new<I>(turns: I) -> Self
    where
        I: IntoIterator<Item = Vec<StreamChunk>>,
    {
        Self::with_turns(turns.into_iter().map(Ok))
    }

    /// Build with explicit `Result` turns so callers can mix successful
    /// streams with setup-time errors.
    pub fn with_turns<I>(turns: I) -> Self
    where
        I: IntoIterator<Item = ScriptedTurn>,
    {
        Self {
            name: "scripted".into(),
            model: "scripted".into(),
            scripts: Mutex::new(turns.into_iter().collect()),
        }
    }

    /// Override the value returned by `name()` so a test can assert on
    /// telemetry that includes the provider name. Optional — defaults
    /// to `"scripted"`.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the value returned by `model()`. Defaults to
    /// `"scripted"`.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl CompletionModel for ScriptedModel {
    type Error = ScriptedError;

    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        let next = self.scripts.lock().unwrap().pop_front();
        match next {
            None => Ok(Box::pin(stream::empty())),
            Some(Err(e)) => Err(e),
            Some(Ok(chunks)) => Ok(Box::pin(stream::iter(chunks.into_iter().map(Ok)))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{FinishReason, Usage};
    use futures::StreamExt;

    #[tokio::test]
    async fn replays_chunks_in_order() {
        let model = ScriptedModel::new([vec![
            StreamChunk::TextDelta {
                delta: "hello".into(),
            },
            StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                usage: Usage::default(),
            },
        ]]);

        let stream = model
            .chat_stream(ChatRequest {
                messages: vec![],
                tools: None,
                system_prompt: None,
                max_tokens: 0,
                additional_params: None,
                temperature: None,
                top_p: None,
                top_k: None,
                stop_sequences: vec![],
            })
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 2);
    }

    #[tokio::test]
    async fn surfaces_setup_time_error_per_turn() {
        let model = ScriptedModel::with_turns([Err(ScriptedError("rate limited".into()))]);

        let result = model
            .chat_stream(ChatRequest {
                messages: vec![],
                tools: None,
                system_prompt: None,
                max_tokens: 0,
                additional_params: None,
                temperature: None,
                top_p: None,
                top_k: None,
                stop_sequences: vec![],
            })
            .await;
        match result {
            Err(ScriptedError(msg)) => assert_eq!(msg, "rate limited"),
            Ok(_) => panic!("expected error"),
        }
    }
}
