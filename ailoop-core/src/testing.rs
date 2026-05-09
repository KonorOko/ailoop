//! Test helpers for `ailoop-core` consumers.
//!
//! `ScriptedModel` is the canonical mock `CompletionModel`: pre-load it
//! with a queue of "what to return on the Nth turn" entries and the
//! engine drives it just like a real provider. Each turn entry can be
//! either a setup-time error (the call itself fails, matching HTTP-level
//! errors), or a list of per-chunk results — letting tests script
//! streams that mix `Ok` chunks with an `Err` mid-stream (e.g. an SSE
//! connection that drops after a few chunks). This is enough to
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

/// One scripted "turn". The outer `Result` distinguishes a setup-time
/// failure (the `chat_stream` call itself returns `Err`) from a stream
/// that was successfully opened. The inner `Vec<Result<_, _>>` is the
/// sequence of chunks the stream will yield, allowing an `Err` to
/// appear mid-stream after some `Ok` chunks have been delivered.
pub type ScriptedTurn = Result<Vec<Result<StreamChunk, ScriptedError>>, ScriptedError>;

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
    /// Build with a sequence of successful turns where every chunk is
    /// an `Ok`. Equivalent to wrapping each chunk in `Ok` and each turn
    /// in `Ok`, then forwarding to [`with_turns`](Self::with_turns).
    pub fn new<I>(turns: I) -> Self
    where
        I: IntoIterator<Item = Vec<StreamChunk>>,
    {
        Self::with_turns(
            turns
                .into_iter()
                .map(|chunks| Ok(chunks.into_iter().map(Ok).collect())),
        )
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
            Some(Ok(chunks)) => Ok(Box::pin(stream::iter(chunks))),
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
    async fn surfaces_mid_stream_error_after_chunks() {
        let model = ScriptedModel::with_turns([Ok(vec![
            Ok(StreamChunk::TextDelta {
                delta: "partial".into(),
            }),
            Err(ScriptedError("connection dropped".into())),
        ])]);

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
            .expect("chat_stream should open the stream");

        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 2, "expected one Ok chunk then one Err");
        assert!(matches!(chunks[0], Ok(StreamChunk::TextDelta { .. })));
        match &chunks[1] {
            Err(ScriptedError(msg)) => assert_eq!(msg, "connection dropped"),
            other => panic!("expected mid-stream Err, got {other:?}"),
        }
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
