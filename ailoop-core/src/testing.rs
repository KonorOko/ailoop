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
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};

use crate::provider_error::{Capability, ProviderError, ProviderErrorKind};
use crate::request::ChatRequest;
use crate::retry::{RetryClassification, Retryable};
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

impl ScriptedError {
    /// Build an error whose message is the marker for `kind`, so its
    /// [`ProviderError::kind`] is `kind` and its
    /// [`Retryable::retry_classification`] matches what a real adapter
    /// reports for that kind (see the [`ProviderError`] impl).
    ///
    /// `retry_after` is written as `:<ms>` and only kept for the
    /// transient kinds ([`Network`](ProviderErrorKind::Network),
    /// [`RateLimited`](ProviderErrorKind::RateLimited),
    /// [`Unavailable`](ProviderErrorKind::Unavailable) and
    /// [`Interrupted`](ProviderErrorKind::Interrupted)); the other
    /// kinds are permanent and ignore it.
    ///
    /// ```
    /// # use std::time::Duration;
    /// # use ailoop_core::{ProviderError, ProviderErrorKind, RetryClassification, Retryable};
    /// # use ailoop_core::testing::ScriptedError;
    /// let err = ScriptedError::of(ProviderErrorKind::RateLimited, Some(Duration::from_millis(90)));
    /// assert_eq!(err.0, "rate_limited:90");
    /// assert_eq!(err.kind(), ProviderErrorKind::RateLimited);
    /// assert_eq!(
    ///     err.retry_classification(),
    ///     RetryClassification::Transient { retry_after: Some(Duration::from_millis(90)) },
    /// );
    /// ```
    pub fn of(kind: ProviderErrorKind, retry_after: Option<Duration>) -> Self {
        let name = marker_name(kind);
        let message = match retry_after {
            Some(delay) if is_transient(kind) => format!("{name}:{}", delay.as_millis()),
            _ => name.to_string(),
        };
        Self(message)
    }

    /// The marker at the start of the message and the text after its
    /// `:`, when the message starts with one. A marker counts only as
    /// the whole message or followed by `:`, so `"network outage"` is
    /// not the `network` marker.
    fn marker(&self) -> Option<(ProviderErrorKind, &str)> {
        let (head, tail) = self.0.split_once(':').unwrap_or((self.0.as_str(), ""));
        let kind = match head {
            "network" => ProviderErrorKind::Network,
            "auth" => ProviderErrorKind::Auth,
            "permission" => ProviderErrorKind::Permission,
            "rate_limited" => ProviderErrorKind::RateLimited,
            "unavailable" => ProviderErrorKind::Unavailable,
            "quota" => ProviderErrorKind::QuotaExhausted,
            "context_overflow" => ProviderErrorKind::ContextOverflow,
            "request_too_large" => ProviderErrorKind::RequestTooLarge,
            "content_filter" => ProviderErrorKind::ContentFilter,
            "model_not_found" => ProviderErrorKind::ModelNotFound,
            "unsupported" => {
                let capability = match tail.split(':').next().unwrap_or("").trim() {
                    "images" => Capability::Images,
                    "tools" => Capability::Tools,
                    _ => return None,
                };
                ProviderErrorKind::Unsupported(capability)
            }
            "interrupted" => ProviderErrorKind::Interrupted,
            "invalid_request" => ProviderErrorKind::InvalidRequest,
            "other" => ProviderErrorKind::Other,
            _ => return None,
        };
        Some((kind, tail))
    }
}

fn marker_name(kind: ProviderErrorKind) -> &'static str {
    match kind {
        ProviderErrorKind::Network => "network",
        ProviderErrorKind::Auth => "auth",
        ProviderErrorKind::Permission => "permission",
        ProviderErrorKind::RateLimited => "rate_limited",
        ProviderErrorKind::Unavailable => "unavailable",
        ProviderErrorKind::QuotaExhausted => "quota",
        ProviderErrorKind::ContextOverflow => "context_overflow",
        ProviderErrorKind::RequestTooLarge => "request_too_large",
        ProviderErrorKind::ContentFilter => "content_filter",
        ProviderErrorKind::ModelNotFound => "model_not_found",
        ProviderErrorKind::Unsupported(Capability::Images) => "unsupported:images",
        ProviderErrorKind::Unsupported(Capability::Tools) => "unsupported:tools",
        ProviderErrorKind::Interrupted => "interrupted",
        ProviderErrorKind::InvalidRequest => "invalid_request",
        ProviderErrorKind::Other => "other",
    }
}

/// Kinds a real adapter reports as transient.
fn is_transient(kind: ProviderErrorKind) -> bool {
    matches!(
        kind,
        ProviderErrorKind::Network
            | ProviderErrorKind::RateLimited
            | ProviderErrorKind::Unavailable
            | ProviderErrorKind::Interrupted
    )
}

/// Leading milliseconds of `tail`, as in `"90"` or `"90: slow down"`.
fn leading_millis(tail: &str) -> Option<Duration> {
    let digits: String = tail
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u64>().ok().map(Duration::from_millis)
}

/// Test-only classification by message convention, so engine-level
/// recovery can be exercised without a real provider.
///
/// A message that starts with a marker, either alone or followed by
/// `:`, gets that marker's kind. Delays are in milliseconds.
///
/// | Marker | Kind |
/// |---|---|
/// | `network[:<ms>]` | [`Network`](ProviderErrorKind::Network) |
/// | `auth` | [`Auth`](ProviderErrorKind::Auth) |
/// | `permission` | [`Permission`](ProviderErrorKind::Permission) |
/// | `rate_limited[:<ms>]` | [`RateLimited`](ProviderErrorKind::RateLimited) |
/// | `unavailable[:<ms>]` | [`Unavailable`](ProviderErrorKind::Unavailable) |
/// | `quota` | [`QuotaExhausted`](ProviderErrorKind::QuotaExhausted) |
/// | `context_overflow` | [`ContextOverflow`](ProviderErrorKind::ContextOverflow) |
/// | `request_too_large` | [`RequestTooLarge`](ProviderErrorKind::RequestTooLarge) |
/// | `content_filter` | [`ContentFilter`](ProviderErrorKind::ContentFilter) |
/// | `model_not_found` | [`ModelNotFound`](ProviderErrorKind::ModelNotFound) |
/// | `unsupported:images`, `unsupported:tools` | [`Unsupported`](ProviderErrorKind::Unsupported) |
/// | `interrupted[:<ms>]` | [`Interrupted`](ProviderErrorKind::Interrupted) |
/// | `invalid_request` | [`InvalidRequest`](ProviderErrorKind::InvalidRequest) |
/// | `other` | [`Other`](ProviderErrorKind::Other) |
///
/// Any other message is [`Other`](ProviderErrorKind::Other), except
/// that one containing `"context_overflow"` anywhere is still a
/// context overflow, as before these markers existed.
/// [`ScriptedError::of`] writes the marker for a kind.
impl ProviderError for ScriptedError {
    fn kind(&self) -> ProviderErrorKind {
        match self.marker() {
            Some((kind, _)) => kind,
            None if self.0.contains("context_overflow") => ProviderErrorKind::ContextOverflow,
            None => ProviderErrorKind::Other,
        }
    }
}

/// Test-only classification by message convention so retry-aware
/// decorators can be exercised without a real provider.
///
/// A message that starts with a [`ProviderError`] marker is classified
/// like a real adapter classifies that kind: `network`,
/// `rate_limited`, `unavailable` and `interrupted` are transient, with
/// `retry_after` from their `:<ms>` suffix; every other marker is
/// permanent. Otherwise the older conventions apply:
///
/// - contains `"permanent"` → [`RetryClassification::Permanent`]
/// - contains `"transient:<ms>"` → transient with `retry_after = ms`
/// - contains `"transient"` → transient without a `retry_after`
/// - anything else → permanent (so unrelated `Err`s in tests don't loop)
impl Retryable for ScriptedError {
    fn retry_classification(&self) -> RetryClassification {
        if let Some((kind, tail)) = self.marker() {
            return if is_transient(kind) {
                RetryClassification::Transient {
                    retry_after: leading_millis(tail),
                }
            } else {
                RetryClassification::Permanent
            };
        }
        let s = self.0.as_str();
        if s.contains("permanent") {
            return RetryClassification::Permanent;
        }
        if let Some((_, after)) = s.split_once("transient") {
            let retry_after = after.strip_prefix(':').and_then(leading_millis);
            return RetryClassification::Transient { retry_after };
        }
        RetryClassification::Permanent
    }
}

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
        // Poisoning only means another thread panicked mid-update; the
        // queue itself is still consistent.
        let next = self
            .scripts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front();
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
                service_tier: None,
            },
        ]]);

        let stream = model
            .chat_stream(ChatRequest::new(vec![], 0))
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
            .chat_stream(ChatRequest::new(vec![], 0))
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

        let result = model.chat_stream(ChatRequest::new(vec![], 0)).await;
        match result {
            Err(ScriptedError(msg)) => assert_eq!(msg, "rate limited"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn scripted_error_is_retryable_by_convention() {
        assert_eq!(
            ScriptedError("permanent: bad auth".into()).retry_classification(),
            RetryClassification::Permanent,
        );
        assert_eq!(
            ScriptedError("transient".into()).retry_classification(),
            RetryClassification::Transient { retry_after: None },
        );
        assert_eq!(
            ScriptedError("transient:75 ms".into()).retry_classification(),
            RetryClassification::Transient {
                retry_after: Some(Duration::from_millis(75))
            },
        );
        assert_eq!(
            ScriptedError("something else entirely".into()).retry_classification(),
            RetryClassification::Permanent,
        );
    }

    #[test]
    fn markers_round_trip_through_of() {
        let kinds = [
            ProviderErrorKind::Network,
            ProviderErrorKind::Auth,
            ProviderErrorKind::Permission,
            ProviderErrorKind::RateLimited,
            ProviderErrorKind::Unavailable,
            ProviderErrorKind::QuotaExhausted,
            ProviderErrorKind::ContextOverflow,
            ProviderErrorKind::RequestTooLarge,
            ProviderErrorKind::ContentFilter,
            ProviderErrorKind::ModelNotFound,
            ProviderErrorKind::Unsupported(Capability::Images),
            ProviderErrorKind::Unsupported(Capability::Tools),
            ProviderErrorKind::Interrupted,
            ProviderErrorKind::InvalidRequest,
            ProviderErrorKind::Other,
        ];
        for kind in kinds {
            let err = ScriptedError::of(kind, None);
            assert_eq!(err.kind(), kind, "{err:?}");
            assert_eq!(
                err.is_context_overflow(),
                kind == ProviderErrorKind::ContextOverflow
            );
            let expected = if is_transient(kind) {
                RetryClassification::Transient { retry_after: None }
            } else {
                RetryClassification::Permanent
            };
            assert_eq!(err.retry_classification(), expected, "{err:?}");
        }
    }

    #[test]
    fn transient_markers_carry_their_delay_in_millis() {
        let delay = Some(Duration::from_millis(90));
        for kind in [
            ProviderErrorKind::Network,
            ProviderErrorKind::RateLimited,
            ProviderErrorKind::Unavailable,
            ProviderErrorKind::Interrupted,
        ] {
            let err = ScriptedError::of(kind, delay);
            assert_eq!(err.kind(), kind);
            assert_eq!(
                err.retry_classification(),
                RetryClassification::Transient { retry_after: delay },
            );
        }
        assert_eq!(
            ScriptedError("rate_limited:500: slow down".into()).retry_classification(),
            RetryClassification::Transient {
                retry_after: Some(Duration::from_millis(500))
            },
        );
    }

    #[test]
    fn permanent_kinds_ignore_a_delay() {
        let err = ScriptedError::of(
            ProviderErrorKind::QuotaExhausted,
            Some(Duration::from_secs(1)),
        );
        assert_eq!(err.0, "quota");
        assert_eq!(err.retry_classification(), RetryClassification::Permanent);
    }

    #[test]
    fn markers_only_count_at_the_start_as_a_whole_word() {
        // Would match two markers with `contains`.
        let err = ScriptedError("permanent network failure".into());
        assert_eq!(err.kind(), ProviderErrorKind::Other);
        assert_eq!(err.retry_classification(), RetryClassification::Permanent);

        for message in [
            "network outage",
            "networking",
            "rate limited",
            "upstream: network",
        ] {
            let err = ScriptedError(message.into());
            assert_eq!(err.kind(), ProviderErrorKind::Other, "{message}");
        }
        assert_eq!(
            ScriptedError("network: connection reset".into()).kind(),
            ProviderErrorKind::Network,
        );
        assert_eq!(
            ScriptedError("unsupported:audio".into()).kind(),
            ProviderErrorKind::Other,
        );
    }

    #[test]
    fn legacy_context_overflow_anywhere_still_counts() {
        let err = ScriptedError("summary failed: context_overflow".into());
        assert_eq!(err.kind(), ProviderErrorKind::ContextOverflow);
        assert!(err.is_context_overflow());
    }
}
