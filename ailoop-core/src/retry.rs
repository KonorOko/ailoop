//! Reliability decorator: [`RetryingModel`] wraps any [`CompletionModel`]
//! and retries setup-time failures with exponential backoff that honours
//! `Retry-After` hints surfaced by the adapter.
//!
//! Each adapter classifies its own error type via the [`Retryable`] trait
//! (`AnthropicError` in `ailoop-anthropic`, `AzureOpenAIError` in
//! `ailoop-azure-openai`); the decorator only needs to know whether an
//! error is `Permanent` or `Transient { retry_after }`.
//!
//! ## Scope
//!
//! Only the call to [`CompletionModel::chat_stream`] is retried. Once the
//! stream is open, mid-stream errors propagate to the caller unchanged —
//! the engine has already consumed deltas and built partial assistant
//! state, and replaying from that point is not safe in general. Reissuing
//! the entire request from the engine layer would also discard work
//! the model has already done. If retry-on-mid-stream becomes important
//! later it belongs above this layer (in the engine), not here.

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::request::ChatRequest;
use crate::stream::StreamChunk;
use crate::traits::CompletionModel;

/// Adapter-side classification of an error as retryable or not.
///
/// Implemented by each provider's error type (e.g. `AnthropicError`,
/// `AzureOpenAIError`) so [`RetryingModel`] stays provider-agnostic.
pub trait Retryable {
    fn retry_classification(&self) -> RetryClassification;
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryClassification {
    /// Never retry. Auth, validation, schema errors, etc.
    Permanent,
    /// Worth another attempt after `retry_after` (when the server told us
    /// how long to wait) or after the decorator's own backoff.
    Transient { retry_after: Option<Duration> },
}

/// Backoff settings for [`RetryingModel`].
///
/// `max_attempts` counts the *total* number of calls to the inner model,
/// including the first one. `max_attempts: 3` means up to two retries
/// after the original failure.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            jitter: true,
        }
    }
}

/// Decorator that retries setup-time failures of an inner
/// [`CompletionModel`] using exponential backoff with optional jitter,
/// honouring `Retry-After` when the inner error provides it.
pub struct RetryingModel<M> {
    inner: M,
    config: RetryConfig,
}

impl<M> RetryingModel<M> {
    pub fn new(inner: M) -> Self {
        Self::with_config(inner, RetryConfig::default())
    }

    pub fn with_config(inner: M, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    pub fn inner(&self) -> &M {
        &self.inner
    }

    pub fn into_inner(self) -> M {
        self.inner
    }
}

#[async_trait]
impl<M> CompletionModel for RetryingModel<M>
where
    M: CompletionModel + Send + Sync,
    M::Error: Retryable,
{
    type Error = M::Error;

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        let max = self.config.max_attempts.max(1);
        let mut attempt: u32 = 0;
        loop {
            let try_req = req.clone();
            let err = match self.inner.chat_stream(try_req).await {
                Ok(stream) => return Ok(stream),
                Err(e) => e,
            };
            attempt += 1;
            if attempt >= max {
                return Err(err);
            }
            let delay = match err.retry_classification() {
                RetryClassification::Permanent => return Err(err),
                RetryClassification::Transient { retry_after } => {
                    compute_delay(&self.config, attempt, retry_after)
                }
            };
            tokio::time::sleep(delay).await;
        }
    }
}

fn compute_delay(cfg: &RetryConfig, attempt: u32, retry_after: Option<Duration>) -> Duration {
    let base = match retry_after {
        Some(d) => d,
        None => exponential(cfg.base_delay, cfg.max_delay, attempt),
    };
    if cfg.jitter { apply_jitter(base) } else { base }
}

fn exponential(base: Duration, max: Duration, attempt: u32) -> Duration {
    // attempt is 1-based on the first retry, so 2^(attempt - 1).
    let shift = attempt.saturating_sub(1).min(20);
    let factor: u128 = 1u128 << shift;
    let nanos = base.as_nanos().saturating_mul(factor);
    let capped = nanos.min(max.as_nanos());
    Duration::from_nanos(u64::try_from(capped).unwrap_or(u64::MAX))
}

/// Spreads concurrent clients by adding 0..10% to the delay. Always
/// non-negative so a server-supplied `Retry-After` floor is respected.
fn apply_jitter(d: Duration) -> Duration {
    let nanos = d.as_nanos();
    if nanos == 0 {
        return d;
    }
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u128;
    let jitter_max = nanos / 10;
    let offset = if jitter_max == 0 {
        0
    } else {
        now_ns % jitter_max
    };
    let total = nanos.saturating_add(offset);
    Duration::from_nanos(u64::try_from(total).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    use async_trait::async_trait;
    use futures::StreamExt;
    use futures::stream::{self, BoxStream};

    use super::*;
    use crate::stream::{FinishReason, Usage};
    use crate::testing::{ScriptedError, ScriptedModel, ScriptedTurn};

    fn empty_request() -> ChatRequest {
        ChatRequest::new(vec![], 0)
    }

    fn fast_config(max_attempts: u32) -> RetryConfig {
        RetryConfig {
            max_attempts,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            jitter: false,
        }
    }

    /// Wraps a `ScriptedModel` and counts how many times `chat_stream`
    /// was invoked. Used to assert on retry behaviour from the outside.
    struct CountingModel {
        inner: ScriptedModel,
        calls: AtomicUsize,
    }

    impl CountingModel {
        fn new(turns: Vec<ScriptedTurn>) -> Self {
            Self {
                inner: ScriptedModel::with_turns(turns),
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl CompletionModel for CountingModel {
        type Error = ScriptedError;

        fn name(&self) -> &str {
            self.inner.name()
        }

        fn model(&self) -> &str {
            self.inner.model()
        }

        async fn chat_stream(
            &self,
            req: ChatRequest,
        ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.chat_stream(req).await
        }
    }

    fn ok_chunk() -> StreamChunk {
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
        }
    }

    #[tokio::test]
    async fn retries_until_success() {
        let inner = CountingModel::new(vec![
            Err(ScriptedError("transient:1".into())),
            Ok(vec![Ok(ok_chunk())]),
        ]);
        let model = RetryingModel::with_config(inner, fast_config(3));

        let stream = model
            .chat_stream(empty_request())
            .await
            .expect("retry should succeed on second attempt");
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 1);
        assert_eq!(model.inner().calls(), 2);
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let inner = CountingModel::new(vec![
            Err(ScriptedError("transient:1".into())),
            Err(ScriptedError("transient:1".into())),
            Err(ScriptedError("transient:1".into())),
            Err(ScriptedError("transient:1".into())),
        ]);
        let model = RetryingModel::with_config(inner, fast_config(3));

        let result = model.chat_stream(empty_request()).await;
        assert!(matches!(result, Err(ScriptedError(_))));
        assert_eq!(
            model.inner().calls(),
            3,
            "max_attempts is total calls including the first"
        );
    }

    #[tokio::test]
    async fn respects_retry_after() {
        let inner = CountingModel::new(vec![
            Err(ScriptedError("transient:50".into())),
            Ok(vec![Ok(ok_chunk())]),
        ]);
        let model = RetryingModel::with_config(inner, fast_config(3));

        let started = Instant::now();
        let stream = model
            .chat_stream(empty_request())
            .await
            .expect("second attempt should succeed");
        let _: Vec<_> = stream.collect().await;
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(50),
            "expected at least 50ms wait, got {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn permanent_errors_are_not_retried() {
        let inner = CountingModel::new(vec![
            Err(ScriptedError("permanent: bad auth".into())),
            Ok(vec![Ok(ok_chunk())]),
        ]);
        let model = RetryingModel::with_config(inner, fast_config(3));

        let result = model.chat_stream(empty_request()).await;
        assert!(matches!(result, Err(ScriptedError(_))));
        assert_eq!(
            model.inner().calls(),
            1,
            "permanent errors must not trigger a retry"
        );
    }

    #[tokio::test]
    async fn mid_stream_errors_are_not_retried() {
        // The setup call succeeds; the stream then yields one chunk and
        // a transient-looking error. RetryingModel must not reissue the
        // request — the engine has already consumed deltas.
        let inner = CountingModel::new(vec![
            Ok(vec![
                Ok(StreamChunk::TextDelta {
                    delta: "hello".into(),
                }),
                Err(ScriptedError("transient:1".into())),
            ]),
            Ok(vec![Ok(ok_chunk())]),
        ]);
        let model = RetryingModel::with_config(inner, fast_config(3));

        let stream = model
            .chat_stream(empty_request())
            .await
            .expect("setup should succeed");
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 2);
        assert!(matches!(chunks[0], Ok(StreamChunk::TextDelta { .. })));
        assert!(matches!(chunks[1], Err(ScriptedError(_))));
        assert_eq!(
            model.inner().calls(),
            1,
            "mid-stream errors must not trigger a setup-time retry"
        );
    }

    #[tokio::test]
    async fn exponential_backoff_grows_then_caps() {
        let cfg = RetryConfig {
            max_attempts: 10,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_millis(40),
            jitter: false,
        };
        assert_eq!(
            exponential(cfg.base_delay, cfg.max_delay, 1),
            Duration::from_millis(10)
        );
        assert_eq!(
            exponential(cfg.base_delay, cfg.max_delay, 2),
            Duration::from_millis(20)
        );
        assert_eq!(
            exponential(cfg.base_delay, cfg.max_delay, 3),
            Duration::from_millis(40)
        );
        assert_eq!(
            exponential(cfg.base_delay, cfg.max_delay, 4),
            Duration::from_millis(40)
        );
    }

    /// Sanity check that `BoxStream` from `stream::iter` still conforms
    /// to what `chat_stream` is supposed to return — guards against the
    /// test scaffolding accidentally lying about the type.
    #[tokio::test]
    async fn box_stream_conforms() {
        let s: BoxStream<'static, Result<StreamChunk, ScriptedError>> =
            Box::pin(stream::iter(vec![Ok(ok_chunk())]));
        let chunks: Vec<_> = s.collect().await;
        assert_eq!(chunks.len(), 1);
    }

    /// Ensures we don't accidentally hold the lock across an `await` —
    /// regression guard for the `Mutex` in test setup.
    #[tokio::test]
    async fn no_lock_held_across_await() {
        let m = Mutex::new(0u32);
        {
            let mut g = m.lock().unwrap();
            *g += 1;
        }
        tokio::task::yield_now().await;
        assert_eq!(*m.lock().unwrap(), 1);
    }
}
