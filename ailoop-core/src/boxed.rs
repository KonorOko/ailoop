//! Type erasure for [`CompletionModel`]: [`BoxedModel`] hides the
//! adapter's concrete model and error types behind one type, so code
//! that picks a provider at runtime can store any of them in the same
//! field, `Vec` or `HashMap`.

use std::any::Any;
use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;

use crate::provider_error::{ProviderError, ProviderErrorKind};
use crate::request::ChatRequest;
use crate::retry::{RetryClassification, Retryable};
use crate::stream::StreamChunk;
use crate::traits::CompletionModel;

/// A [`CompletionModel`] whose concrete type is erased.
///
/// Wraps any model whose error implements [`ProviderError`] and
/// [`Retryable`], which every adapter in the workspace does. The model
/// lives behind an [`Arc`], so `clone()` is a reference count bump and
/// clones share the same model.
///
/// Errors come out as [`BoxedError`], which still answers
/// [`ProviderError::kind`] and [`Retryable::retry_classification`]
/// the way the original error did. A `BoxedModel` therefore composes
/// with [`RetryingModel`](crate::RetryingModel),
/// `SummarizeStrategy` and the engine exactly like the model it wraps.
///
/// ```
/// use std::collections::HashMap;
/// use ailoop_core::{BoxedModel, CompletionModel, ProviderError, Retryable};
///
/// fn register<M>(models: &mut HashMap<String, BoxedModel>, model: M)
/// where
///     M: CompletionModel + 'static,
///     M::Error: ProviderError + Retryable,
/// {
///     let model = BoxedModel::new(model);
///     models.insert(model.model().to_string(), model);
/// }
/// ```
///
/// Wrapping a `BoxedModel` again returns it unchanged rather than
/// adding a second layer.
#[derive(Clone)]
pub struct BoxedModel {
    inner: Arc<dyn ErasedModel>,
}

impl BoxedModel {
    /// Erase the type of `model`.
    pub fn new<M>(model: M) -> Self
    where
        M: CompletionModel + 'static,
        M::Error: ProviderError + Retryable,
    {
        let any: Box<dyn Any> = Box::new(model);
        match any.downcast::<BoxedModel>() {
            Ok(boxed) => *boxed,
            Err(any) => {
                let model = *any
                    .downcast::<M>()
                    .expect("the box holds the model just put in it");
                Self {
                    inner: Arc::new(model),
                }
            }
        }
    }
}

impl fmt::Debug for BoxedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoxedModel")
            .field("name", &self.inner.erased_name())
            .field("model", &self.inner.erased_model())
            .finish()
    }
}

#[async_trait]
impl CompletionModel for BoxedModel {
    type Error = BoxedError;

    fn name(&self) -> &str {
        self.inner.erased_name()
    }

    fn model(&self) -> &str {
        self.inner.erased_model()
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        self.inner.chat_stream_erased(req).await
    }
}

/// The object-safe form of [`CompletionModel`] with the error type
/// fixed to [`BoxedError`].
#[async_trait]
trait ErasedModel: Send + Sync {
    fn erased_name(&self) -> &str;

    fn erased_model(&self) -> &str;

    async fn chat_stream_erased(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, BoxedError>>, BoxedError>;
}

#[async_trait]
impl<M> ErasedModel for M
where
    M: CompletionModel,
    M::Error: ProviderError + Retryable,
{
    fn erased_name(&self) -> &str {
        self.name()
    }

    fn erased_model(&self) -> &str {
        self.model()
    }

    async fn chat_stream_erased(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, BoxedError>>, BoxedError> {
        let stream = self.chat_stream(req).await.map_err(BoxedError::new)?;
        Ok(stream.map(|chunk| chunk.map_err(BoxedError::new)).boxed())
    }
}

/// The error of a [`BoxedModel`]: the wrapped model's error, with its
/// type erased.
///
/// [`kind`](ProviderError::kind) and
/// [`retry_classification`](Retryable::retry_classification) are read
/// once from the original error when it is wrapped and returned as they
/// were. `Display` shows the original error's text, and
/// [`source`](StdError::source) returns the original error itself, so
/// it can be downcast to the adapter's type:
///
/// ```
/// # use ailoop_core::{BoxedError, ProviderError, ProviderErrorKind, RetryClassification, Retryable};
/// #[derive(Debug)]
/// struct Throttled;
/// # impl std::fmt::Display for Throttled {
/// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("throttled") }
/// # }
/// # impl std::error::Error for Throttled {}
/// impl ProviderError for Throttled {
///     fn kind(&self) -> ProviderErrorKind { ProviderErrorKind::RateLimited }
/// }
/// impl Retryable for Throttled {
///     fn retry_classification(&self) -> RetryClassification {
///         RetryClassification::Transient { retry_after: None }
///     }
/// }
///
/// let err = BoxedError::new(Throttled);
/// assert_eq!(err.kind(), ProviderErrorKind::RateLimited);
/// assert_eq!(err.to_string(), "throttled");
/// assert!(err.downcast_ref::<Throttled>().is_some());
/// ```
///
/// A caller that walks the [`source`](StdError::source) chain sees
/// the original error's text twice, once for the `BoxedError` and once
/// for the error it wraps.
pub struct BoxedError {
    inner: Box<dyn StdError + Send + Sync + 'static>,
    kind: ProviderErrorKind,
    retry: RetryClassification,
}

impl BoxedError {
    /// Wrap `error`, reading its kind and retry classification now.
    /// Wrapping a `BoxedError` returns it unchanged.
    pub fn new<E>(error: E) -> Self
    where
        E: StdError + ProviderError + Retryable + Send + Sync + 'static,
    {
        let any: Box<dyn Any> = Box::new(error);
        match any.downcast::<BoxedError>() {
            Ok(boxed) => *boxed,
            Err(any) => {
                let error = *any
                    .downcast::<E>()
                    .expect("the box holds the error just put in it");
                Self {
                    kind: error.kind(),
                    retry: error.retry_classification(),
                    inner: Box::new(error),
                }
            }
        }
    }

    /// The original error, when it is an `E`.
    pub fn downcast_ref<E: StdError + 'static>(&self) -> Option<&E> {
        self.inner.downcast_ref::<E>()
    }

    /// The original error, when it is an `E`; otherwise `self` back.
    pub fn downcast<E: StdError + 'static>(self) -> Result<E, Self> {
        let Self { inner, kind, retry } = self;
        match inner.downcast::<E>() {
            Ok(error) => Ok(*error),
            Err(inner) => Err(Self { inner, kind, retry }),
        }
    }

    /// The original error, as a trait object.
    pub fn into_inner(self) -> Box<dyn StdError + Send + Sync + 'static> {
        self.inner
    }
}

impl fmt::Debug for BoxedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoxedError")
            .field("inner", &self.inner)
            .field("kind", &self.kind)
            .field("retry", &self.retry)
            .finish()
    }
}

impl fmt::Display for BoxedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl StdError for BoxedError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.inner.as_ref())
    }
}

impl ProviderError for BoxedError {
    fn kind(&self) -> ProviderErrorKind {
        self.kind
    }
}

impl Retryable for BoxedError {
    fn retry_classification(&self) -> RetryClassification {
        self.retry.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::retry::{RetryConfig, RetryingModel};
    use crate::stream::{FinishReason, Usage};
    use crate::testing::{ScriptedError, ScriptedModel, ScriptedTurn};

    fn ok_chunk() -> StreamChunk {
        StreamChunk::TurnFinished {
            reason: FinishReason::EndTurn,
            usage: Usage::default(),
            service_tier: None,
            model: None,
        }
    }

    fn fast_config(max_attempts: u32) -> RetryConfig {
        RetryConfig {
            max_attempts,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            jitter: false,
        }
    }

    /// Counts calls through a shared counter, so the count stays
    /// readable after the model is moved into a `BoxedModel`.
    struct CountingModel {
        inner: ScriptedModel,
        calls: Arc<AtomicUsize>,
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

    fn counting(turns: Vec<ScriptedTurn>) -> (CountingModel, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let model = CountingModel {
            inner: ScriptedModel::with_turns(turns),
            calls: calls.clone(),
        };
        (model, calls)
    }

    /// Runs `turns` through `RetryingModel<M>` and
    /// `RetryingModel<BoxedModel>` and returns, for each, how many calls
    /// reached the model and the error text, if any.
    async fn retry_both(
        turns: fn() -> Vec<ScriptedTurn>,
        max_attempts: u32,
    ) -> [(usize, Option<String>); 2] {
        let (model, calls) = counting(turns());
        let typed = RetryingModel::with_config(model, fast_config(max_attempts));
        let typed_err = typed
            .chat_stream(ChatRequest::new(vec![], 0))
            .await
            .err()
            .map(|e| e.to_string());
        let typed_calls = calls.load(Ordering::SeqCst);

        let (model, calls) = counting(turns());
        let boxed = RetryingModel::with_config(BoxedModel::new(model), fast_config(max_attempts));
        let boxed_err = boxed
            .chat_stream(ChatRequest::new(vec![], 0))
            .await
            .err()
            .map(|e| e.to_string());
        let boxed_calls = calls.load(Ordering::SeqCst);

        [(typed_calls, typed_err), (boxed_calls, boxed_err)]
    }

    #[tokio::test]
    async fn retrying_a_boxed_model_retries_transient_errors_like_the_typed_one() {
        let [typed, boxed] = retry_both(
            || {
                vec![
                    Err(ScriptedError::of(ProviderErrorKind::RateLimited, None)),
                    Err(ScriptedError("transient:1".into())),
                    Ok(vec![Ok(ok_chunk())]),
                ]
            },
            3,
        )
        .await;
        assert_eq!(typed, (3, None));
        assert_eq!(boxed, typed);
    }

    #[tokio::test]
    async fn retrying_a_boxed_model_gives_up_like_the_typed_one() {
        let [typed, boxed] = retry_both(
            || {
                vec![
                    Err(ScriptedError::of(ProviderErrorKind::Unavailable, None)),
                    Err(ScriptedError::of(ProviderErrorKind::Network, None)),
                    Err(ScriptedError::of(ProviderErrorKind::Network, None)),
                ]
            },
            2,
        )
        .await;
        assert_eq!(typed, (2, Some("scripted error: network".into())));
        assert_eq!(boxed, typed);
    }

    #[tokio::test]
    async fn retrying_a_boxed_model_does_not_retry_permanent_errors() {
        let [typed, boxed] = retry_both(
            || {
                vec![
                    Err(ScriptedError::of(ProviderErrorKind::ContextOverflow, None)),
                    Ok(vec![Ok(ok_chunk())]),
                ]
            },
            3,
        )
        .await;
        assert_eq!(typed, (1, Some("scripted error: context_overflow".into())));
        assert_eq!(boxed, typed);
    }

    #[tokio::test]
    async fn retrying_a_boxed_model_honours_retry_after() {
        let (model, calls) = counting(vec![
            Err(ScriptedError::of(
                ProviderErrorKind::RateLimited,
                Some(Duration::from_millis(50)),
            )),
            Ok(vec![Ok(ok_chunk())]),
        ]);
        let model = RetryingModel::with_config(BoxedModel::new(model), fast_config(3));
        let started = std::time::Instant::now();
        let _stream = model
            .chat_stream(ChatRequest::new(vec![], 0))
            .await
            .expect("second attempt succeeds");
        assert!(started.elapsed() >= Duration::from_millis(50));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn mid_stream_errors_keep_their_kind_and_source() {
        let model = BoxedModel::new(ScriptedModel::with_turns([Ok(vec![
            Ok(StreamChunk::TextDelta { delta: "hi".into() }),
            Err(ScriptedError::of(ProviderErrorKind::Interrupted, None)),
        ])]));
        let chunks: Vec<_> = model
            .chat_stream(ChatRequest::new(vec![], 0))
            .await
            .expect("stream opens")
            .collect()
            .await;
        assert!(matches!(chunks[0], Ok(StreamChunk::TextDelta { .. })));
        let err = chunks[1].as_ref().expect_err("second chunk fails");
        assert_eq!(err.kind(), ProviderErrorKind::Interrupted);
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient { retry_after: None }
        );
        let source = err
            .source()
            .expect("source")
            .downcast_ref::<ScriptedError>();
        assert_eq!(source, Some(&ScriptedError("interrupted".into())));
    }

    #[test]
    fn boxed_error_reads_the_original_once() {
        let err = BoxedError::new(ScriptedError::of(
            ProviderErrorKind::Unavailable,
            Some(Duration::from_millis(7)),
        ));
        assert_eq!(err.kind(), ProviderErrorKind::Unavailable);
        assert!(!err.is_context_overflow());
        assert_eq!(
            err.retry_classification(),
            RetryClassification::Transient {
                retry_after: Some(Duration::from_millis(7))
            }
        );
        assert_eq!(err.to_string(), "scripted error: unavailable:7");

        let overflow = BoxedError::new(ScriptedError::of(ProviderErrorKind::ContextOverflow, None));
        assert!(overflow.is_context_overflow());
        assert_eq!(
            overflow.retry_classification(),
            RetryClassification::Permanent
        );
    }

    #[test]
    fn boxed_error_downcasts_to_the_original() {
        let err = BoxedError::new(ScriptedError("auth".into()));
        assert_eq!(
            err.downcast_ref::<ScriptedError>(),
            Some(&ScriptedError("auth".into()))
        );
        let err = err
            .downcast::<std::fmt::Error>()
            .expect_err("not a fmt::Error");
        assert_eq!(err.kind(), ProviderErrorKind::Auth);
        assert_eq!(
            err.downcast::<ScriptedError>().ok(),
            Some(ScriptedError("auth".into()))
        );
    }

    #[test]
    fn wrapping_twice_adds_no_layer() {
        let err = BoxedError::new(BoxedError::new(ScriptedError("quota".into())));
        assert!(err.downcast_ref::<ScriptedError>().is_some());
        assert_eq!(err.kind(), ProviderErrorKind::QuotaExhausted);

        let model = BoxedModel::new(ScriptedModel::new([]).with_model("m"));
        let again = BoxedModel::new(model.clone());
        assert!(Arc::ptr_eq(&model.inner, &again.inner));
    }

    #[test]
    fn clones_share_the_model() {
        let model = BoxedModel::new(ScriptedModel::new([]).with_name("n").with_model("m"));
        let clone = model.clone();
        assert!(Arc::ptr_eq(&model.inner, &clone.inner));
        assert_eq!(clone.name(), "n");
        assert_eq!(clone.model(), "m");
        assert_eq!(
            format!("{clone:?}"),
            r#"BoxedModel { name: "n", model: "m" }"#
        );
    }
}
