//! [`CachingTokenProvider`]: TTL-bounded cache around any [`TokenProvider`].

use std::sync::RwLock;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::client::TokenProvider;
use crate::errors::AzureOpenAIError;

/// [`TokenProvider`] wrapper that memoizes the inner provider's token
/// for `ttl - refresh_skew` and triggers a fresh fetch once that window
/// elapses. Sized for Microsoft Entra ID / managed-identity flows where
/// the underlying credential pipeline does a network round-trip on every
/// uncached call and tokens live ~1h.
///
/// `refresh_skew` shaves the trailing edge of the TTL so callers refresh
/// before the upstream considers the token dead. A typical pairing for a
/// 1-hour token is `ttl = 1h, refresh_skew = 5min`, refreshing roughly
/// every 55 minutes. A `refresh_skew` greater than or equal to `ttl`
/// degrades to "refresh every call" — the live window saturates to zero.
///
/// The `TokenProvider` trait surface stays unchanged: the policy lives in
/// construction, not in the trait. Wrap once at startup and pass the
/// wrapper into [`AzureOpenAIClient::with_provider`](crate::AzureOpenAIClient::with_provider):
///
/// ```ignore
/// use std::time::Duration;
/// use ailoop_azure_openai::{AzureOpenAIClient, CachingTokenProvider};
///
/// let cached = CachingTokenProvider::new(
///     my_entra_provider,
///     Duration::from_secs(3600),
///     Duration::from_secs(300),
/// );
/// let client = AzureOpenAIClient::with_provider("https://x.openai.azure.com", cached);
/// ```
///
/// ## Concurrency
///
/// State is held in a `std::sync::RwLock<Option<(String, Instant)>>` so
/// the lock stays sync — `inner.token()` runs *outside* the lock to keep
/// it cheap and avoid holding any lock across an `await`. The trade-off:
/// when the cache expires and two requests race, both may call
/// `inner.token()` concurrently and the later write wins. The duplicate
/// fetch is accepted because the alternative (single-flight via an async
/// `Mutex` or `OnceCell`) would pull in a `tokio` runtime dependency the
/// adapter does not otherwise carry, and the duplicate cost is bounded
/// to one extra credential call per refresh window per concurrent burst.
///
/// Errors from `inner.token()` propagate unchanged and are *not* cached
/// — a transient credential outage does not poison the next call, and a
/// previously cached (now-expired) token is left untouched so subsequent
/// callers can retry the refresh.
pub struct CachingTokenProvider<P: TokenProvider> {
    inner: P,
    ttl: Duration,
    refresh_skew: Duration,
    cached: RwLock<Option<(String, Instant)>>,
}

impl<P: TokenProvider> CachingTokenProvider<P> {
    /// Wrap `inner` with a TTL-bounded cache. The cached token is served
    /// for `ttl - refresh_skew` after each successful fetch; the next
    /// call after that window invokes `inner.token()` again.
    pub fn new(inner: P, ttl: Duration, refresh_skew: Duration) -> Self {
        Self {
            inner,
            ttl,
            refresh_skew,
            cached: RwLock::new(None),
        }
    }

    fn fresh_until(&self, fetched_at: Instant) -> Instant {
        fetched_at + self.ttl.saturating_sub(self.refresh_skew)
    }
}

#[async_trait]
impl<P: TokenProvider> TokenProvider for CachingTokenProvider<P> {
    async fn token(&self) -> Result<String, AzureOpenAIError> {
        {
            let guard = self.cached.read().expect("cache lock poisoned");
            if let Some((token, fetched_at)) = guard.as_ref()
                && Instant::now() < self.fresh_until(*fetched_at)
            {
                return Ok(token.clone());
            }
        }

        let token = self.inner.token().await?;
        let now = Instant::now();
        *self.cached.write().expect("cache lock poisoned") = Some((token.clone(), now));
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::client::{AzureOpenAIAuth, AzureOpenAIClient};

    struct CountingProvider {
        calls: Arc<AtomicUsize>,
        token: String,
    }

    impl CountingProvider {
        fn new(token: impl Into<String>) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: Arc::clone(&calls),
                    token: token.into(),
                },
                calls,
            )
        }
    }

    #[async_trait]
    impl TokenProvider for CountingProvider {
        async fn token(&self) -> Result<String, AzureOpenAIError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.token.clone())
        }
    }

    struct FailingProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl TokenProvider for FailingProvider {
        async fn token(&self) -> Result<String, AzureOpenAIError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(AzureOpenAIError::Config("fetch failed".into()))
        }
    }

    struct StaticProvider(String);

    #[async_trait]
    impl TokenProvider for StaticProvider {
        async fn token(&self) -> Result<String, AzureOpenAIError> {
            Ok(self.0.clone())
        }
    }

    /// Second call within the live window returns the cached string and
    /// does not invoke the inner provider.
    #[tokio::test]
    async fn cache_hit_does_not_call_inner() {
        let (inner, calls) = CountingProvider::new("tok-a");
        let cache = CachingTokenProvider::new(inner, Duration::from_secs(60), Duration::ZERO);

        let first = cache.token().await.unwrap();
        let second = cache.token().await.unwrap();

        assert_eq!(first, "tok-a");
        assert_eq!(second, "tok-a");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// Once the TTL elapses, the next call refreshes from the inner
    /// provider rather than returning the stale token.
    #[tokio::test]
    async fn ttl_expiry_forces_refresh() {
        let (inner, calls) = CountingProvider::new("tok-b");
        let cache = CachingTokenProvider::new(inner, Duration::from_millis(40), Duration::ZERO);

        cache.token().await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        tokio::time::sleep(Duration::from_millis(80)).await;

        cache.token().await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// `refresh_skew` shortens the live window: with `ttl = 200ms` and
    /// `skew = 180ms` the cache is only fresh for ~20ms, so a wait of
    /// 60ms triggers a refresh that the no-skew control does not.
    #[tokio::test]
    async fn refresh_skew_advances_refresh() {
        let (skewed_inner, skewed_calls) = CountingProvider::new("tok-skew");
        let skewed = CachingTokenProvider::new(
            skewed_inner,
            Duration::from_millis(200),
            Duration::from_millis(180),
        );

        let (control_inner, control_calls) = CountingProvider::new("tok-control");
        let control =
            CachingTokenProvider::new(control_inner, Duration::from_millis(200), Duration::ZERO);

        skewed.token().await.unwrap();
        control.token().await.unwrap();
        assert_eq!(skewed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(control_calls.load(Ordering::SeqCst), 1);

        tokio::time::sleep(Duration::from_millis(60)).await;

        skewed.token().await.unwrap();
        control.token().await.unwrap();

        assert_eq!(
            skewed_calls.load(Ordering::SeqCst),
            2,
            "skew should have shortened the window past the 60ms wait",
        );
        assert_eq!(
            control_calls.load(Ordering::SeqCst),
            1,
            "control without skew should still be inside the 200ms window",
        );
    }

    /// Inner errors propagate verbatim and are not cached: a subsequent
    /// call re-invokes the inner provider rather than replaying the
    /// failure from memory.
    #[tokio::test]
    async fn inner_error_propagates_and_is_not_cached() {
        let calls = Arc::new(AtomicUsize::new(0));
        let cache = CachingTokenProvider::new(
            FailingProvider {
                calls: Arc::clone(&calls),
            },
            Duration::from_secs(60),
            Duration::ZERO,
        );

        match cache.token().await {
            Err(AzureOpenAIError::Config(msg)) => assert!(msg.contains("fetch failed")),
            other => panic!("expected Config error, got {other:?}"),
        }
        match cache.token().await {
            Err(AzureOpenAIError::Config(_)) => {}
            other => panic!("expected Config error, got {other:?}"),
        }

        assert_eq!(calls.load(Ordering::SeqCst), 2, "errors must not cache");
    }

    /// `CachingTokenProvider<StaticProvider>` satisfies the bound on
    /// [`AzureOpenAIClient::with_provider`] and lands in the `Provider`
    /// auth variant — the type stack used in production
    /// (`Arc<CachingTokenProvider<...>>` as `Arc<dyn TokenProvider>`)
    /// compiles and resolves correctly.
    #[test]
    fn caching_provider_plugs_into_with_provider() {
        let cached = CachingTokenProvider::new(
            StaticProvider("t".into()),
            Duration::from_secs(3600),
            Duration::from_secs(300),
        );
        let client = AzureOpenAIClient::with_provider("https://x.openai.azure.com", cached);
        assert!(matches!(client.auth, AzureOpenAIAuth::Provider(_)));

        let arc_cached: Arc<CachingTokenProvider<StaticProvider>> =
            Arc::new(CachingTokenProvider::new(
                StaticProvider("t".into()),
                Duration::from_secs(60),
                Duration::from_secs(5),
            ));
        let auth = AzureOpenAIAuth::Provider(arc_cached);
        assert!(matches!(auth, AzureOpenAIAuth::Provider(_)));
    }
}
