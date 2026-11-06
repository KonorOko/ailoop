//! A failed run tells what went wrong at the provider through
//! `RunError::provider_kind`, without downcasting the adapter's error.

use std::time::Duration;

use ailoop::{
    CompactionError, Conversation, EngineError, ProviderErrorKind, RetryConfig, RetryingModel,
};
use ailoop_core::testing::{ScriptedError, ScriptedModel};

#[tokio::test]
async fn a_model_error_reports_its_kind() {
    let mut chat = Conversation::builder(ScriptedModel::with_turns([Err(ScriptedError::of(
        ProviderErrorKind::QuotaExhausted,
        None,
    ))]))
    .build()
    .expect("build");

    let err = chat.run("go").await.expect_err("quota");
    assert!(matches!(err.kind(), EngineError::Model(_)), "{err:?}");
    assert_eq!(err.provider_kind(), Some(ProviderErrorKind::QuotaExhausted));
}

#[tokio::test]
async fn retrying_model_retries_the_transient_markers() {
    let mut config = RetryConfig::default();
    config.max_attempts = 3;
    config.base_delay = Duration::from_millis(1);
    config.max_delay = Duration::from_millis(1);
    config.jitter = false;
    let model = RetryingModel::with_config(
        ScriptedModel::with_turns([
            Err(ScriptedError::of(ProviderErrorKind::Network, None)),
            Err(ScriptedError::of(
                ProviderErrorKind::RateLimited,
                Some(Duration::from_millis(1)),
            )),
            Err(ScriptedError::of(ProviderErrorKind::Unavailable, None)),
        ]),
        config,
    );
    let mut chat = Conversation::builder(model).build().expect("build");

    let err = chat.run("go").await.expect_err("every attempt fails");
    assert_eq!(
        err.provider_kind(),
        Some(ProviderErrorKind::Unavailable),
        "the last attempt's error surfaces"
    );
}

#[test]
fn engine_errors_without_a_model_error_have_no_kind() {
    let err: EngineError<ScriptedError> = EngineError::Suspended;
    assert_eq!(err.provider_kind(), None);

    // The summarizer's kind stays on the compaction error.
    let compaction = CompactionError::summarization_failed_from(ScriptedError::of(
        ProviderErrorKind::RateLimited,
        None,
    ));
    let err: EngineError<ScriptedError> = EngineError::Context(compaction);
    assert_eq!(err.provider_kind(), None);
    let EngineError::Context(compaction) = &err else {
        unreachable!()
    };
    assert_eq!(
        compaction.provider_kind(),
        Some(ProviderErrorKind::RateLimited)
    );
}
