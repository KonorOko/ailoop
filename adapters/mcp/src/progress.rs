//! Routes MCP `notifications/progress` to the tool call that asked for
//! them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use ailoop_core::ToolProgress;
use ailoop_tools::ProgressSink;
use rmcp::ClientHandler;
use rmcp::model::{NumberOrString, ProgressNotificationParam, ProgressToken};
use rmcp::service::{MaybeSendFuture, NotificationContext, RoleClient};

/// The client side of an MCP connection: does what `()` does, plus
/// forwarding each progress notification to the [`ProgressSink`] of the
/// call whose `progressToken` it carries.
#[derive(Clone, Default)]
pub(crate) struct ProgressRouter {
    sinks: Arc<Mutex<HashMap<String, ProgressSink>>>,
}

impl ProgressRouter {
    /// Send updates for `token` to `sink` until the returned guard is
    /// dropped, which also covers a call whose future is dropped
    /// mid-flight (a cancelled tool call).
    pub(crate) fn register(&self, token: String, sink: ProgressSink) -> Registration {
        self.sinks().insert(token.clone(), sink);
        Registration {
            router: self.clone(),
            token,
        }
    }

    /// Forward `params` to its call; unknown tokens (a call that already
    /// finished, or a token we never sent) are dropped.
    pub(crate) fn route(&self, params: ProgressNotificationParam) {
        let token = token_key(&params.progress_token);
        let Some(sink) = self.sinks().get(&token).cloned() else {
            return;
        };
        let mut progress = ToolProgress::new().with_progress(params.progress);
        if let Some(total) = params.total {
            progress = progress.with_total(total);
        }
        if let Some(message) = params.message {
            progress = progress.with_message(message);
        }
        sink.report(progress);
    }

    fn sinks(&self) -> MutexGuard<'_, HashMap<String, ProgressSink>> {
        // One map call per operation, so a poisoned lock is still
        // consistent.
        self.sinks.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Keeps a token routed; unregisters it on drop.
pub(crate) struct Registration {
    router: ProgressRouter,
    token: String,
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.router.sinks().remove(&self.token);
    }
}

/// The map key for a progress token, string or number.
pub(crate) fn token_key(token: &ProgressToken) -> String {
    match &token.0 {
        NumberOrString::String(s) => s.to_string(),
        NumberOrString::Number(n) => n.to_string(),
    }
}

impl ClientHandler for ProgressRouter {
    fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) -> impl Future<Output = ()> + MaybeSendFuture + '_ {
        self.route(params);
        std::future::ready(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(token: &str, progress: f64) -> ProgressNotificationParam {
        ProgressNotificationParam::new(
            ProgressToken(NumberOrString::String(token.into())),
            progress,
        )
    }

    #[tokio::test]
    async fn routes_by_token_and_drops_unknown_ones() {
        let router = ProgressRouter::default();
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let registration = router.register("run:call_1".into(), ProgressSink::new("call_1", tx));

        let mut with_fields = param("run:call_1", 2.0);
        with_fields.total = Some(4.0);
        with_fields.message = Some("halfway".into());
        router.route(with_fields);
        router.route(param("run:other", 1.0));

        let (call_id, progress) = rx.try_recv().expect("routed");
        assert_eq!(call_id, "call_1");
        assert_eq!(
            progress,
            ToolProgress::new()
                .with_progress(2.0)
                .with_total(4.0)
                .with_message("halfway")
        );
        assert!(rx.try_recv().is_err(), "unknown token is dropped");

        drop(registration);
        router.route(param("run:call_1", 3.0));
        assert!(rx.try_recv().is_err(), "unregistered token is dropped");
    }

    #[test]
    fn numeric_tokens_have_a_key_too() {
        assert_eq!(token_key(&ProgressToken(NumberOrString::Number(7))), "7");
    }
}
