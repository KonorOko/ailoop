//! Per-dispatch context handed to every [`Tool`](crate::Tool) /
//! [`ToolDyn`](crate::ToolDyn) call.
//!
//! [`ToolContext`] carries the [`RunId`] and [`StepId`] of the current
//! dispatch plus a [`ToolActivation`] handle into the per-run active
//! tool set. Handlers that need to mutate which tools are visible on
//! the next turn (deferred-tools / dynamic tool loading patterns) call
//! the activation methods directly — no shared `Arc<Mutex<...>>`
//! plumbing on the user side, no middleware to filter `req.tools`.
//!
//! The handle is per-run, so two concurrent runs do not share state.

use std::sync::{Arc, Mutex};

use ailoop_core::{RunId, StepId, ToolDefinition};
use indexmap::{IndexMap, IndexSet};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::registry::ToolDyn;

/// Failures from [`ToolActivation`] mutations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToolActivationError {
    /// Tool is not in the catalog. Either the name is misspelled or
    /// it was never registered with the [`ToolRegistry`](crate::ToolRegistry).
    #[error("Tool '{0}' is not registered")]
    NotFound(String),

    /// The handle is detached from any registry — typical for
    /// [`ToolContext::detached`] used in standalone tests. Activation
    /// is a no-op concept without a backing registry.
    #[error("ToolActivation is detached and cannot mutate any registry")]
    Detached,
}

/// Context delivered to a tool handler on every dispatch.
///
/// Tools that don't need it ignore the parameter; tools that do need
/// it (typically a meta-tool like `search_tools`) reach the per-run
/// active set through [`Self::tools`] to enable / disable other tools
/// on the next turn.
#[derive(Clone)]
pub struct ToolContext {
    run_id: RunId,
    step_id: StepId,
    activation: ToolActivation,
    cancellation: CancellationToken,
}

impl ToolContext {
    /// Construct a context bound to a real run. Used by the engine on
    /// each tool dispatch — tests and standalone callers want
    /// [`Self::detached`] instead.
    pub fn new(
        run_id: RunId,
        step_id: StepId,
        activation: ToolActivation,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            run_id,
            step_id,
            activation,
            cancellation,
        }
    }

    /// Build a detached context with synthetic identifiers, a no-op
    /// activation handle, and a fresh never-cancelled token. Use for
    /// standalone
    /// [`ToolRegistry::tool_call`](crate::ToolRegistry::tool_call)
    /// invocations and unit tests where no engine is in the loop.
    pub fn detached() -> Self {
        Self {
            run_id: RunId::new(),
            step_id: StepId::new(),
            activation: ToolActivation::detached(),
            cancellation: CancellationToken::new(),
        }
    }

    /// `RunId` of the run this dispatch belongs to.
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// `StepId` of the step this dispatch belongs to.
    pub fn step_id(&self) -> &StepId {
        &self.step_id
    }

    /// Handle into the per-run active tool set.
    pub fn tools(&self) -> &ToolActivation {
        &self.activation
    }

    /// Cancellation token tied to the enclosing run.
    ///
    /// Mirrors [`RunConfig::cancellation`][rc]: firing the token the
    /// caller supplied to the engine triggers `cancelled()` here.
    /// When the run was started without a token, the engine hands the
    /// tool a fresh never-cancelled handle so the getter never returns
    /// `None` — `await`ing `cancelled()` simply pends forever.
    ///
    /// The engine already wraps each tool call in a `select!` against
    /// the run-wide abort signal, so on cancellation the tool future
    /// is dropped — that already cancels any in-flight async I/O
    /// (reqwest, async DB drivers, etc.). Reach for this token only
    /// when drop-cancellation isn't enough:
    ///
    /// - [`tokio::task::spawn_blocking`] — the blocking task survives
    ///   its parent future. Check `token.is_cancelled()` between
    ///   iterations or `select!` on `cancelled()` in async code that
    ///   spawned it.
    /// - Sub-processes via [`tokio::process::Command::spawn`] — drop
    ///   does not kill the child; send SIGTERM/SIGKILL explicitly when
    ///   the token fires.
    /// - Ordered cleanup — flush buffers, close DB transactions, log
    ///   "aborted by caller" before the future is dropped.
    /// - Fan-out with [`tokio::task::JoinSet`] — clone
    ///   `token.child_token()` into each spawned task so all of them
    ///   stop when the run cancels, without one child cancelling the
    ///   whole run.
    ///
    /// Note: the run-wide abort future also fires on
    /// [`RunConfig::timeout`][to], but the timeout is not propagated
    /// into this token. Tools that need to react to a timeout
    /// cooperatively should wrap the relevant work in
    /// [`tokio::time::timeout`] themselves.
    ///
    /// The token is cheap to clone (`Arc` internally).
    ///
    /// [rc]: ailoop_core::RunConfig::cancellation
    /// [to]: ailoop_core::RunConfig::timeout
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }
}

/// Per-run handle to the active tool set.
///
/// Reads (`list_*`, `is_active`) are cheap snapshots; mutations
/// (`activate`, `deactivate`) take a brief lock. The handle is
/// `Clone` and shareable across tasks within a run.
///
/// A handle obtained via [`Self::detached`] has no catalog and no
/// active set — every read returns empty and every mutation returns
/// [`ToolActivationError::Detached`].
#[derive(Clone)]
pub struct ToolActivation {
    inner: Option<ToolActivationInner>,
}

#[derive(Clone)]
struct ToolActivationInner {
    catalog: Arc<IndexMap<String, Arc<dyn ToolDyn>>>,
    active: Arc<Mutex<IndexSet<String>>>,
}

impl ToolActivation {
    /// Construct a handle backed by a shared catalog and a per-run
    /// active set. Used by the engine; downstream callers rarely
    /// build this themselves.
    pub fn new(
        catalog: Arc<IndexMap<String, Arc<dyn ToolDyn>>>,
        active: Arc<Mutex<IndexSet<String>>>,
    ) -> Self {
        Self {
            inner: Some(ToolActivationInner { catalog, active }),
        }
    }

    /// Detached handle with no backing registry. All mutations fail
    /// with [`ToolActivationError::Detached`]; reads return empty.
    pub fn detached() -> Self {
        Self { inner: None }
    }

    /// Whether `name` is currently active. `false` for detached
    /// handles or tools that aren't registered.
    pub fn is_active(&self, name: &str) -> bool {
        let Some(inner) = &self.inner else {
            return false;
        };
        inner
            .active
            .lock()
            .expect("ToolActivation lock")
            .contains(name)
    }

    /// Add `name` to the active set. Returns
    /// [`ToolActivationError::NotFound`] if the tool was never
    /// registered, [`ToolActivationError::Detached`] for detached
    /// handles. Idempotent — activating an already-active tool is a
    /// no-op success.
    pub fn activate(&self, name: &str) -> Result<(), ToolActivationError> {
        let inner = self.inner.as_ref().ok_or(ToolActivationError::Detached)?;
        if !inner.catalog.contains_key(name) {
            return Err(ToolActivationError::NotFound(name.to_string()));
        }
        inner
            .active
            .lock()
            .expect("ToolActivation lock")
            .insert(name.to_string());
        Ok(())
    }

    /// Remove `name` from the active set. Idempotent — silently
    /// no-ops for unknown names (consistent with
    /// [`ToolRegistry::deactivate_tool`](crate::ToolRegistry::deactivate_tool)).
    /// Returns [`ToolActivationError::Detached`] for detached handles.
    pub fn deactivate(&self, name: &str) -> Result<(), ToolActivationError> {
        let inner = self.inner.as_ref().ok_or(ToolActivationError::Detached)?;
        inner
            .active
            .lock()
            .expect("ToolActivation lock")
            .shift_remove(name);
        Ok(())
    }

    /// Snapshot of currently active tool definitions, in registration
    /// order. The model would see these on the next turn.
    pub fn list_active(&self) -> Vec<ToolDefinition> {
        let Some(inner) = &self.inner else {
            return Vec::new();
        };
        let active = inner.active.lock().expect("ToolActivation lock");
        inner
            .catalog
            .iter()
            .filter(|(name, _)| active.contains(*name))
            .map(|(_, tool)| tool.tool_definition())
            .collect()
    }

    /// Snapshot of registered-but-inactive tool definitions, in
    /// registration order. The complement of [`Self::list_active`] —
    /// the natural input for a `search_tools` meta-tool that wants
    /// to surface tools the model has not yet been shown.
    pub fn list_inactive(&self) -> Vec<ToolDefinition> {
        let Some(inner) = &self.inner else {
            return Vec::new();
        };
        let active = inner.active.lock().expect("ToolActivation lock");
        inner
            .catalog
            .iter()
            .filter(|(name, _)| !active.contains(*name))
            .map(|(_, tool)| tool.tool_definition())
            .collect()
    }

    /// Snapshot of every registered tool definition (active +
    /// inactive), in registration order.
    pub fn list_all(&self) -> Vec<ToolDefinition> {
        let Some(inner) = &self.inner else {
            return Vec::new();
        };
        inner
            .catalog
            .values()
            .map(|tool| tool.tool_definition())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{ToolDefinition, ToolResultContent};
    use serde_json::json;

    struct StubTool {
        name: &'static str,
    }

    #[async_trait::async_trait]
    impl ToolDyn for StubTool {
        fn name(&self) -> String {
            self.name.into()
        }
        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                self.name,
                "stub",
                json!({"type":"object","properties":{},"required":[]}),
                vec![],
            )
        }
        async fn call(&self, _args: serde_json::Value, _ctx: &ToolContext) -> ToolResultContent {
            ToolResultContent::text("")
        }
    }

    fn build_handle(active_now: &[&str], all: &[&str]) -> ToolActivation {
        let mut catalog: IndexMap<String, Arc<dyn ToolDyn>> = IndexMap::new();
        for &n in all {
            catalog.insert(n.into(), Arc::new(StubTool { name: leak(n) }));
        }
        let active: IndexSet<String> = active_now.iter().map(|s| (*s).into()).collect();
        ToolActivation::new(Arc::new(catalog), Arc::new(Mutex::new(active)))
    }

    fn leak(s: &str) -> &'static str {
        Box::leak(s.to_string().into_boxed_str())
    }

    fn names(defs: Vec<ToolDefinition>) -> Vec<String> {
        defs.into_iter().map(|d| d.name).collect()
    }

    #[test]
    fn detached_reads_return_empty_and_mutations_error() {
        let h = ToolActivation::detached();
        assert!(h.list_active().is_empty());
        assert!(h.list_inactive().is_empty());
        assert!(h.list_all().is_empty());
        assert!(!h.is_active("anything"));
        assert!(matches!(
            h.activate("anything"),
            Err(ToolActivationError::Detached)
        ));
        assert!(matches!(
            h.deactivate("anything"),
            Err(ToolActivationError::Detached)
        ));
    }

    #[test]
    fn activate_unknown_tool_errors_with_notfound() {
        let h = build_handle(&[], &["foo"]);
        match h.activate("bar") {
            Err(ToolActivationError::NotFound(n)) => assert_eq!(n, "bar"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn activate_then_list_active_returns_in_registration_order() {
        let h = build_handle(&[], &["alpha", "beta", "gamma"]);
        h.activate("gamma").unwrap();
        h.activate("alpha").unwrap();
        assert_eq!(names(h.list_active()), vec!["alpha", "gamma"]);
        assert_eq!(names(h.list_inactive()), vec!["beta"]);
    }

    #[test]
    fn deactivate_is_idempotent_for_unknown_names() {
        let h = build_handle(&["foo"], &["foo"]);
        h.deactivate("never-existed").unwrap();
        assert_eq!(names(h.list_active()), vec!["foo"]);
    }

    #[test]
    fn list_all_returns_full_catalog_regardless_of_active() {
        let h = build_handle(&["foo"], &["foo", "bar", "baz"]);
        assert_eq!(names(h.list_all()), vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn detached_context_exposes_never_cancelled_token() {
        let ctx = ToolContext::detached();
        assert!(!ctx.cancellation().is_cancelled());
    }

    #[test]
    fn getter_returns_the_token_supplied_to_new() {
        let token = CancellationToken::new();
        let ctx = ToolContext::new(
            RunId::new(),
            StepId::new(),
            ToolActivation::detached(),
            token.clone(),
        );
        assert!(!ctx.cancellation().is_cancelled());
        token.cancel();
        assert!(ctx.cancellation().is_cancelled());
    }

    #[test]
    fn cloning_the_token_is_cheap_and_shares_state() {
        // `CancellationToken` is `Arc` internally — clones must observe
        // the same cancellation event without going through the context.
        let token = CancellationToken::new();
        let ctx = ToolContext::new(
            RunId::new(),
            StepId::new(),
            ToolActivation::detached(),
            token.clone(),
        );
        let cloned = ctx.cancellation().clone();
        assert!(!cloned.is_cancelled());
        token.cancel();
        assert!(cloned.is_cancelled());
    }
}
