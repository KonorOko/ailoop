//! Per-dispatch context handed to every [`Tool`](crate::Tool) /
//! [`ToolDyn`] call.
//!
//! [`ToolContext`] carries the [`RunId`], [`StepId`] and call id of the
//! current dispatch plus a [`ToolActivation`] handle into the per-run active
//! tool set. Handlers that need to mutate which tools are visible on
//! the next turn (deferred-tools / dynamic tool loading patterns) call
//! the activation methods directly — no shared `Arc<Mutex<...>>`
//! plumbing on the user side, no middleware to filter `req.tools`.
//!
//! The handle is per-run, so two concurrent runs do not share state.
//!
//! Tools that spend tokens outside the engine's own provider turns (a
//! sub-agent, a tool that calls an LLM directly) report that spend
//! through [`ToolContext::report_usage`]; the engine folds it into the
//! run total on `RunFinished.usage`. See [`UsageSink`].

use std::sync::{Arc, Mutex, MutexGuard};

use ailoop_core::{RunId, StepId, ToolDefinition, Usage};
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
    call_id: String,
    activation: ToolActivation,
    cancellation: CancellationToken,
    usage: UsageSink,
}

impl ToolContext {
    /// Construct a context bound to a real run. Used by the engine on
    /// each tool dispatch — tests and standalone callers want
    /// [`Self::detached`] instead.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
    pub fn new(
        run_id: RunId,
        step_id: StepId,
        call_id: impl Into<String>,
        activation: ToolActivation,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            run_id,
            step_id,
            call_id: call_id.into(),
            activation,
            cancellation,
            usage: UsageSink::new(),
        }
    }

    /// Replace the [`UsageSink`] that [`Self::report_usage`] writes to.
    /// The engine calls this on every dispatch so reports land in the
    /// run's total; standalone callers can pass their own sink to read
    /// back what a tool reported.
    pub fn with_usage_sink(mut self, sink: UsageSink) -> Self {
        self.usage = sink;
        self
    }

    /// Build a detached context with synthetic identifiers, a no-op
    /// activation handle, and a fresh never-cancelled token. Use for
    /// standalone
    /// [`ToolRegistry::tool_call`](crate::ToolRegistry::tool_call)
    /// invocations and unit tests where no engine is in the loop.
    pub fn detached() -> Self {
        let run_id = RunId::new();
        Self {
            call_id: format!("detached-{run_id}"),
            run_id,
            step_id: StepId::new(),
            activation: ToolActivation::detached(),
            cancellation: CancellationToken::new(),
            usage: UsageSink::new(),
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

    /// Provider-assigned id of the tool call being dispatched: the same
    /// value as `call_id` on the [`ToolCallInfo`](ailoop_core::ToolCallInfo)
    /// the middleware tool hooks receive and on the `ToolResult` chunk.
    /// Synthetic for [`Self::detached`].
    pub fn call_id(&self) -> &str {
        &self.call_id
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

    /// Report tokens this tool spent outside the engine's own provider
    /// turns — typically a tool that calls an LLM itself. The engine
    /// adds the report to the run's [`Usage`] total, so it shows up in
    /// `RunFinished.usage` / `RunOutcome.usage`.
    ///
    /// Reports count as soon as they are made: if the run aborts
    /// (timeout, cancellation) and the tool future is dropped
    /// afterwards, what was already reported stays in the aborted
    /// run's total. Report each spend once; the engine does not
    /// deduplicate. `ailoop::SubAgentTool` reports its child run's usage
    /// automatically.
    ///
    /// On a [`Self::detached`] context the report only accumulates in
    /// the context's own sink (readable via [`Self::usage_sink`]).
    pub fn report_usage(&self, usage: Usage) {
        self.usage.report(usage);
    }

    /// Sink behind [`Self::report_usage`]. Hand a clone to a nested run
    /// or a spawned task that needs to report on the tool's behalf.
    pub fn usage_sink(&self) -> &UsageSink {
        &self.usage
    }
}

/// Shared accumulator for [`Usage`] reported by tools.
///
/// Cheap to clone (`Arc` internally); every clone adds into the same
/// total. A sink built with [`Self::forwarding_to`] also adds every
/// report into its parent, so usage spent by nested runs rolls up to
/// the outermost run as it happens rather than at the end.
#[derive(Clone, Default)]
pub struct UsageSink {
    inner: Arc<UsageSinkInner>,
}

#[derive(Default)]
struct UsageSinkInner {
    total: Mutex<Usage>,
    parent: Option<UsageSink>,
}

impl UsageSinkInner {
    fn lock(&self) -> MutexGuard<'_, Usage> {
        // Poisoning only means another thread panicked mid-update; the
        // total itself is still consistent.
        self.total.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl UsageSink {
    /// Standalone sink with a zero total.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sink with a zero total that also forwards every report into
    /// `parent` (and, transitively, into the parent's parents).
    pub fn forwarding_to(parent: UsageSink) -> Self {
        Self {
            inner: Arc::new(UsageSinkInner {
                total: Mutex::new(Usage::default()),
                parent: Some(parent),
            }),
        }
    }

    /// Add `usage` to this sink's total and to every parent's.
    pub fn report(&self, usage: Usage) {
        *self.inner.lock() += usage;
        if let Some(parent) = &self.inner.parent {
            parent.report(usage);
        }
    }

    /// Sum of every report made to this sink (and its clones) so far.
    pub fn total(&self) -> Usage {
        *self.inner.lock()
    }
}

impl std::fmt::Debug for UsageSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsageSink")
            .field("total", &self.total())
            .field("forwarding", &self.inner.parent.is_some())
            .finish()
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

impl ToolActivationInner {
    fn lock(&self) -> MutexGuard<'_, IndexSet<String>> {
        // Poisoning only means another thread panicked mid-update; the
        // set itself is still consistent.
        self.active.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl ToolActivation {
    /// Construct a handle backed by a shared catalog and a per-run
    /// active set. Used by the engine; downstream callers rarely
    /// build this themselves.
    ///
    /// Engine plumbing, hidden from the docs and not covered by semver:
    /// its signature may change in any release.
    #[doc(hidden)]
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
        inner.lock().contains(name)
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
        inner.lock().insert(name.to_string());
        Ok(())
    }

    /// Remove `name` from the active set. Idempotent — silently
    /// no-ops for unknown names (consistent with
    /// [`ToolRegistry::deactivate`](crate::ToolRegistry::deactivate)).
    /// Returns [`ToolActivationError::Detached`] for detached handles.
    pub fn deactivate(&self, name: &str) -> Result<(), ToolActivationError> {
        let inner = self.inner.as_ref().ok_or(ToolActivationError::Detached)?;
        inner.lock().shift_remove(name);
        Ok(())
    }

    /// Snapshot of currently active tool definitions, in registration
    /// order. The model would see these on the next turn.
    pub fn list_active(&self) -> Vec<ToolDefinition> {
        let Some(inner) = &self.inner else {
            return Vec::new();
        };
        let active = inner.lock();
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
        let active = inner.lock();
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
        fn name(&self) -> &str {
            self.name
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
    fn call_id_returns_the_id_supplied_to_new() {
        let ctx = ToolContext::new(
            RunId::new(),
            StepId::new(),
            "toolu_42",
            ToolActivation::detached(),
            CancellationToken::new(),
        );
        assert_eq!(ctx.call_id(), "toolu_42");
    }

    #[test]
    fn detached_context_has_a_synthetic_call_id() {
        let a = ToolContext::detached();
        let b = ToolContext::detached();
        assert!(!a.call_id().is_empty());
        assert_ne!(a.call_id(), b.call_id());
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
            "toolu_x",
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
            "toolu_x",
            ToolActivation::detached(),
            token.clone(),
        );
        let cloned = ctx.cancellation().clone();
        assert!(!cloned.is_cancelled());
        token.cancel();
        assert!(cloned.is_cancelled());
    }

    fn tokens(input: u64, output: u64) -> Usage {
        let mut u = Usage::default();
        u.input_tokens = input;
        u.output_tokens = output;
        u
    }

    #[test]
    fn report_usage_accumulates_in_the_context_sink() {
        let ctx = ToolContext::detached();
        ctx.report_usage(tokens(10, 2));
        ctx.report_usage(tokens(5, 1));
        let total = ctx.usage_sink().total();
        assert_eq!((total.input_tokens, total.output_tokens), (15, 3));
    }

    #[test]
    fn with_usage_sink_routes_reports_to_the_supplied_sink() {
        let sink = UsageSink::new();
        let ctx = ToolContext::detached().with_usage_sink(sink.clone());
        ctx.report_usage(tokens(7, 3));
        assert_eq!(sink.total().input_tokens, 7);
    }

    #[test]
    fn forwarding_sink_propagates_to_every_ancestor() {
        let root = UsageSink::new();
        let mid = UsageSink::forwarding_to(root.clone());
        let leaf = UsageSink::forwarding_to(mid.clone());
        leaf.report(tokens(4, 1));
        mid.report(tokens(1, 1));
        assert_eq!(leaf.total().input_tokens, 4);
        assert_eq!(mid.total().input_tokens, 5);
        assert_eq!(root.total().input_tokens, 5);
        assert_eq!(root.total().output_tokens, 2);
    }

    /// Run `f` on another thread and wait for it to panic. `f` panics
    /// while holding a lock, which leaves that lock poisoned.
    fn panic_on_thread(f: impl FnOnce() + Send + 'static) {
        assert!(std::thread::spawn(f).join().is_err());
    }

    #[test]
    fn usage_sink_keeps_working_after_its_lock_is_poisoned() {
        let sink = UsageSink::new();
        sink.report(tokens(2, 1));
        let inner = sink.inner.clone();
        panic_on_thread(move || {
            let _guard = inner.total.lock().unwrap();
            panic!("poison the lock");
        });
        assert!(sink.inner.total.is_poisoned());

        sink.report(tokens(3, 1));
        assert_eq!(sink.total(), tokens(5, 2));
    }

    #[test]
    fn tool_activation_keeps_working_after_its_lock_is_poisoned() {
        let h = build_handle(&["a"], &["a", "b"]);
        let active = h.inner.as_ref().unwrap().active.clone();
        let held = active.clone();
        panic_on_thread(move || {
            let _guard = held.lock().unwrap();
            panic!("poison the lock");
        });
        assert!(active.is_poisoned());

        assert!(h.is_active("a"));
        h.activate("b").unwrap();
        h.deactivate("a").unwrap();
        assert_eq!(names(h.list_active()), vec!["b"]);
        assert_eq!(names(h.list_inactive()), vec!["a"]);
    }
}
