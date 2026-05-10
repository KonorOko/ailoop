//! [`ToolRegistry`] container plus the [`Tool`] / [`ToolDyn`] trait
//! pair. The blanket `impl<T: Tool> ToolDyn for T` lets every typed
//! tool slot directly into the dynamic dispatch path used by the
//! engine.

use std::sync::{Arc, Mutex};

use ailoop_core::{ToolDefinition, ToolResultContent, ToolTag};
use indexmap::{IndexMap, IndexSet};
use serde::Serialize;

use crate::context::ToolContext;
use crate::errors::ToolRegistryError;

/// Owns every tool a [`Conversation`] can dispatch to and tracks which
/// of them are currently active.
///
/// Registration order matters: [`active_tools`](Self::active_tools)
/// iterates in insertion order (backed by [`IndexMap`]), and that
/// order is what the engine forwards in [`ChatRequest::tools`]. The
/// model sees tools in the same order the builder registered them.
///
/// `register` auto-activates the new tool. Capability filters
/// ([`activate_by_tags`](Self::activate_by_tags) /
/// [`deactivate_by_tags`](Self::deactivate_by_tags)) and the
/// [`deactivate_all`](Self::deactivate_all) reset operate on the
/// active set without touching registration.
///
/// [`Conversation`]: https://docs.rs/ailoop
/// [`ChatRequest::tools`]: ailoop_core::ChatRequest::tools
pub struct ToolRegistry {
    tools: IndexMap<String, Arc<dyn ToolDyn>>,
    active_tools: IndexSet<String>,
}

/// Typed tool entry point. Implement (or derive via
/// [`#[ailoop_tool]`](https://docs.rs/ailoop)) to expose a function
/// to the model with deserialized [`Args`](Self::Args) in and a
/// serializable [`Output`](Self::Output) out.
///
/// Every `Tool` is automatically a [`ToolDyn`] via the blanket impl,
/// so the same value can be registered with either
/// [`ConversationBuilder::tool`](https://docs.rs/ailoop) (typed,
/// preferred) or
/// [`ConversationBuilder::tool_dyn`](https://docs.rs/ailoop) (dynamic,
/// for plugin loaders / MCP).
pub trait Tool: Send + Sync + Sized {
    /// Wire-visible name of the tool. Must match
    /// `^[a-zA-Z0-9_-]{1,64}$` for Anthropic compatibility (Azure
    /// OpenAI is more permissive).
    const NAME: &'static str;

    /// Argument struct deserialized from the model's JSON `input`.
    type Args: for<'a> serde::Deserialize<'a> + Send;
    /// Successful return value. Serialized back as the tool result;
    /// strings round-trip as a single
    /// [`ToolResultBlock::Text`](ailoop_core::ToolResultBlock::Text)
    /// inside [`ToolResultContent`], anything else is JSON-encoded
    /// into the same shape.
    type Output: Serialize + Send;
    /// Failure type. Surfaced to the model as a [`ToolResultContent`]
    /// with `is_error: true` (rendered via `Display`); never reaches
    /// [`EngineError`](https://docs.rs/ailoop).
    type Error: std::error::Error + Send + Sync + 'static;

    /// JSON-Schema description of this tool that the model sees in
    /// every [`ChatRequest::tools`] entry. Typically derived by the
    /// [`#[ailoop_tool]`](https://docs.rs/ailoop) macro from the
    /// function signature plus optional doc-comments.
    ///
    /// [`ChatRequest::tools`]: ailoop_core::ChatRequest::tools
    fn definition(&self) -> ToolDefinition;
    /// Invoke the tool. The blanket [`ToolDyn::call`] impl
    /// deserializes the model's JSON into [`Args`](Self::Args),
    /// dispatches here, then serializes the result.
    ///
    /// `ctx` carries the [`RunId`](ailoop_core::RunId) /
    /// [`StepId`](ailoop_core::StepId) of the current dispatch and a
    /// [`ToolActivation`](crate::ToolActivation) handle into the per-run
    /// active set. Most handlers ignore it; meta-tools that load other
    /// tools on demand (`search_tools`, MCP discovery) call
    /// `ctx.tools().activate(name)` to expose tools on the next turn.
    fn call(
        &self,
        args: Self::Args,
        ctx: &ToolContext,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

/// Object-safe sibling of [`Tool`] used wherever the registry needs
/// `Arc<dyn ToolDyn>` — runtime tool discovery (MCP, plugin loaders,
/// config-driven catalogs). The blanket `impl<T: Tool> ToolDyn for T`
/// promotes every static tool to a dynamic one for free.
///
/// Implementing `ToolDyn` directly is the right move for tools whose
/// names or schemas come from a server. For everything else, prefer
/// [`Tool`] + the blanket impl: typed args, no manual JSON parsing.
#[async_trait::async_trait]
pub trait ToolDyn: Send + Sync {
    /// Wire-visible name. The blanket impl returns
    /// `T::NAME.to_string()`; manual impls (MCP) return the engine-
    /// facing composed name (e.g. `mcp__time__get_current_time`).
    fn name(&self) -> String;
    /// Tool definition the engine forwards to the provider.
    fn tool_definition(&self) -> ToolDefinition;
    /// Dispatch a tool call. Errors that originate inside the tool
    /// (bad args, exception during execution) come back as a
    /// [`ToolResultContent`] with `is_error: true` so the model sees
    /// them as a tool reply — never as `Err`. The `Result`-shaped
    /// surface for transport-level failures lives on
    /// [`ToolRegistry::tool_call`].
    ///
    /// `ctx` carries per-dispatch identifiers and a
    /// [`ToolActivation`](crate::ToolActivation) handle for tools
    /// that need to mutate the active set (deferred / dynamic
    /// tool loading). Handlers that don't need it simply ignore
    /// the parameter.
    async fn call(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolResultContent;
}

#[async_trait::async_trait]
impl<T: Tool> ToolDyn for T {
    fn name(&self) -> String {
        T::NAME.to_string()
    }

    fn tool_definition(&self) -> ToolDefinition {
        T::definition(&self)
    }

    async fn call(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolResultContent {
        let typed: T::Args = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return ToolResultContent::error(format!("Invalid args: {e}")),
        };

        match T::call(&self, typed, ctx).await {
            Ok(out) => match serde_json::to_value(&out) {
                Ok(serde_json::Value::String(s)) => ToolResultContent::text(s),
                Ok(v) => ToolResultContent::text(v.to_string()),
                Err(e) => ToolResultContent::error(format!("Failed to serialize tool output: {e}")),
            },
            Err(e) => ToolResultContent::error(format!("{e}")),
        }
    }
}

impl ToolRegistry {
    /// Build an empty registry with no tools and no active set.
    pub fn new() -> Self {
        Self {
            tools: IndexMap::new(),
            active_tools: IndexSet::new(),
        }
    }

    /// Look up `name` and dispatch the call with a freshly built
    /// detached [`ToolContext`]. Convenience entry point for
    /// standalone callers (tests, scripts) that don't carry an
    /// engine-issued context. Returns [`ToolRegistryError::NotFound`]
    /// when no tool with that name was registered (the engine handles
    /// this in-band by feeding an `Error` tool result back to the
    /// model rather than aborting the run). Tool-internal failures
    /// come back inside the [`ToolResultContent`] payload — they do
    /// not surface as `Err` here.
    pub async fn tool_call(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResultContent, ToolRegistryError> {
        self.tool_call_with_ctx(name, args, &ToolContext::detached())
            .await
    }

    /// Look up `name` and dispatch the call, threading the supplied
    /// [`ToolContext`] through to the handler. The engine uses this
    /// path with a per-dispatch context so handlers can mutate the
    /// per-run active set; standalone callers prefer
    /// [`Self::tool_call`].
    pub async fn tool_call_with_ctx(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolResultContent, ToolRegistryError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolRegistryError::NotFound(name.into()))?;

        Ok(tool.call(args, ctx).await)
    }

    /// Return a shared snapshot of the catalog (every registered
    /// tool, by name, in registration order). The engine takes one
    /// at the start of each run and hands it to every per-dispatch
    /// [`ToolActivation`] so that `list_*` reads are cheap and don't
    /// reach back into the registry.
    pub fn catalog_arc(&self) -> Arc<IndexMap<String, Arc<dyn ToolDyn>>> {
        Arc::new(self.tools.clone())
    }

    /// Build a fresh per-run active-set handle initialised from the
    /// current active set. Returns the `Arc<Mutex<...>>` the engine
    /// hands to every [`ToolActivation`] for this run; mutations
    /// inside a tool handler are visible to the engine on the next
    /// turn without touching the underlying registry.
    pub fn snapshot_active(&self) -> Arc<Mutex<IndexSet<String>>> {
        Arc::new(Mutex::new(self.active_tools.clone()))
    }

    /// Iterate over the currently active tools in registration
    /// (insertion) order. The engine uses this to assemble
    /// [`ChatRequest::tools`] before each turn, so the model sees
    /// tools in the order the builder registered them.
    ///
    /// [`ChatRequest::tools`]: ailoop_core::ChatRequest::tools
    pub fn active_tools(&self) -> impl Iterator<Item = &Arc<dyn ToolDyn>> {
        self.tools
            .iter()
            .filter(|(name, _)| self.active_tools.contains(*name))
            .map(|(_, tool)| tool)
    }

    /// Iterate over registered-but-inactive tools, in registration
    /// order. The complement of [`active_tools`](Self::active_tools).
    pub fn inactive_tools(&self) -> impl Iterator<Item = &Arc<dyn ToolDyn>> {
        self.tools
            .iter()
            .filter(|(name, _)| !self.active_tools.contains(*name))
            .map(|(_, tool)| tool)
    }

    /// Register `tool` and mark it active. Returns
    /// [`ToolRegistryError::AlreadyRegistered`] when a tool with the
    /// same wire name is already present — names are unique across
    /// the registry. Auto-activation matches the convention that a
    /// freshly registered tool should be reachable without a separate
    /// activation step; capability filters can subtract from the
    /// active set after the fact.
    pub fn register(&mut self, tool: Arc<dyn ToolDyn>) -> Result<(), ToolRegistryError> {
        let name = tool.tool_definition().name;
        if self.tools.contains_key(&name) {
            return Err(ToolRegistryError::AlreadyRegistered(name));
        }

        self.active_tools.insert(name.clone());
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Add `tool_name` to the active set. Returns
    /// [`ToolRegistryError::NotFound`] when the tool was never
    /// registered. No-op when the tool is already active.
    pub fn activate_tool(&mut self, tool_name: &str) -> Result<(), ToolRegistryError> {
        if !self.tools.contains_key(tool_name) {
            return Err(ToolRegistryError::NotFound(tool_name.to_string()));
        }
        self.active_tools.insert(tool_name.into());
        Ok(())
    }

    /// Remove `tool_name` from the active set. Silent no-op for an
    /// unknown name (asymmetric with
    /// [`activate_tool`](Self::activate_tool), which errors): the
    /// "tool is no longer active" state is the same whether the tool
    /// exists or not, so the call is idempotent.
    pub fn deactivate_tool(&mut self, tool_name: &str) -> Result<(), ToolRegistryError> {
        self.active_tools.shift_remove(tool_name);
        Ok(())
    }

    /// Activate every registered tool whose declared tags overlap with `tags`.
    ///
    /// Additive: tools already active stay active. Tools with no declared
    /// tags are never matched.
    pub fn activate_by_tags(&mut self, tags: &[ToolTag]) {
        let to_activate: Vec<String> = self
            .tools
            .iter()
            .filter_map(|(name, tool)| {
                let def = tool.tool_definition();
                def.tags
                    .iter()
                    .any(|t| tags.contains(t))
                    .then(|| name.clone())
            })
            .collect();

        for name in to_activate {
            self.active_tools.insert(name);
        }
    }

    /// Deactivate every registered tool whose declared tags overlap with `tags`.
    ///
    /// Subtractive: tools with no declared tags are never matched and stay
    /// in whichever state they were.
    pub fn deactivate_by_tags(&mut self, tags: &[ToolTag]) {
        let to_deactivate: Vec<String> = self
            .tools
            .iter()
            .filter_map(|(name, tool)| {
                let def = tool.tool_definition();
                def.tags
                    .iter()
                    .any(|t| tags.contains(t))
                    .then(|| name.clone())
            })
            .collect();

        for name in to_deactivate {
            self.active_tools.shift_remove(&name);
        }
    }

    /// Clear the active set. Every registered tool becomes inactive.
    pub fn deactivate_all(&mut self) {
        self.active_tools.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::request::{ToolDefinition, ToolTag};
    use serde::Deserialize;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn temp_with_content(content: &str) -> NamedTempFile {
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{content}").unwrap();
        tmp
    }

    #[derive(Deserialize)]
    struct ReadFileArgs {
        path: String,
    }

    struct ReadFile;

    impl Tool for ReadFile {
        const NAME: &'static str = "read_file";

        type Args = ReadFileArgs;
        type Error = std::io::Error;
        type Output = String;

        fn call(
            &self,
            args: Self::Args,
            _ctx: &ToolContext,
        ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
            async move { std::fs::read_to_string(&args.path) }
        }

        fn definition(&self) -> ToolDefinition {
            let parameters = serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "path to read file" },
                },
                "required": []
            });

            ToolDefinition::new(Self::NAME, "read file from path", parameters, vec![])
        }
    }

    #[test]
    fn register_tools() {
        let mut registry = ToolRegistry::new();

        registry
            .register(Arc::new(ReadFile))
            .expect("Failed in registry tool");

        assert!(true);
    }

    #[tokio::test]
    async fn call_tool() {
        let mut registry = ToolRegistry::new();

        registry
            .register(Arc::new(ReadFile))
            .expect("Failed in registry tool");

        let path = temp_with_content("content");

        let result = registry
            .tool_call(
                "read_file",
                serde_json::json!({
                    "path": path.path()
                }),
            )
            .await
            .expect("Failed in read file");

        assert_eq!(result.as_text(), Some("content"));
        assert!(!result.is_error);
    }

    struct TaggedTool {
        name: &'static str,
        tags: Vec<ToolTag>,
    }

    #[async_trait::async_trait]
    impl ToolDyn for TaggedTool {
        fn name(&self) -> String {
            self.name.into()
        }

        fn tool_definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                self.name,
                "tagged",
                serde_json::json!({"type":"object","properties":{},"required":[]}),
                self.tags.clone(),
            )
        }

        async fn call(&self, _args: serde_json::Value, _ctx: &ToolContext) -> ToolResultContent {
            ToolResultContent::text("")
        }
    }

    fn names<'a>(iter: impl Iterator<Item = &'a Arc<dyn ToolDyn>>) -> Vec<String> {
        iter.map(|t| t.tool_definition().name).collect()
    }

    #[test]
    fn register_auto_activates() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TaggedTool {
                name: "fetch",
                tags: vec![ToolTag::ReadOnly, ToolTag::Network],
            }))
            .unwrap();

        assert_eq!(names(registry.active_tools()), vec!["fetch"]);
    }

    #[test]
    fn deactivate_tool_removes_from_active() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TaggedTool {
                name: "fetch",
                tags: vec![ToolTag::ReadOnly],
            }))
            .unwrap();

        registry.deactivate_tool("fetch").unwrap();
        assert!(names(registry.active_tools()).is_empty());
        assert_eq!(names(registry.inactive_tools()), vec!["fetch"]);
    }

    #[test]
    fn activate_by_tags_only_matches_overlapping() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TaggedTool {
                name: "fetch",
                tags: vec![ToolTag::ReadOnly, ToolTag::Network],
            }))
            .unwrap();
        registry
            .register(Arc::new(TaggedTool {
                name: "rm",
                tags: vec![ToolTag::Destructive, ToolTag::WritesFiles],
            }))
            .unwrap();
        registry
            .register(Arc::new(TaggedTool {
                name: "noop",
                tags: vec![],
            }))
            .unwrap();

        registry.deactivate_all();
        registry.activate_by_tags(&[ToolTag::ReadOnly]);

        assert_eq!(names(registry.active_tools()), vec!["fetch"]);
    }

    #[test]
    fn deactivate_by_tags_removes_matching() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TaggedTool {
                name: "fetch",
                tags: vec![ToolTag::ReadOnly],
            }))
            .unwrap();
        registry
            .register(Arc::new(TaggedTool {
                name: "rm",
                tags: vec![ToolTag::Destructive],
            }))
            .unwrap();

        registry.deactivate_by_tags(&[ToolTag::Destructive]);

        assert_eq!(names(registry.active_tools()), vec!["fetch"]);
    }

    #[test]
    fn untagged_tool_never_matches_capability_filter() {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(TaggedTool {
                name: "noop",
                tags: vec![],
            }))
            .unwrap();

        registry.deactivate_all();
        registry.activate_by_tags(&[ToolTag::ReadOnly, ToolTag::Network]);

        assert!(names(registry.active_tools()).is_empty());
    }
}
