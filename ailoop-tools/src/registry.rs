use std::sync::Arc;

use ailoop_core::{ToolDefinition, ToolResultContent, ToolTag};
use indexmap::{IndexMap, IndexSet};
use serde::Serialize;

use crate::errors::ToolRegistryError;

pub struct ToolRegistry {
    tools: IndexMap<String, Arc<dyn ToolDyn>>,
    active_tools: IndexSet<String>,
}

pub trait Tool: Send + Sync + Sized {
    const NAME: &'static str;

    type Args: for<'a> serde::Deserialize<'a> + Send;
    type Output: Serialize + Send;
    type Error: std::error::Error + Send + Sync + 'static;

    fn definition(&self) -> ToolDefinition;
    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send;
}

#[async_trait::async_trait]
pub trait ToolDyn: Send + Sync {
    fn name(&self) -> String;
    fn tool_definition(&self) -> ToolDefinition;
    async fn call(&self, args: serde_json::Value) -> ToolResultContent;
}

#[async_trait::async_trait]
impl<T: Tool> ToolDyn for T {
    fn name(&self) -> String {
        T::NAME.to_string()
    }

    fn tool_definition(&self) -> ToolDefinition {
        T::definition(&self)
    }

    async fn call(&self, args: serde_json::Value) -> ToolResultContent {
        let typed: T::Args = match serde_json::from_value(args) {
            Ok(a) => a,
            Err(e) => return ToolResultContent::Error(format!("Invalid args: {e}")),
        };

        match T::call(&self, typed).await {
            Ok(out) => match serde_json::to_value(&out) {
                Ok(serde_json::Value::String(s)) => ToolResultContent::Text(s),
                Ok(v) => ToolResultContent::Text(v.to_string()),
                Err(e) => ToolResultContent::Error(format!("Failed to serialize tool output: {e}")),
            },
            Err(e) => ToolResultContent::Error(format!("{e}")),
        }
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: IndexMap::new(),
            active_tools: IndexSet::new(),
        }
    }

    pub async fn tool_call(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResultContent, ToolRegistryError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolRegistryError::NotFound(name.into()))?;

        Ok(tool.call(args).await)
    }

    pub fn active_tools(&self) -> impl Iterator<Item = &Arc<dyn ToolDyn>> {
        self.tools
            .iter()
            .filter(|(name, _)| self.active_tools.contains(*name))
            .map(|(_, tool)| tool)
    }

    pub fn inactive_tools(&self) -> impl Iterator<Item = &Arc<dyn ToolDyn>> {
        self.tools
            .iter()
            .filter(|(name, _)| !self.active_tools.contains(*name))
            .map(|(_, tool)| tool)
    }

    pub fn register(&mut self, tool: Arc<dyn ToolDyn>) -> Result<(), ToolRegistryError> {
        let name = tool.tool_definition().name;
        if self.tools.contains_key(&name) {
            return Err(ToolRegistryError::AlreadyRegistered(name));
        }

        self.active_tools.insert(name.clone());
        self.tools.insert(name, tool);
        Ok(())
    }

    pub fn activate_tool(&mut self, tool_name: &str) -> Result<(), ToolRegistryError> {
        if !self.tools.contains_key(tool_name) {
            return Err(ToolRegistryError::NotFound(tool_name.to_string()));
        }
        self.active_tools.insert(tool_name.into());
        Ok(())
    }

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

            ToolDefinition {
                name: Self::NAME.into(),
                description: "read file from path".into(),
                input_schema: parameters,
                tags: vec![],
                cache_control: None,
            }
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

        match result {
            ToolResultContent::Text(text) => assert_eq!("content", text),
            ToolResultContent::Error(error) => panic!("Failed to read"),
        }
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
            ToolDefinition {
                name: self.name.into(),
                description: "tagged".into(),
                input_schema: serde_json::json!({"type":"object","properties":{},"required":[]}),
                tags: self.tags.clone(),
                cache_control: None,
            }
        }

        async fn call(&self, _args: serde_json::Value) -> ToolResultContent {
            ToolResultContent::Text(String::new())
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
