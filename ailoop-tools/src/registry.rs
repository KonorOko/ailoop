use std::sync::Arc;

use ailoop_core::{ToolDefinition, ToolResultContent};
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
        self.tools.values()
    }

    pub fn inactive_tools(&self) -> impl Iterator<Item = &Arc<dyn ToolDyn>> {
        self.tools
            .values()
            .filter(|tool| !self.active_tools.contains(&tool.tool_definition().name))
    }

    pub fn register(&mut self, tool: Arc<dyn ToolDyn>) -> Result<(), ToolRegistryError> {
        let name = tool.tool_definition().name;
        if self.tools.contains_key(&name) {
            return Err(ToolRegistryError::AlreadyRegistered(name));
        }

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::request::ToolDefinition;
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
}
