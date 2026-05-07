use ailoop::{Conversation, ToolTag, ailoop_tool};
use ailoop_core::{ChatRequest, CompletionModel, StreamChunk};
use futures::stream::BoxStream;

struct MockModel;

#[async_trait::async_trait]
impl CompletionModel for MockModel {
    type Error = std::convert::Infallible;

    fn name(&self) -> &str {
        "mock"
    }
    fn model(&self) -> &str {
        "mock"
    }

    async fn chat_stream(
        &self,
        _req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[ailoop_tool(description = "list a directory", tags(ReadOnly))]
async fn list_dir(_path: String) -> i32 {
    0
}

#[ailoop_tool(description = "delete a file", tags(Destructive, WritesFiles))]
async fn delete_file(_path: String) -> i32 {
    0
}

#[ailoop_tool(description = "make an http request", tags(Network))]
async fn http_get(_url: String) -> i32 {
    0
}

#[ailoop_tool(description = "untagged helper")]
async fn untagged_helper() -> i32 {
    0
}

#[test]
fn no_capabilities_filter_keeps_all_tools_active() {
    let chat = Conversation::builder(MockModel)
        .tool(ListDir)
        .tool(DeleteFile)
        .tool(UntaggedHelper)
        .build()
        .unwrap();

    let mut active = chat.active_tool_names();
    active.sort();
    assert_eq!(
        active,
        vec![
            "delete_file".to_string(),
            "list_dir".to_string(),
            "untagged_helper".to_string()
        ]
    );
}

#[test]
fn read_only_capability_excludes_destructive_and_untagged() {
    let chat = Conversation::builder(MockModel)
        .tool(ListDir)
        .tool(DeleteFile)
        .tool(UntaggedHelper)
        .with_capabilities(&[ToolTag::ReadOnly])
        .build()
        .unwrap();

    assert_eq!(chat.active_tool_names(), vec!["list_dir".to_string()]);
}

#[test]
fn capabilities_apply_regardless_of_call_order() {
    let chat = Conversation::builder(MockModel)
        .with_capabilities(&[ToolTag::ReadOnly])
        .tool(ListDir)
        .tool(DeleteFile)
        .build()
        .unwrap();

    assert_eq!(chat.active_tool_names(), vec!["list_dir".to_string()]);
}

#[test]
fn capabilities_can_combine_multiple_tags() {
    let chat = Conversation::builder(MockModel)
        .tool(ListDir)
        .tool(HttpGet)
        .tool(DeleteFile)
        .with_capabilities(&[ToolTag::ReadOnly, ToolTag::Network])
        .build()
        .unwrap();

    let mut active = chat.active_tool_names();
    active.sort();
    assert_eq!(active, vec!["http_get".to_string(), "list_dir".to_string()]);
}

#[test]
fn empty_capabilities_yields_no_active_tools() {
    let chat = Conversation::builder(MockModel)
        .tool(ListDir)
        .tool(DeleteFile)
        .with_capabilities(&[])
        .build()
        .unwrap();

    assert!(chat.active_tool_names().is_empty());
}

#[test]
fn last_with_capabilities_call_wins() {
    let chat = Conversation::builder(MockModel)
        .tool(ListDir)
        .tool(DeleteFile)
        .with_capabilities(&[ToolTag::Destructive])
        .with_capabilities(&[ToolTag::ReadOnly])
        .build()
        .unwrap();

    assert_eq!(chat.active_tool_names(), vec!["list_dir".to_string()]);
}
