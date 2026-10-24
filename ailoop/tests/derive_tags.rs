use ailoop::{Tool, ToolTag, ailoop_tool};

#[ailoop_tool(description = "fetch a url", tags(ReadOnly, Network))]
async fn fetch(url: String) -> String {
    url
}

#[ailoop_tool(description = "delete a file", tags(Destructive, WritesFiles))]
async fn delete_file(path: String) -> bool {
    let _ = path;
    true
}

#[ailoop_tool(description = "no tags")]
async fn untagged() -> i32 {
    1
}

#[test]
fn derive_emits_tag_list() {
    let def = Fetch.definition();
    assert_eq!(def.tags.len(), 2);
    assert!(matches!(def.tags[0], ToolTag::ReadOnly));
    assert!(matches!(def.tags[1], ToolTag::Network));
}

#[test]
fn derive_emits_destructive_writesfiles() {
    let def = DeleteFile.definition();
    assert_eq!(def.tags.len(), 2);
    assert!(matches!(def.tags[0], ToolTag::Destructive));
    assert!(matches!(def.tags[1], ToolTag::WritesFiles));
}

#[test]
fn derive_emits_empty_when_tags_omitted() {
    let def = Untagged.definition();
    assert!(def.tags.is_empty());
}

#[ailoop_tool(description = "read a file", tags(ReadOnly), concurrency_safe)]
async fn read_file(path: String) -> String {
    path
}

#[test]
fn derive_sets_concurrency_safe_only_when_asked() {
    use ailoop::ToolDyn;
    assert!(ReadFile.concurrency_safe());
    const { assert!(<ReadFile as Tool>::CONCURRENCY_SAFE) };
    assert!(!Fetch.concurrency_safe());
    assert!(matches!(
        ReadFile.definition().tags[..],
        [ToolTag::ReadOnly]
    ));
}

#[ailoop_tool(description = "run a long job", backgroundable)]
async fn long_job(target: String) -> String {
    target
}

#[test]
fn derive_sets_backgroundable_only_when_asked() {
    use ailoop::ToolDyn;
    assert!(LongJob.backgroundable());
    const { assert!(<LongJob as Tool>::BACKGROUNDABLE) };
    assert!(!ReadFile.backgroundable());
    assert!(!LongJob.concurrency_safe());
}
