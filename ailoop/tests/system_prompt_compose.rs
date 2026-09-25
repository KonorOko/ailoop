//! Integration tests for how the internal `SystemPromptMiddleware`
//! composes the builder's system prompt with one written by a user
//! middleware. The internal middleware runs *after* user middlewares
//! (it needs the final `req.tools` for per-group sections), so it must
//! merge rather than overwrite: builder prompt first, user overlay
//! after, `cache_control` on user blocks preserved.
//!
//! Each test drives a real turn through `Conversation::run` with a
//! [`CapturingModel`] that records the `system_prompt` the model
//! actually receives.

use std::io::Write;
use std::sync::{Arc, Mutex};

use ailoop::{
    CacheControl, ChatMiddleware, ChatRequest, CompletionModel, Conversation, FinishReason, RunId,
    StepId, StreamChunk, SystemBlock, SystemPrompt, Usage, ailoop_tool,
};
use futures::stream::BoxStream;
use tempfile::NamedTempFile;

/// `CompletionModel` that records the `system_prompt` of every incoming
/// request and replies with a one-chunk `EndTurn`.
#[derive(Clone, Default)]
struct CapturingModel {
    system_prompt: Arc<Mutex<Option<Option<SystemPrompt>>>>,
}

impl CapturingModel {
    fn last_system_prompt(&self) -> Option<SystemPrompt> {
        self.system_prompt
            .lock()
            .unwrap()
            .clone()
            .expect("CapturingModel: chat_stream was never called")
    }
}

#[async_trait::async_trait]
impl CompletionModel for CapturingModel {
    type Error = std::convert::Infallible;

    fn name(&self) -> &str {
        "capture"
    }

    fn model(&self) -> &str {
        "capture"
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, Self::Error>>, Self::Error> {
        *self.system_prompt.lock().unwrap() = Some(req.system_prompt);
        let chunks: Vec<Result<StreamChunk, Self::Error>> = vec![
            Ok(StreamChunk::TextDelta { delta: "ok".into() }),
            Ok(StreamChunk::TurnFinished {
                reason: FinishReason::EndTurn,
                usage: Usage::default(),
                service_tier: None,
            }),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

/// User middleware that writes a fixed `system_prompt` on every request.
struct SetSystemPrompt(SystemPrompt);

#[async_trait::async_trait]
impl ChatMiddleware for SetSystemPrompt {
    async fn on_chat_request(&self, _run_id: &RunId, _step_id: &StepId, req: &mut ChatRequest) {
        req.system_prompt = Some(self.0.clone());
    }
}

#[ailoop_tool(description = "search the web")]
async fn web_search(_q: String) -> i32 {
    0
}

const BUILDER: &str = "BUILDER_SENTINEL: base instructions";
const USER: &str = "USER_SENTINEL: per-request overlay";

fn write_tempfile(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().expect("create tempfile");
    write!(f, "{content}").expect("write tempfile");
    f
}

/// `Plain + Plain`: both prompts reach the model, builder first, joined
/// by a blank line.
#[tokio::test]
async fn plain_user_prompt_is_appended_after_builder_prompt() {
    let model = CapturingModel::default();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        .system_prompt(BUILDER)
        .middleware(Arc::new(SetSystemPrompt(USER.into())))
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    match probe.last_system_prompt() {
        Some(SystemPrompt::Plain(text)) => {
            assert_eq!(text, format!("{BUILDER}\n\n{USER}"));
        }
        other => panic!("expected SystemPrompt::Plain, got {other:?}"),
    }
}

/// Tool-group sections are still rendered from the final `req.tools`
/// and sit inside the builder prefix, ahead of the user overlay.
#[tokio::test]
async fn tool_group_sections_stay_in_builder_prefix() {
    const GUIDE: &str = "GUIDE_SENTINEL: rules for web search";
    let guide = write_tempfile(GUIDE);
    let model = CapturingModel::default();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        .system_prompt(BUILDER)
        .tool(WebSearch)
        .tools_with_prompt_file(["web_search"], guide.path())
        .middleware(Arc::new(SetSystemPrompt(USER.into())))
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    let text = probe.last_system_prompt().expect("system prompt").as_text();
    let builder_at = text.find(BUILDER).expect("builder prompt present");
    let guide_at = text.find(GUIDE).expect("tool guide present");
    let user_at = text.find(USER).expect("user overlay present");
    assert!(
        builder_at < guide_at && guide_at < user_at,
        "expected builder < guide < user, got {text:?}"
    );
}

/// User `Blocks` force a `Blocks` result: the builder prompt becomes a
/// leading block without a cache breakpoint, and the user's blocks —
/// including their `cache_control` — come through unchanged.
#[tokio::test]
async fn blocks_user_prompt_preserves_cache_control() {
    let model = CapturingModel::default();
    let probe = model.clone();

    let user = SystemPrompt::Blocks(vec![
        SystemBlock::new("USER_CACHED").with_cache_control(CacheControl::Ephemeral),
        SystemBlock::new("USER_VOLATILE"),
    ]);

    let mut chat = Conversation::builder(model)
        .system_prompt(BUILDER)
        .middleware(Arc::new(SetSystemPrompt(user)))
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    match probe.last_system_prompt() {
        Some(SystemPrompt::Blocks(blocks)) => {
            let shape: Vec<(&str, Option<&CacheControl>)> = blocks
                .iter()
                .map(|b| (b.text.as_str(), b.cache_control.as_ref()))
                .collect();
            assert_eq!(
                shape,
                vec![
                    (BUILDER, None),
                    ("USER_CACHED", Some(&CacheControl::Ephemeral)),
                    ("USER_VOLATILE", None),
                ]
            );
        }
        other => panic!("expected SystemPrompt::Blocks, got {other:?}"),
    }
}

/// With no builder prompt, the user's prompt must pass through
/// untouched — no empty leading block, no separator.
#[tokio::test]
async fn empty_builder_prompt_leaves_user_prompt_intact() {
    let model = CapturingModel::default();
    let probe = model.clone();

    let user = SystemPrompt::Blocks(vec![
        SystemBlock::new("USER_CACHED").with_cache_control(CacheControl::Ephemeral),
    ]);

    let mut chat = Conversation::builder(model)
        .middleware(Arc::new(SetSystemPrompt(user)))
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    match probe.last_system_prompt() {
        Some(SystemPrompt::Blocks(blocks)) => {
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].text, "USER_CACHED");
            assert_eq!(blocks[0].cache_control, Some(CacheControl::Ephemeral));
        }
        other => panic!("expected SystemPrompt::Blocks, got {other:?}"),
    }
}
