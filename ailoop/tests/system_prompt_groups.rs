//! Integration tests for grouped per-tool system-prompt sections —
//! [`ConversationBuilder::tools_with_prompt_file`] and the underlying
//! `SystemPromptMiddleware` group-based rendering. These tests verify
//! that a guide shared by several tools is rendered at most once per
//! turn, regardless of how many of the grouped tools are active, and
//! that render order follows group registration order (not the order of
//! tools in `req.tools`). They also pin the no-regression contract for
//! the existing 1:1 [`ConversationBuilder::tool_with_prompt_file`].
//!
//! The strategy is to drive a real turn through `Conversation::run` with
//! a [`CapturingModel`] that records the final [`ChatRequest`]
//! `system_prompt` after every middleware (including the internal
//! `SystemPromptMiddleware`, which is appended *after* user middlewares)
//! has mutated the request.

use std::io::Write;
use std::sync::{Arc, Mutex};

use ailoop::{Conversation, ailoop_tool};
use ailoop_core::{ChatRequest, CompletionModel, FinishReason, StreamChunk, Usage};
use futures::stream::BoxStream;
use tempfile::NamedTempFile;

/// `CompletionModel` that records every incoming [`ChatRequest`] and
/// replies with a one-chunk `EndTurn`. The recorded request lets tests
/// assert on `system_prompt` after all middlewares (including the
/// internal `SystemPromptMiddleware`) have run — user middlewares run
/// before SP-assembly, so they can't observe the rendered prompt.
#[derive(Clone, Default)]
struct CapturingModel {
    last_request: Arc<Mutex<Option<ChatRequest>>>,
}

impl CapturingModel {
    fn new() -> Self {
        Self::default()
    }

    fn last_system_prompt(&self) -> String {
        self.last_request
            .lock()
            .unwrap()
            .as_ref()
            .expect("CapturingModel: chat_stream was never called")
            .system_prompt
            .as_ref()
            .map(|sp| sp.as_text())
            .unwrap_or_default()
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
        *self.last_request.lock().unwrap() = Some(req);
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

fn write_tempfile(content: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().expect("create tempfile");
    write!(f, "{content}").expect("write tempfile");
    f
}

fn occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

#[ailoop_tool(description = "create an Excel file")]
async fn excel_create(_path: String) -> i32 {
    0
}

#[ailoop_tool(description = "read an Excel file")]
async fn excel_read(_path: String) -> i32 {
    0
}

#[ailoop_tool(description = "edit an Excel file")]
async fn excel_edit(_path: String) -> i32 {
    0
}

#[ailoop_tool(description = "find data in an Excel file")]
async fn excel_find(_query: String) -> i32 {
    0
}

#[ailoop_tool(description = "send an email")]
async fn send_email(_to: String) -> i32 {
    0
}

#[ailoop_tool(description = "search the web")]
async fn web_search(_q: String) -> i32 {
    0
}

const EXCEL_GUIDE: &str = "EXCEL_GUIDE_SENTINEL: rules for the excel family";
const EMAIL_GUIDE: &str = "EMAIL_GUIDE_SENTINEL: rules for outgoing email";
const SOLO_GUIDE: &str = "SOLO_GUIDE_SENTINEL: rules for the single tool";

/// A group of four tools sharing one guide: when **every** tool in the
/// group is active, the section must be appended exactly once — not
/// once per tool. This pins the fix for the four-times-duplicated guide
/// the field-keyed HashMap produced.
#[tokio::test]
async fn group_guide_renders_once_when_all_tools_active() {
    let guide = write_tempfile(EXCEL_GUIDE);
    let model = CapturingModel::new();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        .tool(ExcelCreate)
        .tool(ExcelRead)
        .tool(ExcelEdit)
        .tool(ExcelFind)
        .tools_with_prompt_file(
            ["excel_create", "excel_read", "excel_edit", "excel_find"],
            guide.path(),
        )
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    let prompt = probe.last_system_prompt();
    assert_eq!(
        occurrences(&prompt, EXCEL_GUIDE),
        1,
        "guide must appear exactly once when all 4 grouped tools are active; got prompt: {prompt:?}"
    );
}

/// Same group, but only a single member of the group is active for the
/// turn (the deferred-tools / `search_tools` shape). The guide must
/// still appear, exactly once.
#[tokio::test]
async fn group_guide_renders_once_when_subset_active() {
    let guide = write_tempfile(EXCEL_GUIDE);
    let model = CapturingModel::new();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        .tool(ExcelCreate)
        .tool(ExcelRead)
        .tool(ExcelEdit)
        .tool(ExcelFind)
        .tools_with_prompt_file(
            ["excel_create", "excel_read", "excel_edit", "excel_find"],
            guide.path(),
        )
        .initial_active_tools(["excel_read"])
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    let prompt = probe.last_system_prompt();
    assert_eq!(
        occurrences(&prompt, EXCEL_GUIDE),
        1,
        "guide must appear once even when only a subset of the group is active"
    );
}

/// When none of the grouped tools is active in the request, the guide
/// must be absent — the section is gated on the group's intersection
/// with `req.tools` being non-empty.
#[tokio::test]
async fn group_guide_absent_when_no_member_active() {
    let guide = write_tempfile(EXCEL_GUIDE);
    let model = CapturingModel::new();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        .tool(ExcelCreate)
        .tool(ExcelRead)
        .tool(ExcelEdit)
        .tool(ExcelFind)
        .tool(WebSearch)
        .tools_with_prompt_file(
            ["excel_create", "excel_read", "excel_edit", "excel_find"],
            guide.path(),
        )
        // Activate something outside the group so the active set is
        // non-empty but the intersection with the group is.
        .initial_active_tools(["web_search"])
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    let prompt = probe.last_system_prompt();
    assert_eq!(
        occurrences(&prompt, EXCEL_GUIDE),
        0,
        "guide must not appear when no member of its group is active"
    );
}

/// Two groups with distinct guides: the rendered system prompt must
/// emit them in **group-registration order**, regardless of the order
/// the tools themselves appear in `req.tools`. To make the test
/// meaningful, we register the email group first but register the
/// email tool *after* the excel tools — so `req.tools` would list
/// excel tools before email, while the section order should be the
/// group order (email first).
#[tokio::test]
async fn render_order_follows_group_registration_order() {
    let excel_guide = write_tempfile(EXCEL_GUIDE);
    let email_guide = write_tempfile(EMAIL_GUIDE);
    let model = CapturingModel::new();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        // Register the excel tools first → they appear first in
        // `req.tools` (ToolRegistry iterates in insertion order).
        .tool(ExcelCreate)
        .tool(ExcelRead)
        .tool(SendEmail)
        // Register the EMAIL group before the EXCEL group → email
        // section should render first.
        .tools_with_prompt_file(["send_email"], email_guide.path())
        .tools_with_prompt_file(["excel_create", "excel_read"], excel_guide.path())
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    let prompt = probe.last_system_prompt();
    let email_pos = prompt
        .find(EMAIL_GUIDE)
        .expect("email guide must appear in prompt");
    let excel_pos = prompt
        .find(EXCEL_GUIDE)
        .expect("excel guide must appear in prompt");
    assert!(
        email_pos < excel_pos,
        "group registration order must drive section order: email registered first → email section first. prompt: {prompt:?}"
    );
    assert_eq!(occurrences(&prompt, EMAIL_GUIDE), 1);
    assert_eq!(occurrences(&prompt, EXCEL_GUIDE), 1);
}

/// The base system prompt (set with `.system_prompt(...)`) must render
/// *before* any per-group section, matching the contract that
/// `SystemPromptMiddleware` appends group sections to a clone of the
/// base prompt.
#[tokio::test]
async fn base_system_prompt_renders_before_group_sections() {
    const BASE: &str = "BASE_PROMPT_SENTINEL";
    let guide = write_tempfile(EXCEL_GUIDE);
    let model = CapturingModel::new();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        .system_prompt(BASE)
        .tool(ExcelCreate)
        .tool(ExcelRead)
        .tools_with_prompt_file(["excel_create", "excel_read"], guide.path())
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    let prompt = probe.last_system_prompt();
    let base_pos = prompt.find(BASE).expect("base must appear");
    let guide_pos = prompt.find(EXCEL_GUIDE).expect("group section must appear");
    assert!(
        base_pos < guide_pos,
        "base prompt must precede group sections; got: {prompt:?}"
    );
}

/// `tool_with_prompt_file` and `tools_with_prompt_file` must coexist
/// cleanly: a single-tool registration via the legacy API and a
/// multi-tool group via the new API render independently, each
/// section appearing once when its tool(s) are active. This pins the
/// no-regression contract for the 1:1 API.
#[tokio::test]
async fn legacy_single_tool_and_group_coexist() {
    let solo_guide = write_tempfile(SOLO_GUIDE);
    let excel_guide = write_tempfile(EXCEL_GUIDE);
    let model = CapturingModel::new();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        .tool_with_prompt_file(WebSearch, solo_guide.path())
        .tool(ExcelCreate)
        .tool(ExcelRead)
        .tools_with_prompt_file(["excel_create", "excel_read"], excel_guide.path())
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    let prompt = probe.last_system_prompt();
    assert_eq!(
        occurrences(&prompt, SOLO_GUIDE),
        1,
        "single-tool guide must appear exactly once"
    );
    assert_eq!(
        occurrences(&prompt, EXCEL_GUIDE),
        1,
        "group guide must appear exactly once"
    );
    // Registration order: solo group first, then excel group → that
    // should be the order in the rendered prompt.
    let solo_pos = prompt.find(SOLO_GUIDE).unwrap();
    let excel_pos = prompt.find(EXCEL_GUIDE).unwrap();
    assert!(solo_pos < excel_pos);
}

/// No-regression: the legacy 1:1 `tool_with_prompt_file` must still
/// gate its section on the registered tool being active. When the tool
/// is deactivated for the turn, the section must not appear.
#[tokio::test]
async fn legacy_tool_with_prompt_file_gates_on_active_tool() {
    let solo_guide = write_tempfile(SOLO_GUIDE);
    let model = CapturingModel::new();
    let probe = model.clone();

    let mut chat = Conversation::builder(model)
        .tool_with_prompt_file(WebSearch, solo_guide.path())
        .tool(SendEmail)
        // Activate only the unrelated tool so `web_search` is in the
        // catalog but not in `req.tools` for this turn.
        .initial_active_tools(["send_email"])
        .build()
        .expect("build");

    chat.run("hi").await.expect("run");

    let prompt = probe.last_system_prompt();
    assert_eq!(
        occurrences(&prompt, SOLO_GUIDE),
        0,
        "legacy single-tool guide must NOT appear when its tool is inactive"
    );
}

/// `tools_with_prompt_file([], …)` is meaningless (a group with no
/// tools could never fire). The builder must surface an error rather
/// than silently dropping the section.
#[tokio::test]
async fn empty_tool_group_surfaces_build_error() {
    let guide = write_tempfile(EXCEL_GUIDE);
    let model = CapturingModel::new();

    let result = Conversation::builder(model)
        .tools_with_prompt_file(Vec::<String>::new(), guide.path())
        .build();

    match result {
        Err(ailoop::BuildError::EmptyToolGroup) => {}
        Err(other) => panic!("expected BuildError::EmptyToolGroup, got {other:?}"),
        Ok(_) => panic!("expected empty tool group to fail the builder"),
    }
}
