//! Per-tool and per-group system-prompt sections.
//!
//! Three patterns shown side-by-side:
//!
//! 1. **1:1** — one tool, one guide, via
//!    [`tool_with_prompt_file`]. The section is appended to the
//!    system prompt only when that tool is active in the turn.
//! 2. **Group** — N tools share one guide, via
//!    [`tools_with_prompt_file`]. The section appears at most once
//!    per turn regardless of how many of the grouped tools are
//!    active. Tools are registered separately with `.tool(...)`.
//! 3. **Layered** — a tool participates in both a single-tool guide
//!    *and* a group guide. Each section is independent and renders
//!    in the order its group was registered on the builder.
//!
//! Run with `ANTHROPIC_API_KEY=… cargo run -p tool-prompts`.
//!
//! [`tool_with_prompt_file`]: ailoop::ConversationBuilder::tool_with_prompt_file
//! [`tools_with_prompt_file`]: ailoop::ConversationBuilder::tools_with_prompt_file

use ailoop::{Conversation, RetryingModel, ailoop_tool};
use ailoop_anthropic::AnthropicClient;

// ---------- 1:1 (single tool, single guide) ----------------------------------

#[ailoop_tool(description = "Run a web search and return the top result snippets.")]
async fn web_search(query: String) -> String {
    // Stub — pretend we fetched something useful.
    format!("(stub) results for '{query}'")
}

// ---------- Group: every editor shares the same edit-files guide -------------

#[ailoop_tool(description = "Read an Excel sheet and return its contents as text.")]
async fn excel_read(path: String) -> String {
    format!("(stub) read excel {path}")
}

#[ailoop_tool(description = "Edit an Excel sheet by writing a value to a cell.")]
async fn excel_edit(path: String, cell: String, value: String) -> String {
    format!("(stub) wrote {value} to {cell} in {path}")
}

#[ailoop_tool(description = "Edit a Word document by replacing a span of text.")]
async fn word_edit(path: String, find: String, replace: String) -> String {
    format!("(stub) replaced '{find}' with '{replace}' in {path}")
}

#[ailoop_tool(description = "Edit a PDF by replacing text on a given page.")]
async fn pdf_edit(path: String, page: u32, find: String, replace: String) -> String {
    format!("(stub) replaced '{find}' with '{replace}' on page {page} of {path}")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = RetryingModel::new(AnthropicClient::from_env()?.model("claude-sonnet-4-6"));

    // Resolve paths relative to the example's manifest dir so the
    // binary can be launched from anywhere in the workspace.
    let prompts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("prompts");

    let mut chat = Conversation::builder(model)
        .system_prompt(
            "You are a desktop-automation assistant. You can search the web and \
             edit Office documents. Follow every guide that applies to the tools \
             you choose to call.",
        )
        // ---- 1:1 ----
        // `web_search` is the only tool that needs `web-search.md`, so
        // the 1:1 sugar fits: registers the tool AND attaches the
        // guide in one call.
        .tool_with_prompt_file(WebSearch, prompts_dir.join("web-search.md"))
        // ---- Group + layering ----
        // The four editor tools share `edit-files.md`. The Excel
        // family additionally has its own quirks captured in
        // `excel.md`. Tools are registered with plain `.tool(...)`;
        // the guides are attached as groups in a separate step.
        .tool(ExcelRead)
        .tool(ExcelEdit)
        .tool(WordEdit)
        .tool(PdfEdit)
        // Shared group guide — appears once per turn when at least
        // one editor is active, no matter how many.
        .tools_with_prompt_file(
            ["excel_read", "excel_edit", "word_edit", "pdf_edit"],
            prompts_dir.join("edit-files.md"),
        )
        // Layered Excel-only guide — also appears at most once. With
        // both groups registered, render order follows the
        // registration order on the builder: edit-files.md first
        // (above), then excel.md (below).
        .tools_with_prompt_file(["excel_read", "excel_edit"], prompts_dir.join("excel.md"))
        .build()?;

    // A prompt that exercises the editor group: `excel_edit` is
    // active, so the rendered system prompt for this turn is:
    //
    //   <base prompt>
    //   <edit-files.md>   (group with editors; intersection: excel_edit)
    //   <excel.md>        (excel-only group; intersection: excel_edit)
    //
    // `web-search.md` does NOT appear because `web_search` was not
    // called — the section is gated on its tool being in `req.tools`
    // for the turn.
    let outcome = chat
        .run("Set cell B7 of /tmp/report.xlsx to 42 and tell me what you did.")
        .await?;
    println!("{}", outcome.final_text.unwrap_or_default());

    Ok(())
}
