//! Deferred tools: register many tools but only expose a meta-tool
//! `search_tools` to the model up front. The model discovers and
//! activates the rest on demand by calling `search_tools` — the
//! handler uses the [`ToolContext`] to flip tools on, and the next
//! turn's `req.tools` automatically picks up the new active set.
//!
//! This pattern is useful when:
//!
//! - Total tool count is large enough that sending every schema on
//!   every turn wastes input tokens.
//! - Tool relevance varies per task and the model can self-select.
//! - You'd otherwise reach for middleware that filters `req.tools`
//!   from a shared `Arc<Mutex<HashSet<String>>>`.

use ailoop::{Conversation, RetryingModel, ToolContext, ailoop_tool};
use ailoop_anthropic::AnthropicClient;

#[ailoop_tool(
    description = "List or activate tools that are registered but not yet visible to you. \
                   Pass an empty query to see every hidden tool with its description; pass a \
                   keyword to activate only the matching ones. Activated tools become \
                   available on the next turn."
)]
async fn search_tools(query: String, ctx: &ToolContext) -> String {
    let inactive = ctx.tools().list_inactive();
    if inactive.is_empty() {
        return "No hidden tools — all registered tools are already active.".to_string();
    }

    let q = query.trim().to_lowercase();
    let mut activated = Vec::new();
    let mut listed = Vec::new();

    for def in &inactive {
        let matches = q.is_empty()
            || def.name.to_lowercase().contains(&q)
            || def.description.to_lowercase().contains(&q);
        if !matches {
            continue;
        }
        if q.is_empty() {
            listed.push(format!("- {} — {}", def.name, def.description));
        } else if ctx.tools().activate(&def.name).is_ok() {
            activated.push(def.name.clone());
        }
    }

    if q.is_empty() {
        format!("Hidden tools ({}):\n{}", listed.len(), listed.join("\n"))
    } else if activated.is_empty() {
        format!("No hidden tools matched '{q}'.")
    } else {
        format!(
            "Activated {} tool(s) matching '{q}': {}. They are now available on the next turn.",
            activated.len(),
            activated.join(", ")
        )
    }
}

#[ailoop_tool(description = "Add two integers")]
async fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[ailoop_tool(description = "Multiply two integers")]
async fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

#[ailoop_tool(description = "Compute the great-circle distance in km between two points on Earth")]
async fn haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0_f64;
    let to_rad = |d: f64| d * std::f64::consts::PI / 180.0;
    let (p1, p2) = (to_rad(lat1), to_rad(lat2));
    let dp = to_rad(lat2 - lat1);
    let dl = to_rad(lon2 - lon1);
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

#[ailoop_tool(description = "Reverse a string")]
async fn reverse(text: String) -> String {
    text.chars().rev().collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = RetryingModel::new(AnthropicClient::from_env()?.model("claude-sonnet-4-6"));

    // Register every tool, but expose only `search_tools` initially.
    // The model has to discover and activate the others on demand.
    let mut chat = Conversation::builder(model)
        .system_prompt(
            "You have access to a `search_tools` meta-tool. The list of tools you see is \
             intentionally incomplete — use `search_tools` with a relevant keyword to \
             surface and enable additional tools whenever you need capabilities you don't \
             currently have.",
        )
        .tool(SearchTools)
        .tool(Add)
        .tool(Multiply)
        .tool(Haversine)
        .tool(Reverse)
        .initial_active_tools(["search_tools"])
        .build()?;

    let prompt = "What is the great-circle distance from Mexico City \
                  (19.4326, -99.1332) to Tokyo (35.6762, 139.6503)? \
                  Use whichever tool is appropriate.";

    let outcome = chat.run(prompt).await?;
    println!("{}", outcome.final_text.unwrap_or_default());

    Ok(())
}
