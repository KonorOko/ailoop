//! Cross-crate token-counting trait.
//!
//! `Tokenizer` is the single contract every ailoop crate measures
//! against when it needs to know how many tokens a piece of text or a
//! conversation is worth. It lives in `ailoop-prompts` because the
//! types that get tokenized most often — [`crate::Prompt`],
//! [`crate::PromptSection`], and [`ailoop_core::Message`] — are
//! reachable from here without pulling in higher-level crates.
//!
//! Implementations fall in two families:
//! - Offline tokenizers ship a model-specific BPE table and produce
//!   exact counts (e.g. tiktoken for OpenAI/Azure). Out of scope for
//!   the MVP — left to a follow-up.
//! - Online-calibrated tokenizers maintain an EMA over the
//!   tokens-per-char ratio observed in real provider responses
//!   (e.g. `ailoop_anthropic::OnlineCalibratedTokenizer`). Cheap and
//!   "good enough" for compaction budgets.
//!
//! [`CharTokenizer`] is a deliberately rough fallback (`len() / 4`)
//! used by `ailoop-context` when no real tokenizer has been wired up.
//! It is documented as a fallback rather than a recommended default:
//! production callers should plug in a provider-specific implementation.

use ailoop_core::{AssistantBlock, Message, ToolResultContent, UserBlock};

/// Counts tokens in text and full messages.
///
/// Implementations only have to provide [`Self::count_text`]; the
/// message-level helpers walk every block kind that contributes to
/// what the model actually sees on the wire (text, tool calls and
/// their JSON args, tool results, reasoning text and signatures,
/// redacted reasoning payloads).
///
/// Implementations must be `Send + Sync` because consumers wrap them
/// behind `Arc<dyn Tokenizer>` and use them across `await` boundaries
/// (the engine, conversation history, and middlewares are all multi-
/// task by design).
pub trait Tokenizer: Send + Sync {
    /// Count tokens in a flat string. The only required method.
    fn count_text(&self, text: &str) -> usize;

    /// Count tokens in a single [`Message`], walking every block kind
    /// the provider sees. Defaults to summing block-level
    /// [`Self::count_text`] calls; override if your tokenizer has a
    /// cheaper batch path.
    fn count_message(&self, message: &Message) -> usize {
        let mut total = 0;
        match message {
            Message::User { blocks } => {
                for block in blocks {
                    match block {
                        UserBlock::Text { text, .. } => total += self.count_text(text),
                        UserBlock::ToolResult {
                            call_id, content, ..
                        } => {
                            total += self.count_text(call_id);
                            match content {
                                ToolResultContent::Text(text) => total += self.count_text(text),
                                ToolResultContent::Error(text) => total += self.count_text(text),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Message::Assistant { blocks } => {
                for block in blocks {
                    match block {
                        AssistantBlock::Text { text, .. } => total += self.count_text(text),
                        AssistantBlock::ToolCall { id, name, args, .. } => {
                            total += self.count_text(id)
                                + self.count_text(name)
                                + self.count_text(&args.to_string());
                        }
                        AssistantBlock::Reasoning { text, signature } => {
                            total += self.count_text(text);
                            if let Some(sig) = signature {
                                total += self.count_text(sig);
                            }
                        }
                        AssistantBlock::RedactedReasoning { data } => {
                            total += self.count_text(data);
                        }
                        _ => {}
                    }
                }
            }
        }
        total
    }

    /// Count tokens in a slice of messages — the budget unit
    /// `ContextManager::compact_if_needed` measures against.
    fn count_messages(&self, messages: &[Message]) -> usize {
        messages.iter().map(|m| self.count_message(m)).sum()
    }
}

/// Fallback `Tokenizer` that approximates tokens as `text.len() / 4`.
///
/// This is the rule-of-thumb every provider documentation suggests
/// for back-of-envelope sizing — accurate enough to spot the difference
/// between "10 tokens" and "10k tokens", but **not** a substitute for
/// a real tokenizer when budgets are tight. It is the silent default
/// in `ailoop-context::ContextManagerBuilder` so dev/test code does not
/// have to wire one up; production callers should pass an explicit
/// provider-specific tokenizer
/// (e.g. `ailoop_anthropic::OnlineCalibratedTokenizer`) via
/// `ContextManagerBuilder::tokenizer`.
pub struct CharTokenizer;

impl Tokenizer for CharTokenizer {
    fn count_text(&self, text: &str) -> usize {
        text.len() / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ailoop_core::{AssistantBlock, ToolResultContent, UserBlock};
    use serde_json::json;

    /// Test tokenizer that counts whitespace-delimited words. Useful for
    /// asserting that `count_message` walks every block kind without
    /// having to reason about a chars-based fallback.
    struct WordTokenizer;

    impl Tokenizer for WordTokenizer {
        fn count_text(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    #[test]
    fn char_tokenizer_matches_legacy_len_div_four() {
        let t = CharTokenizer;
        assert_eq!(t.count_text(""), 0);
        assert_eq!(t.count_text("abcd"), 1);
        assert_eq!(t.count_text("hello world"), 2); // 11 / 4 = 2
    }

    #[test]
    fn count_message_walks_user_blocks() {
        let t = WordTokenizer;
        let msg = Message::User {
            blocks: vec![
                UserBlock::text("hello world from user"),
                UserBlock::tool_result("call_42", ToolResultContent::Text("ok done".into())),
            ],
        };
        // text(4) + call_id(1) + tool_result text(2) = 7
        assert_eq!(t.count_message(&msg), 7);
    }

    #[test]
    fn count_message_walks_assistant_blocks_including_reasoning() {
        let t = WordTokenizer;
        let msg = Message::Assistant {
            blocks: vec![
                AssistantBlock::text("two words"),
                AssistantBlock::tool_call("id_1", "tool_name", json!({"k": "v"})),
                AssistantBlock::Reasoning {
                    text: "thinking aloud".into(),
                    signature: Some("sig token".into()),
                },
                AssistantBlock::RedactedReasoning {
                    data: "redacted_blob".into(),
                },
            ],
        };
        // text(2) + tool_call: id(1) + name(1) + args.to_string()=`{"k":"v"}`(1)
        //   + reasoning(2) + signature(2) + redacted(1)
        // = 2 + 1 + 1 + 1 + 2 + 2 + 1 = 10
        assert_eq!(t.count_message(&msg), 10);
    }

    #[test]
    fn count_messages_sums_each_message() {
        let t = WordTokenizer;
        let msgs = vec![
            Message::user("one two three"),
            Message::assistant_text("four five"),
        ];
        assert_eq!(t.count_messages(&msgs), 5);
    }
}
