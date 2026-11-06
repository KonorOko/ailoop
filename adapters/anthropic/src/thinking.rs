//! How a [`ReasoningEffort`] reaches the Messages API.
//!
//! Claude 4.6 and later take adaptive thinking
//! (`thinking: {"type":"adaptive"}` + `output_config.effort`) and reject
//! `thinking.budget_tokens` with a 400. Claude 4.5 and earlier, and the
//! Claude 3 family, only understand `budget_tokens`. The adapter keeps a
//! closed list of those legacy ids and treats every other id, including
//! unknown future ones, as adaptive.

use ailoop_core::ReasoningEffort;
use serde_json::{Value, json};

/// Which wire shape [`AnthropicChatModel`](crate::AnthropicChatModel)
/// uses to send a [`ReasoningEffort`].
///
/// By default the adapter picks it from the model id (see
/// [`AnthropicChatModel::thinking_mode`](crate::AnthropicChatModel::thinking_mode)).
/// Set it explicitly for an id the adapter does not recognise, such as a
/// custom alias behind a proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ThinkingMode {
    /// Legacy extended thinking:
    /// `thinking: {"type":"enabled","budget_tokens":N}`, or
    /// `{"type":"disabled"}` for [`ReasoningEffort::Minimal`].
    Budget,
    /// Adaptive thinking: `thinking: {"type":"adaptive"}` plus
    /// `output_config.effort`.
    Adaptive,
}

/// Resolved shape for one request. `Adaptive` carries whether the model
/// understands `"xhigh"` (Claude Opus 4.6 and Sonnet 4.6 do not).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    Budget,
    Adaptive { xhigh: bool },
}

impl Shape {
    pub(crate) fn resolve(model: &str, override_mode: Option<ThinkingMode>) -> Self {
        let id = ModelId::parse(model);
        match override_mode {
            Some(ThinkingMode::Budget) => Shape::Budget,
            Some(ThinkingMode::Adaptive) => Shape::Adaptive {
                xhigh: !id.is_some_and(ModelId::is_4_6),
            },
            None => match id {
                Some(id) if id.is_legacy() => Shape::Budget,
                Some(id) => Shape::Adaptive {
                    xhigh: !id.is_4_6(),
                },
                None => Shape::Adaptive { xhigh: true },
            },
        }
    }
}

/// `claude-{family}-{major}[-{minor}]`, with the family and version
/// read from any of the id spellings in use: bare (`claude-opus-4-5`),
/// dated (`claude-sonnet-4-5-20250929`), Vertex (`claude-opus-4-5@20251101`)
/// and Bedrock (`anthropic.claude-haiku-4-5-20251001-v1:0`,
/// `us.anthropic.claude-…`). `claude-3-*` ids put the version first
/// and are recorded with `claude3 = true`.
#[derive(Debug, Clone, Copy)]
struct ModelId {
    claude3: bool,
    major: u32,
    minor: Option<u32>,
}

impl ModelId {
    fn parse(model: &str) -> Option<Self> {
        let lower = model.to_ascii_lowercase();
        let start = lower.find("claude-")?;
        let rest = &lower[start + "claude-".len()..];
        let rest = rest.split(['@', ':']).next().unwrap_or(rest);
        let mut parts = rest.split('-');
        let first = parts.next()?;
        if first.starts_with('3') {
            return Some(Self {
                claude3: true,
                major: 3,
                minor: None,
            });
        }
        if !matches!(first, "opus" | "sonnet" | "haiku") {
            return None;
        }
        let major = parts.next()?.parse().ok()?;
        // A minor version is one or two digits; an 8-digit segment is a
        // snapshot date (`claude-sonnet-4-20250514` is Sonnet 4.0).
        let minor = parts
            .next()
            .filter(|p| (1..=2).contains(&p.len()))
            .and_then(|p| p.parse().ok());
        Some(Self {
            claude3: false,
            major,
            minor,
        })
    }

    /// Claude 3.x, and every Claude 4 model up to and including 4.5.
    fn is_legacy(self) -> bool {
        self.claude3 || self.major < 4 || (self.major == 4 && self.minor.unwrap_or(0) <= 5)
    }

    fn is_4_6(self) -> bool {
        !self.claude3 && self.major == 4 && self.minor == Some(6)
    }
}

/// The body fields one effort produces: `thinking`, and on adaptive
/// models `output_config.effort`. `None` for unknown future variants.
pub(crate) struct Lowered {
    pub thinking: Value,
    pub effort: Option<&'static str>,
}

pub(crate) fn lower(effort: ReasoningEffort, shape: Shape) -> Option<Lowered> {
    match shape {
        Shape::Budget => {
            let budget = match effort {
                ReasoningEffort::Minimal => {
                    return Some(Lowered {
                        thinking: json!({ "type": "disabled" }),
                        effort: None,
                    });
                }
                ReasoningEffort::Low => 1024,
                ReasoningEffort::Medium => 4096,
                ReasoningEffort::High => 16384,
                ReasoningEffort::XHigh => 32768,
                ReasoningEffort::Max => 65536,
                ReasoningEffort::Budget(n) => n,
                _ => return None,
            };
            Some(Lowered {
                thinking: json!({ "type": "enabled", "budget_tokens": budget }),
                effort: None,
            })
        }
        Shape::Adaptive { xhigh } => {
            let level = match effort {
                // Adaptive models cannot switch thinking off (several
                // reject `disabled` with a 400), so Minimal asks for the
                // least thinking instead.
                ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
                ReasoningEffort::XHigh if xhigh => "xhigh",
                ReasoningEffort::XHigh => "high",
                ReasoningEffort::Max => "max",
                ReasoningEffort::Budget(n) if n < 2048 => "low",
                ReasoningEffort::Budget(n) if n < 8192 => "medium",
                ReasoningEffort::Budget(_) => "high",
                _ => return None,
            };
            Some(Lowered {
                thinking: json!({ "type": "adaptive" }),
                effort: Some(level),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auto(model: &str) -> Shape {
        Shape::resolve(model, None)
    }

    const ADAPTIVE: Shape = Shape::Adaptive { xhigh: true };
    const ADAPTIVE_4_6: Shape = Shape::Adaptive { xhigh: false };

    #[test]
    fn claude_3_family_is_legacy() {
        for id in [
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-latest",
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-latest",
        ] {
            assert_eq!(auto(id), Shape::Budget, "{id}");
        }
    }

    #[test]
    fn claude_4_0_and_4_1_are_legacy() {
        for id in [
            "claude-opus-4-20250514",
            "claude-opus-4-0",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-0",
            "claude-opus-4-1",
            "claude-opus-4-1-20250805",
        ] {
            assert_eq!(auto(id), Shape::Budget, "{id}");
        }
    }

    #[test]
    fn claude_4_5_family_is_legacy() {
        for id in [
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
        ] {
            assert_eq!(auto(id), Shape::Budget, "{id}");
        }
    }

    #[test]
    fn claude_4_6_is_adaptive_without_xhigh() {
        for id in ["claude-opus-4-6", "claude-sonnet-4-6"] {
            assert_eq!(auto(id), ADAPTIVE_4_6, "{id}");
        }
    }

    #[test]
    fn claude_4_7_and_later_are_adaptive() {
        for id in [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-sonnet-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
            "claude-fable-5",
            "claude-fable-5-1",
        ] {
            assert_eq!(auto(id), ADAPTIVE, "{id}");
        }
    }

    #[test]
    fn unknown_ids_are_adaptive() {
        for id in [
            "claude",
            "claude-next",
            "my-proxy-alias",
            "claude-opus-latest",
        ] {
            assert_eq!(auto(id), ADAPTIVE, "{id}");
        }
    }

    #[test]
    fn platform_spellings_are_recognised() {
        assert_eq!(auto("claude-opus-4-5@20251101"), Shape::Budget);
        assert_eq!(
            auto("anthropic.claude-haiku-4-5-20251001-v1:0"),
            Shape::Budget
        );
        assert_eq!(
            auto("us.anthropic.claude-3-7-sonnet-20250219-v1:0"),
            Shape::Budget
        );
        assert_eq!(auto("anthropic.claude-opus-5-5"), ADAPTIVE);
        assert_eq!(auto("CLAUDE-SONNET-4-6"), ADAPTIVE_4_6);
    }

    #[test]
    fn override_wins_over_the_id() {
        assert_eq!(
            Shape::resolve("claude-opus-5-5", Some(ThinkingMode::Budget)),
            Shape::Budget
        );
        assert_eq!(
            Shape::resolve("claude-sonnet-4-5", Some(ThinkingMode::Adaptive)),
            ADAPTIVE
        );
        assert_eq!(
            Shape::resolve("claude-opus-4-6", Some(ThinkingMode::Adaptive)),
            ADAPTIVE_4_6
        );
        assert_eq!(
            Shape::resolve("my-alias", Some(ThinkingMode::Budget)),
            Shape::Budget
        );
    }
}
