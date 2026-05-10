//! Tool name composition and sanitization.
//!
//! MCP tool names get prefixed `mcp__<server_label>__<tool_name>` so
//! tools from multiple servers can coexist in one [`ToolRegistry`]
//! without collision (matches the convention used by Claude Desktop).
//! Anthropic's tool-name validator only accepts `^[a-zA-Z0-9_-]{1,64}$`,
//! so we also sanitize invalid characters and truncate-with-hash when
//! the composed name overflows.
//!
//! [`ToolRegistry`]: ailoop_tools::ToolRegistry

const MAX_TOOL_NAME_LEN: usize = 64;

/// Build the engine-facing tool name for a tool discovered on
/// `server_label`. Always returns a string matching
/// `^[a-zA-Z0-9_-]{1,64}$`.
pub(crate) fn compose(server_label: &str, tool_name: &str) -> String {
    let raw = format!("mcp__{}__{}", server_label, tool_name);
    let sanitized = sanitize(&raw);

    if sanitized.len() <= MAX_TOOL_NAME_LEN {
        sanitized
    } else {
        truncate_with_hash(&sanitized, MAX_TOOL_NAME_LEN)
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Truncate to `max` chars, reserving the last 5 (`_xxxx`) for a short
/// deterministic hash so distinct long names don't collide after
/// truncation.
fn truncate_with_hash(s: &str, max: usize) -> String {
    debug_assert!(max >= 5, "max must leave room for the `_xxxx` suffix");
    let head_len = max - 5;
    let head = &s[..head_len.min(s.len())];
    format!("{}_{}", head, short_hash(s))
}

/// FNV-1a 32-bit, keep low 16 bits as 4 hex chars. Deterministic across
/// runs (unlike `DefaultHasher`'s per-process randomized seed).
fn short_hash(s: &str) -> String {
    let mut hash: u32 = 2_166_136_261;
    for b in s.bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    format!("{:04x}", hash & 0xffff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composes_with_default_pattern() {
        assert_eq!(
            compose("time", "get_current_time"),
            "mcp__time__get_current_time"
        );
    }

    #[test]
    fn sanitizes_invalid_characters() {
        // `.` and `:` are not in the Anthropic allowed set; both go to `_`.
        assert_eq!(compose("svc.x", "do:thing"), "mcp__svc_x__do_thing");
    }

    #[test]
    fn truncates_with_hash_when_too_long() {
        let huge = "a".repeat(200);
        let composed = compose("time", &huge);
        assert_eq!(composed.len(), MAX_TOOL_NAME_LEN);
        // Suffix is `_xxxx` (underscore + 4 hex).
        let suffix = &composed[composed.len() - 5..];
        assert!(suffix.starts_with('_'));
        assert!(suffix[1..].chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn truncation_is_deterministic_per_input() {
        let huge = "x".repeat(300);
        let a = compose("s", &huge);
        let b = compose("s", &huge);
        assert_eq!(a, b);
    }

    #[test]
    fn truncation_disambiguates_distinct_inputs() {
        // Two long names sharing a long common prefix would collide
        // under naive truncation; the FNV hash separates them.
        let common = "z".repeat(100);
        let a = format!("{common}__alpha");
        let b = format!("{common}__beta");
        assert_ne!(compose("s", &a), compose("s", &b));
    }

    #[test]
    fn output_always_within_anthropic_charset() {
        for input in [
            ("时间", "tool"),                 // non-ASCII server
            ("ok", "weird name with spaces"), // spaces
            ("ok", "with/slash"),
            ("ok", "with.dot"),
        ] {
            let out = compose(input.0, input.1);
            assert!(
                out.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "compose({input:?}) returned {out:?} which contains invalid chars",
            );
            assert!((1..=MAX_TOOL_NAME_LEN).contains(&out.len()));
        }
    }
}
