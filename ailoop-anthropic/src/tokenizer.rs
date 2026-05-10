//! Online-calibrated [`Tokenizer`] for Anthropic providers.
//!
//! Anthropic does not publish an official open tokenizer for the
//! Claude family, so an exact offline count would require shipping a
//! heavyweight BPE table per model. What the provider *does* surface,
//! on every response, is the real input and output token count — see
//! `Usage` (`message_start` / `message_delta`).
//!
//! [`OnlineCalibratedTokenizer`] turns those measurements into a
//! cheap, self-tuning approximation: it tracks an exponential moving
//! average of the *tokens-per-char* ratio and applies it whenever a
//! caller (typically `ailoop-context::ContextManager::compact_if_needed`)
//! asks how many tokens a piece of text or a message is worth before
//! sending it to the API.
//!
//! ## Bootstrapping
//!
//! With no observations the ratio defaults to `0.25` (the rule-of-
//! thumb "4 chars per token" guidance every provider gives), matching
//! the [`CharTokenizer`] fallback. As real `Usage` reports come back
//! the ratio drifts toward whatever the deployed model actually
//! charges — for English text on Sonnet/Haiku this is typically a
//! touch lower than 0.25; on languages with many short tokens or
//! heavy punctuation it can be markedly higher.
//!
//! ## Wiring it up
//!
//! The tokenizer cannot observe `Usage` on its own. Drop a small
//! middleware into your conversation that:
//! 1. Records the character length of the outgoing request before
//!    `on_chat_request` returns (or sums each delta on the way back
//!    for output sampling).
//! 2. Calls [`OnlineCalibratedTokenizer::observe`] on
//!    `StreamChunk::TurnFinished { usage, .. }` with the recorded
//!    char count and `usage.input_tokens` (and / or `output_tokens`).
//!
//! The `Arc<OnlineCalibratedTokenizer>` is shared between the
//! middleware and the [`ContextManagerBuilder::tokenizer`] passed to
//! the conversation builder, so both views read and write the same
//! EMA. See the crate-level docs for a worked example.
//!
//! [`Tokenizer`]: ailoop_core::Tokenizer
//! [`CharTokenizer`]: ailoop_core::CharTokenizer
//! [`ContextManagerBuilder::tokenizer`]: https://docs.rs/ailoop-context

use std::sync::{Arc, RwLock};

use ailoop_core::Tokenizer;

/// Default initial ratio: 4 chars per token, matching the
/// `CharTokenizer` fallback so an un-calibrated
/// `OnlineCalibratedTokenizer` behaves identically to it until real
/// observations arrive.
const DEFAULT_INITIAL_RATIO: f32 = 0.25;

/// Default EMA smoothing factor. With `alpha = 0.2` each new
/// observation pulls the ratio 20% of the way to the freshly observed
/// value, giving the first ~10 observations enough weight to settle
/// near the true rate without single noisy samples whipsawing the
/// estimate.
const DEFAULT_ALPHA: f32 = 0.2;

/// Online-calibrated `Tokenizer` for Anthropic providers.
///
/// See module-level docs for the wiring pattern. Internals:
///
/// - The ratio (tokens / char) is held under [`RwLock`] because
///   [`Tokenizer::count_text`] takes `&self` (read-mostly) and
///   [`Self::observe`] needs to mutate the running EMA.
/// - The struct is cheap to `Clone`-via-`Arc`; share one instance
///   between the calibration middleware and the context manager.
pub struct OnlineCalibratedTokenizer {
    ratio: Arc<RwLock<f32>>,
    alpha: f32,
}

impl OnlineCalibratedTokenizer {
    /// Build a tokenizer seeded with the rule-of-thumb 4 chars/token
    /// ratio and the default EMA smoothing factor (`alpha = 0.2`).
    pub fn new() -> Self {
        Self::with_initial_ratio(DEFAULT_INITIAL_RATIO)
    }

    /// Build a tokenizer with a caller-supplied starting ratio. Useful
    /// when you have prior knowledge of how a particular language or
    /// content type tokenizes (Japanese / code / heavy markdown can
    /// drift well above `0.25`). `ratio` is the expected
    /// tokens-per-char rate; values must be `> 0`.
    pub fn with_initial_ratio(ratio: f32) -> Self {
        assert!(
            ratio > 0.0,
            "OnlineCalibratedTokenizer ratio must be > 0 (got {ratio})"
        );
        Self {
            ratio: Arc::new(RwLock::new(ratio)),
            alpha: DEFAULT_ALPHA,
        }
    }

    /// Override the EMA smoothing factor. Higher = more weight on the
    /// latest observation (faster convergence, noisier); lower =
    /// more weight on history (smoother, slower to track regime
    /// changes). Must be in `(0, 1]`.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        assert!(
            alpha > 0.0 && alpha <= 1.0,
            "OnlineCalibratedTokenizer alpha must be in (0, 1] (got {alpha})"
        );
        self.alpha = alpha;
        self
    }

    /// Update the EMA with a single (text length, observed tokens)
    /// sample. `text_chars` is the byte length of the text the
    /// provider tokenized; `observed_tokens` is the tokens it billed
    /// for that text. Samples with `text_chars == 0` are ignored
    /// (division by zero, no signal anyway).
    ///
    /// Thread-safe. Multiple middlewares can call this concurrently;
    /// the underlying lock serializes them.
    pub fn observe(&self, text_chars: usize, observed_tokens: u32) {
        if text_chars == 0 {
            return;
        }
        let sample = observed_tokens as f32 / text_chars as f32;
        let mut ratio = self.ratio.write().expect("tokenizer ratio lock poisoned");
        *ratio = (1.0 - self.alpha) * (*ratio) + self.alpha * sample;
    }

    /// Current tokens-per-char ratio. Useful for diagnostics and
    /// tests. Returns `DEFAULT_INITIAL_RATIO` (0.25) before any
    /// observations have been fed in.
    pub fn ratio(&self) -> f32 {
        *self.ratio.read().expect("tokenizer ratio lock poisoned")
    }
}

impl Default for OnlineCalibratedTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for OnlineCalibratedTokenizer {
    fn count_text(&self, text: &str) -> usize {
        let ratio = self.ratio();
        // `ceil` so the budget never undershoots on tiny strings — a
        // single-char text under ratio 0.25 still counts as 1 token,
        // matching what the model would charge for any non-empty
        // input.
        ((text.len() as f32) * ratio).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no observations the tokenizer behaves identically to the
    /// `CharTokenizer` rule-of-thumb (4 chars per token), so swapping
    /// it in for a fresh conversation does not silently shift the
    /// budget.
    #[test]
    fn defaults_to_quarter_ratio() {
        let t = OnlineCalibratedTokenizer::new();
        assert!(
            (t.ratio() - DEFAULT_INITIAL_RATIO).abs() < f32::EPSILON,
            "fresh tokenizer ratio: {}",
            t.ratio()
        );
        // 12 chars at 0.25 = 3 tokens. `ceil` keeps non-empty inputs
        // from rounding to zero.
        assert_eq!(t.count_text("hello world!"), 3);
        assert_eq!(t.count_text("a"), 1);
        assert_eq!(t.count_text(""), 0);
    }

    /// Streaming many same-rate samples must drag the EMA toward the
    /// observed rate. We don't require exact convergence, just that
    /// the ratio is meaningfully closer to the truth than the default
    /// after a handful of observations.
    #[test]
    fn observe_converges_toward_steady_state() {
        let t = OnlineCalibratedTokenizer::new();
        // Truth ratio: 0.5 tokens/char (twice the default).
        for _ in 0..50 {
            t.observe(1000, 500);
        }
        let r = t.ratio();
        assert!(
            (r - 0.5).abs() < 0.01,
            "expected EMA to converge near 0.5, got {r}"
        );
    }

    /// `observe(0, _)` must be a no-op — there is nothing to learn
    /// from a zero-length sample, and treating it as one would be a
    /// division by zero.
    #[test]
    fn observe_ignores_empty_text() {
        let t = OnlineCalibratedTokenizer::new();
        t.observe(0, 999);
        assert!((t.ratio() - DEFAULT_INITIAL_RATIO).abs() < f32::EPSILON);
    }

    /// `with_alpha(1.0)` is the "trust each sample completely" mode:
    /// after one observation the ratio equals that sample. Useful for
    /// asserting the math more precisely than the smoothed EMA path.
    #[test]
    fn with_alpha_one_jumps_to_each_sample() {
        let t = OnlineCalibratedTokenizer::new().with_alpha(1.0);
        t.observe(100, 30);
        assert!((t.ratio() - 0.30).abs() < 1e-6);
        t.observe(100, 80);
        assert!((t.ratio() - 0.80).abs() < 1e-6);
    }

    /// Custom starting ratios are honoured for callers who already
    /// know the model's regime and want to skip the bootstrap noise.
    #[test]
    fn with_initial_ratio_seeds_estimate() {
        let t = OnlineCalibratedTokenizer::with_initial_ratio(0.5);
        // 100 chars at 0.5 = 50 tokens.
        assert_eq!(t.count_text(&"x".repeat(100)), 50);
    }

    /// End-to-end shape used by a calibration middleware: pretend a
    /// stream of `Usage` reports comes back from the provider for
    /// requests of known character length, and confirm that
    /// `count_text` shifts in the right direction. We drive the
    /// sampling shape directly because the wiring middleware (request
    /// chars in / `Usage` out) is per-application.
    #[test]
    fn online_calibration_tracks_streamed_usage() {
        let t = OnlineCalibratedTokenizer::new();

        // Default ratio is 0.25, so a 200-char input is sized at 50
        // tokens.
        assert_eq!(t.count_text(&"x".repeat(200)), 50);

        // Stream of provider responses telling us the true rate is ~0.4.
        for _ in 0..40 {
            t.observe(200, 80);
        }

        // Same 200-char input now sizes much closer to 80 tokens.
        let new_estimate = t.count_text(&"x".repeat(200));
        assert!(
            new_estimate >= 75 && new_estimate <= 81,
            "expected ~80-token estimate after calibration, got {new_estimate}"
        );
    }
}
