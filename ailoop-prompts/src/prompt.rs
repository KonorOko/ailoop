use std::{fmt::Display, path::Path};

use ailoop_core::Tokenizer;

use crate::errors::PromptError;

/// Composable system prompt: an ordered list of [`PromptSection`]s.
///
/// The render order matches the construction order. Use
/// [`Prompt::builder`] for a fluent constructor or [`Prompt::new`] +
/// [`add_section`](Self::add_section) when sections come from a loop.
/// Convert into the string the model sees with
/// [`render`](Self::render) or via the [`Display`] / `Into<String>`
/// impls.
#[derive(Debug, Clone)]
pub struct Prompt {
    sections: Vec<PromptSection>,
}

/// One section of a [`Prompt`].
///
/// Sections without a name render verbatim; named sections render with
/// a Markdown `## {name}` header followed by the content. The name is
/// load-bearing for the rendered shape — keep it short, keep it
/// stable.
#[derive(Debug, Clone)]
pub struct PromptSection {
    name: Option<String>,
    content: String,
}

impl PromptSection {
    /// Build an unnamed section. Renders as `{content}\n\n` with no
    /// header.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            name: None,
            content: content.into(),
        }
    }

    /// Build a named section. Renders as `## {name}\n\n{content}\n\n`.
    pub fn with_name(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            content: content.into(),
        }
    }

    /// Load a section's content from `path` (sync `std::fs::read_to_string`).
    /// The resulting section is unnamed; chain with
    /// [`Self::with_name`] manually if a header is wanted.
    ///
    /// Sync I/O is intentional: prompts are typically loaded once at
    /// startup. Failures surface as [`PromptError::LoadFile`] carrying
    /// the offending path and the underlying [`std::io::Error`]. Do
    /// not call from within a hot async path — wrap in
    /// `tokio::task::spawn_blocking` if the underlying filesystem may
    /// be slow.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PromptError> {
        let content = std::fs::read_to_string(&path).map_err(|source| PromptError::LoadFile {
            path: path.as_ref().to_path_buf(),
            source,
        })?;

        Ok(Self::new(content))
    }

    /// Section name supplied at construction (`None` for unnamed
    /// sections built via [`Self::new`] / [`Self::from_file`]).
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Section body text (without any header [`Prompt::render`] would
    /// inject).
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Approximate token count of this section's content under
    /// `tokenizer`. The optional `## name` header that
    /// [`Prompt::render`] would prepend is **not** included — counting
    /// the header is the rendered prompt's responsibility, not the
    /// section's. For the full prompt-level cost, use
    /// [`Prompt::token_count`].
    pub fn token_count(&self, tokenizer: &dyn Tokenizer) -> usize {
        tokenizer.count_text(&self.content)
    }
}

impl Default for Prompt {
    fn default() -> Self {
        Self::new()
    }
}

impl Prompt {
    /// Begin assembling a prompt with the fluent
    /// [`PromptBuilder`] API.
    pub fn builder() -> PromptBuilder {
        PromptBuilder::new()
    }

    /// Build an empty prompt. Equivalent to [`Self::default`].
    pub fn new() -> Self {
        Prompt { sections: vec![] }
    }

    /// Borrow the section vector in render order.
    pub fn sections(&self) -> &Vec<PromptSection> {
        &self.sections
    }

    /// Append `section` after every existing one. The section will
    /// appear last in the rendered output.
    pub fn add_section(&mut self, section: PromptSection) {
        self.sections.push(section);
    }

    /// Remove the first section whose name matches `name`. No-op when
    /// no such section exists; unnamed sections are never matched.
    pub fn remove_section(&mut self, name: &str) {
        if let Some(idx) = self
            .sections
            .iter()
            .position(|section| section.name() == Some(name))
        {
            self.sections.remove(idx);
        };
    }

    /// Render every section into one string, in order.
    ///
    /// Named sections produce `## {name}\n\n{content}\n\n`; unnamed
    /// sections produce `{content}\n\n`. The trailing blank line
    /// between sections is intentional — it gives the model a clean
    /// boundary without forcing callers to inject one in `content`.
    /// This exact byte sequence is also what
    /// [`token_count`](Self::token_count) measures.
    pub fn render(&self) -> String {
        let mut out = String::new();

        for section in &self.sections {
            if let Some(name) = section.name() {
                out.push_str("## ");
                out.push_str(name);
                out.push_str("\n\n");
            }
            out.push_str(&section.content);
            out.push_str("\n\n");
        }
        out
    }

    /// Approximate token count of the rendered prompt under
    /// `tokenizer`. Counts the same string the model would receive,
    /// including `## name` headers and trailing blank lines emitted by
    /// [`Self::render`].
    pub fn token_count(&self, tokenizer: &dyn Tokenizer) -> usize {
        tokenizer.count_text(&self.render())
    }
}

impl Display for Prompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render())
    }
}

impl From<&str> for Prompt {
    fn from(value: &str) -> Self {
        Prompt {
            sections: vec![PromptSection::new(value)],
        }
    }
}

/// Fluent builder for [`Prompt`]. Each [`section`](Self::section)
/// call appends to the running list; [`build`](Self::build) finalizes
/// it.
pub struct PromptBuilder {
    sections: Vec<PromptSection>,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptBuilder {
    /// Build an empty builder. Equivalent to [`Self::default`].
    pub fn new() -> Self {
        PromptBuilder { sections: vec![] }
    }

    /// Append `section`. The next [`section`](Self::section) call (or
    /// the rendered output) appears after it.
    pub fn section(mut self, section: PromptSection) -> Self {
        self.sections.push(section);
        self
    }

    /// Finalize the builder and return the [`Prompt`].
    pub fn build(self) -> Prompt {
        Prompt {
            sections: self.sections,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn temp_with_content(content: &str) -> NamedTempFile {
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{content}").unwrap();
        tmp
    }

    #[test]
    fn section_from_md_file() {
        let test_file = temp_with_content("Test 1");

        let section =
            PromptSection::from_file(test_file.path()).expect("Error reading test 1 file.");

        assert_eq!(section.content, "Test 1");
    }

    #[test]
    fn render_unnamed_section_has_no_header() {
        let prompt = Prompt::builder()
            .section(PromptSection::new("hello"))
            .build();

        assert_eq!(prompt.render(), "hello\n\n");
    }

    #[test]
    fn render_named_section_emits_h2_header() {
        let prompt = Prompt::builder()
            .section(PromptSection::with_name("Tone", "Be concise."))
            .build();

        assert_eq!(prompt.render(), "## Tone\n\nBe concise.\n\n");
    }

    #[test]
    fn render_mixes_named_and_unnamed_sections() {
        let prompt = Prompt::builder()
            .section(PromptSection::new("preamble"))
            .section(PromptSection::with_name("Tone", "Be concise."))
            .build();

        assert_eq!(prompt.render(), "preamble\n\n## Tone\n\nBe concise.\n\n");
    }

    /// `Prompt::token_count` must measure exactly what the model sees:
    /// the rendered string including `## name` headers and the trailing
    /// blank lines `render()` emits between sections.
    #[test]
    fn prompt_token_count_matches_rendered_text_under_word_tokenizer() {
        struct WordTokenizer;
        impl ailoop_core::Tokenizer for WordTokenizer {
            fn count_text(&self, text: &str) -> usize {
                text.split_whitespace().count()
            }
        }

        let prompt = Prompt::builder()
            .section(PromptSection::with_name("Tone", "Be concise."))
            .section(PromptSection::new("preamble line"))
            .build();

        // Rendered: "## Tone\n\nBe concise.\n\npreamble line\n\n"
        // Words: ##, Tone, Be, concise., preamble, line = 6
        assert_eq!(prompt.token_count(&WordTokenizer), 6);
        // PromptSection::token_count must NOT include the header.
        let sec = PromptSection::with_name("Header", "body of section");
        assert_eq!(sec.token_count(&WordTokenizer), 3);
    }

    #[test]
    fn display_matches_render() {
        let prompt = Prompt::builder()
            .section(PromptSection::with_name("Tone", "Be concise."))
            .section(PromptSection::new("trailing"))
            .build();

        assert_eq!(format!("{}", prompt), prompt.render());
    }
}
