use std::{fmt::Display, path::Path};

use crate::errors::PromptError;

#[derive(Debug, Clone)]
pub struct Prompt {
    sections: Vec<PromptSection>,
}

#[derive(Debug, Clone)]
pub struct PromptSection {
    name: Option<String>,
    content: String,
}

impl PromptSection {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            name: None,
            content: content.into(),
        }
    }

    pub fn with_name(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            content: content.into(),
        }
    }
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PromptError> {
        let content = std::fs::read_to_string(&path).map_err(|source| PromptError::LoadFile {
            path: path.as_ref().to_path_buf(),
            source,
        })?;

        Ok(Self::new(content))
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

impl Prompt {
    pub fn builder() -> PromptBuilder {
        PromptBuilder::new()
    }

    pub fn new() -> Self {
        Prompt { sections: vec![] }
    }

    pub fn sections(&self) -> &Vec<PromptSection> {
        &self.sections
    }

    pub fn add_section(&mut self, section: PromptSection) {
        self.sections.push(section);
    }

    pub fn remove_section(&mut self, name: &str) {
        if let Some(idx) = self
            .sections
            .iter()
            .position(|section| section.name() == Some(name))
        {
            self.sections.remove(idx);
        };
    }

    pub fn render(&self) -> String {
        let mut out = String::new();

        for section in &self.sections {
            out.push_str(&section.content);
            out.push_str("\n\n");
        }
        out
    }
}

impl Display for Prompt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut prompt = String::new();

        for section in self.sections.iter() {
            prompt.push_str(&section.content);
            prompt.push_str("\n\n");
        }

        write!(f, "{}", prompt)
    }
}

impl From<&str> for Prompt {
    fn from(value: &str) -> Self {
        Prompt {
            sections: vec![PromptSection::new(value)],
        }
    }
}

pub struct PromptBuilder {
    sections: Vec<PromptSection>,
}

impl PromptBuilder {
    pub fn new() -> Self {
        PromptBuilder { sections: vec![] }
    }

    pub fn section(mut self, section: PromptSection) -> Self {
        self.sections.push(section);
        self
    }

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
}
