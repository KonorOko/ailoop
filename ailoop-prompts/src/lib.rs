mod errors;
mod prompt;

pub use errors::PromptError;
pub use prompt::{Prompt, PromptSection};

#[cfg(test)]
mod tests {
    use super::*;
}
