use ailoop_core::{AssistantBlock, Message, ToolResultContent, UserBlock};

pub trait TokenEstimator {
    fn estimate_text(&self, text: &str) -> usize;

    fn estimate_message(&self, message: &Message) -> usize {
        match message {
            Message::User { blocks } => {
                let mut total = 0;
                blocks.iter().for_each(|block| match block {
                    UserBlock::Text(text) => total += self.estimate_text(text),
                    UserBlock::ToolResult { call_id, content } => match content {
                        ToolResultContent::Error(error) => {
                            total += self.estimate_text(call_id) + self.estimate_text(error)
                        }
                        ToolResultContent::Text(text) => {
                            total += self.estimate_text(call_id) + self.estimate_text(text)
                        }
                    },
                });

                total
            }

            Message::Assistant { blocks } => {
                let mut total = 0;
                blocks.iter().for_each(|block| match block {
                    AssistantBlock::Text(text) => total += self.estimate_text(text),
                    AssistantBlock::ToolCall { id, name, args } => {
                        total += self.estimate_text(id)
                            + self.estimate_text(name)
                            + self.estimate_text(&args.to_string())
                    }
                });

                total
            }
        }
    }

    fn estimate_context(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| self.estimate_message(m))
            .sum::<usize>()
    }
}

pub struct CharEstimator;

impl TokenEstimator for CharEstimator {
    fn estimate_text(&self, text: &str) -> usize {
        text.len() / 4
    }
}
