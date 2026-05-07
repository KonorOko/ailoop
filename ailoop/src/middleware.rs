use std::collections::HashMap;

use ailoop_core::{ChatMiddleware, ChatRequest};

use crate::{Prompt, PromptSection};

pub struct SystemPromptMiddleware {
    pub base: Prompt,
    pub tools_sections: HashMap<String, PromptSection>,
}

#[async_trait::async_trait]
impl ChatMiddleware for SystemPromptMiddleware {
    async fn on_chat_request(&self, req: &mut ChatRequest) {
        let mut prompt = self.base.clone();

        if let Some(tools) = &req.tools {
            for tool in tools {
                if let Some(tool_prompt) = self.tools_sections.get(&tool.name) {
                    prompt.add_section(tool_prompt.clone());
                }
            }
        }

        req.system_prompt = Some(prompt.render());
    }
}
