use std::sync::Arc;

use crate::middleware::ChatMiddleware;

pub struct RunConfig {
    pub system_prompt: Option<String>,
    pub max_iterations: usize,
    pub max_tokens: u32,
    pub middlewares: Vec<Arc<dyn ChatMiddleware>>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_iterations: 10,
            max_tokens: 4096,
            middlewares: vec![],
        }
    }
}

impl RunConfig {
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..Default::default()
        }
    }
}
