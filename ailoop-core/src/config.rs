use std::sync::Arc;

use crate::ids::RunId;
use crate::middleware::ChatMiddleware;

pub struct RunConfig {
    pub system_prompt: Option<String>,
    pub max_iterations: usize,
    pub max_tokens: u32,
    pub middlewares: Vec<Arc<dyn ChatMiddleware>>,
    /// Caller-supplied id for the run. When `None`, the engine mints a
    /// fresh UUID v4. Set this when an outer system needs to correlate
    /// the run with its own trace id.
    pub run_id: Option<RunId>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_iterations: 10,
            max_tokens: 4096,
            middlewares: vec![],
            run_id: None,
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
