use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompactionError {
    #[error("Not enough history to compact")]
    NotEnoughHistory,

    #[error("Budget exceeded")]
    BudgetExceeded,
}
