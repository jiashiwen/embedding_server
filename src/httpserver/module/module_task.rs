use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct TaskId {
    pub task_id: String,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct TaskIds {
    pub task_ids: Vec<String>,
}
