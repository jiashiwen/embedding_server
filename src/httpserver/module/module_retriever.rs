use std::collections::HashMap;

use qdrant_client::qdrant::Value;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct RespRetriever {
    pub id: String,
    pub payload: HashMap<String, Value>,
    pub score: f32,
}
