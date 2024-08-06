use super::embedding_setence;
use crate::{configure::get_config, resources::resource_qdrant::search_points};
use anyhow::Result;

use qdrant_client::qdrant::SearchResponse;

pub async fn retriever(content: &str, limit: u64) -> Result<SearchResponse> {
    let collection_name = get_config()?.qdrant.collection;
    let embedding = embedding_setence(content).await?;
    let vector = embedding[0].clone();
    let r = search_points(collection_name, vector, limit).await?;
    Ok(r)
}
