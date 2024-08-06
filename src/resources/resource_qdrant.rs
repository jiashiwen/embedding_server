use crate::configure::get_config;
use anyhow::Result;
use once_cell::sync::Lazy;
use qdrant_client::{
    qdrant::{SearchPointsBuilder, SearchResponse},
    Qdrant,
};
use std::{sync::Arc, time::Duration};

pub static GLOBAL_QDRANT: Lazy<Arc<Qdrant>> = Lazy::new(|| {
    let config = get_config().unwrap().qdrant;
    let mut q_config = Qdrant::from_url(&config.uri)
        .timeout(Duration::from_secs(config.timeout))
        .connect_timeout(Duration::from_secs(config.connect_timeout))
        .api_key(config.api_key);
    if config.keep_alive_while_idle {
        q_config = q_config.keep_alive_while_idle();
    }
    let client = match q_config.build() {
        Ok(q) => q,
        Err(err) => panic!("{}", err),
    };
    Arc::new(client)
});

pub async fn health_check() -> Result<()> {
    let _ = GLOBAL_QDRANT.health_check().await?;
    Ok(())
}

pub async fn search_points(
    collection_name: impl Into<String>,
    vector: impl Into<Vec<f32>>,
    limit: u64,
) -> Result<SearchResponse> {
    let search_result = GLOBAL_QDRANT
        .search_points(SearchPointsBuilder::new(collection_name, vector, limit).with_payload(true))
        .await?;
    Ok(search_result)
}
