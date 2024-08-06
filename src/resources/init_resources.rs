use anyhow::Result;

use super::resource_qdrant::health_check;

pub async fn init_resources() -> Result<()> {
    let _ = health_check().await?;
    Ok(())
}
