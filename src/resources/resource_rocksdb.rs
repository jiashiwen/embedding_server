use crate::configure::get_config;
use anyhow::anyhow;
use anyhow::Result;
use once_cell::sync::Lazy;
use rocksdb::{DBWithThreadMode, MultiThreaded, Options};
use std::sync::Arc;

pub const CF_TASK_CHECKPOINTS: &'static str = "cf_task_checkpoints";
pub const CF_TASK: &'static str = "cf_task";
pub const CF_TASK_STATUS: &'static str = "cf_task_status";

pub static GLOBAL_ROCKSDB: Lazy<Arc<DBWithThreadMode<MultiThreaded>>> = Lazy::new(|| {
    let config = get_config().unwrap();
    // let rocksdb = match init_rocksdb("oss_pipe_rocksdb") {
    let rocksdb = match init_rocksdb(&config.rocksdb.path) {
        Ok(db) => db,
        Err(err) => panic!("{}", err),
    };
    Arc::new(rocksdb)
});

pub fn init_rocksdb(db_path: &str) -> Result<DBWithThreadMode<MultiThreaded>> {
    let mut cf_opts = Options::default();
    cf_opts.set_allow_concurrent_memtable_write(true);
    cf_opts.set_max_write_buffer_number(16);
    cf_opts.set_write_buffer_size(128 * 1024 * 1024);
    cf_opts.set_disable_auto_compactions(true);

    let mut db_opts = Options::default();
    db_opts.create_missing_column_families(true);
    db_opts.create_if_missing(true);

    let db = DBWithThreadMode::<MultiThreaded>::open_cf_with_opts(
        &db_opts,
        db_path,
        vec![
            (CF_TASK_CHECKPOINTS, cf_opts.clone()),
            (CF_TASK, cf_opts.clone()),
            (CF_TASK_STATUS, cf_opts.clone()),
        ],
    )?;
    Ok(db)
}

pub fn remove_checkpoint_from_cf(task_id: &str) -> Result<()> {
    let cf = match GLOBAL_ROCKSDB.cf_handle(CF_TASK_CHECKPOINTS) {
        Some(cf) => cf,
        None => return Err(anyhow!("column family not exist")),
    };
    GLOBAL_ROCKSDB.delete_cf(&cf, task_id.as_bytes())?;
    Ok(())
}

pub fn remove_task_from_cf(task_id: &str) -> Result<()> {
    let cf = match GLOBAL_ROCKSDB.cf_handle(CF_TASK) {
        Some(cf) => cf,
        None => return Err(anyhow!("column family not exist")),
    };
    GLOBAL_ROCKSDB.delete_cf(&cf, task_id.as_bytes())?;
    Ok(())
}

pub fn remove_task_status_from_cf(task_id: &str) -> Result<()> {
    let cf = match GLOBAL_ROCKSDB.cf_handle(CF_TASK_STATUS) {
        Some(cf) => cf,
        None => return Err(anyhow!("column family not exist")),
    };
    GLOBAL_ROCKSDB.delete_cf(&cf, task_id.as_bytes())?;
    Ok(())
}
