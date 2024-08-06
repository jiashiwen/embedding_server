use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct ConfigRocksDB {
    #[serde(default = "ConfigRocksDB::path_default")]
    pub path: String,
}

impl ConfigRocksDB {
    pub fn path_default() -> String {
        "rocksdb".to_string()
    }
}

impl Default for ConfigRocksDB {
    fn default() -> Self {
        Self {
            path: ConfigRocksDB::path_default(),
        }
    }
}
