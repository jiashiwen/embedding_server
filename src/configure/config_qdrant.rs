use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct ConfigQdrant {
    #[serde(default = "ConfigQdrant::default_uri")]
    pub uri: String,
    #[serde(default = "ConfigQdrant::default_timeout")]
    pub timeout: u64,
    #[serde(default = "ConfigQdrant::default_connect_timeout")]
    pub connect_timeout: u64,
    #[serde(default = "ConfigQdrant::default_keep_alive_while_idle")]
    pub keep_alive_while_idle: bool,
    #[serde(default = "ConfigQdrant::default_api_key")]
    pub api_key: Option<String>,
    #[serde(default = "ConfigQdrant::default_collection")]
    pub collection: String,
}

impl ConfigQdrant {
    pub fn default_uri() -> String {
        String::from("http://localhost:6334")
    }

    pub fn default_timeout() -> u64 {
        5
    }
    pub fn default_connect_timeout() -> u64 {
        5
    }
    pub fn default_keep_alive_while_idle() -> bool {
        true
    }
    pub fn default_api_key() -> Option<String> {
        None
    }
    pub fn default_collection() -> String {
        String::from("default_collection")
    }
}

impl Default for ConfigQdrant {
    fn default() -> Self {
        Self {
            uri: Self::default_uri(),
            timeout: Self::default_timeout(),
            connect_timeout: Self::default_connect_timeout(),
            keep_alive_while_idle: Self::default_keep_alive_while_idle(),
            api_key: Self::default_api_key(),
            collection: Self::default_collection(),
        }
    }
}
