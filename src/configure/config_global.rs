use super::config_qdrant::ConfigQdrant;
use super::{config_http::ConfigHttp, config_model::ConfigModel};
use crate::configure::config_error::{ConfigError, ConfigErrorType};
use anyhow::Result;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_yaml::from_str;
use std::fs;
use std::path::Path;
use std::sync::RwLock;

pub static GLOBAL_CONFIG: Lazy<RwLock<Config>> = Lazy::new(|| {
    let config = RwLock::new(Config::default());
    config
});

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "Config::http_default")]
    pub http: ConfigHttp,
    #[serde(default = "ConfigModel::default")]
    pub model: ConfigModel,
    #[serde(default = "ConfigQdrant::default")]
    pub qdrant: ConfigQdrant,
}

impl Config {
    pub fn default() -> Self {
        Self {
            http: ConfigHttp::default(),
            model: ConfigModel::default(),
            qdrant: ConfigQdrant::default(),
        }
    }

    pub fn http_default() -> ConfigHttp {
        ConfigHttp::default()
    }

    pub fn get_config_image(&self) -> Self {
        self.clone()
    }
}

pub fn generate_default_config(path: &str) -> Result<()> {
    let config = Config::default();
    let yml = serde_yaml::to_string(&config)?;
    fs::write(path, yml)?;
    Ok(())
}

pub fn set_config(path: &str) {
    let mut global_config = GLOBAL_CONFIG.write().unwrap();
    if path.is_empty() {
        if Path::new("config.yml").exists() {
            let contents =
                fs::read_to_string("config.yml").expect("Read config file config.yml error!");
            let config = from_str::<Config>(contents.as_str()).expect("Parse config.yml error!");
            *global_config = config;
        }
        return;
    }

    let err_str = format!("Read config file {} error!", path);
    let contents = fs::read_to_string(path).expect(err_str.as_str());
    let config = from_str::<Config>(contents.as_str()).expect("Parse config.yml error!");
    *global_config = config;
}

pub fn get_config() -> Result<Config> {
    let locked_config = GLOBAL_CONFIG.read().map_err(|e| {
        return ConfigError::from_err(e.to_string(), ConfigErrorType::UnknowErr);
    })?;
    Ok(locked_config.get_config_image())
}

pub fn get_current_config_yml() -> Result<String> {
    let c = get_config()?;
    let yml = serde_yaml::to_string(&c)?;
    Ok(yml)
}
