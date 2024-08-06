use serde::Serialize;

#[derive(Debug, PartialEq, Serialize, serde::Deserialize, Clone)]
pub struct ConfigHttp {
    #[serde(default = "ConfigHttp::port_default")]
    pub port: u16,
    #[serde(default = "ConfigHttp::bind_default")]
    pub bind: String,
}

impl Default for ConfigHttp {
    fn default() -> Self {
        Self {
            port: ConfigHttp::port_default(),
            bind: ConfigHttp::bind_default(),
        }
    }
}

impl ConfigHttp {
    pub fn port_default() -> u16 {
        3000
    }
    pub fn bind_default() -> String {
        "::0".to_string()
    }
}
