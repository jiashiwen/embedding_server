use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct ConfigModel {
    #[serde(default = "ConfigModel::model_id_default")]
    pub model_id: String,
    #[serde(default = "ConfigModel::revision_default")]
    pub revision: String,
    #[serde(default = "ConfigModel::use_pth_default")]
    pub use_pth: bool,
    #[serde(default = "ConfigModel::approximate_gelu_default")]
    pub approximate_gelu: bool,
}

impl Default for ConfigModel {
    fn default() -> Self {
        Self {
            model_id: Self::model_id_default(),
            revision: Self::revision_default(),
            use_pth: Self::use_pth_default(),
            approximate_gelu: Self::approximate_gelu_default(),
        }
    }
}

impl ConfigModel {
    fn model_id_default() -> String {
        "moka-ai/m3e-large".to_string()
    }
    fn revision_default() -> String {
        "main".to_string()
    }
    fn use_pth_default() -> bool {
        true
    }
    fn approximate_gelu_default() -> bool {
        false
    }
}
