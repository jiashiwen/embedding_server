use anyhow::{Context, Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::{LayerNorm, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use once_cell::sync::Lazy;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::{
    runtime::{Builder, Runtime},
    sync::{OnceCell, RwLock},
};

use crate::configure::{get_config, ModelConfig};

pub static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    let runtime = match init_runtime() {
        Ok(db) => db,
        Err(err) => panic!("{}", err),
    };
    Arc::new(runtime)
});

// pub static GLOBAL_MODEL: Lazy<Arc<RwLock<(BertModel, Tokenizer)>>> = Lazy::new(|| {
//     GLOBAL_RUNTIME.block_on(async {
//         let config = get_config().unwrap();
//         let m_t = match build_model_and_tokenizer(&config.model).await {
//             Ok((m, t)) => RwLock::new((m, t)),
//             Err(err) => panic!("{}", err),
//         };
//         Arc::new(m_t)
//     })
// });

pub static GLOBAL_MODEL: OnceCell<Arc<RwLock<(BertModel, Tokenizer)>>> = OnceCell::const_new();

fn init_runtime() -> Result<Runtime> {
    let rt = Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .max_io_events_per_tick(32)
        .build()?;
    Ok(rt)
}

pub async fn init_model_and_tokenizer() -> Arc<RwLock<(BertModel, Tokenizer)>> {
    let config = get_config().unwrap();
    let (m, t) = build_model_and_tokenizer(&config.model).await.unwrap();
    Arc::new(RwLock::new((m, t)))
}

async fn build_model_and_tokenizer(model_config: &ModelConfig) -> Result<(BertModel, Tokenizer)> {
    // let device = candle_examples::device(self.cpu)?;
    let device = Device::new_cuda(0)?;
    let repo = Repo::with_revision(
        model_config.model_id.clone(),
        RepoType::Model,
        model_config.revision.clone(),
    );
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json").await?;
        let tokenizer = api.get("tokenizer.json").await?;
        let weights = if model_config.use_pth {
            api.get("pytorch_model.bin").await?
        } else {
            api.get("model.safetensors").await?
        };
        (config, tokenizer, weights)
    };
    let config = std::fs::read_to_string(config_filename)?;
    let mut config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = if model_config.use_pth {
        VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
    };
    if model_config.approximate_gelu {
        config.hidden_act = HiddenAct::GeluApproximate;
    }
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

pub async fn model_device_is_cuda() -> bool {
    let m_t = GLOBAL_MODEL.get().unwrap().read().await;
    m_t.0.device.is_cuda()
}

pub async fn get_token(content: &str) -> Result<Vec<Vec<f32>>> {
    let mut m_t = GLOBAL_MODEL.get().unwrap().write().await;
    let tokenizer = m_t
        .1
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(content, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], &m_t.0.device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    let sequence_output = m_t.0.forward(&token_ids, &token_type_ids)?;
    let (_n_sentence, n_tokens, _hidden_size) = sequence_output.dims3()?;
    let embeddings = (sequence_output.sum(1).unwrap() / (n_tokens as f64)).unwrap();
    let embeddings = normalize_l2(&embeddings).unwrap();
    let encodings = embeddings.to_vec2::<f32>().unwrap();

    // let t_vec = ys.to_vec3::<f32>()?.to_vec();
    Ok(encodings)
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
