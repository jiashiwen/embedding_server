use anyhow::{anyhow, Context, Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase};
use candle_transformers::models::qwen2_moe::{Config as ConfigMoe, Model as ModelMoe};
use hf_hub::api::tokio::ApiBuilder;
use hf_hub::{Repo, RepoType};
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use tokio::sync::OnceCell;

use super::token_output_stream::TokenOutputStream;

pub const MODEL_ID: &'static str = "Qwen/Qwen2-7B";
// pub static GLOBAL_INFERENCE_MODEL: OnceCell<Arc<RwLock<ModelBase>>> = OnceCell::const_new();
// pub static GLOBAL_INFERENCE_MODEL: OnceCell<RwLock<ModelBase>> = OnceCell::const_new();
pub static GLOBAL_PIPELINE: OnceCell<Arc<RwLock<TextGeneration>>> = OnceCell::const_new();

// pub async fn init_inference_model() -> RwLock<ModelBase> {
//     let model = build_inference_model().await.unwrap();
//     RwLock::new(model)
// }

// pub async fn build_inference_model() -> Result<ModelBase> {
//     let device = Device::new_cuda(0)?;
//     let api = ApiBuilder::new().build()?;
//     let repo = api.repo(Repo::with_revision(
//         MODEL_ID.to_string(),
//         RepoType::Model,
//         "main".to_string(),
//     ));

//     let filenames = vec![repo.get("model.safetensors").await?];
//     let dtype = DType::BF16;
//     let config_file = repo.get("config.json").await?;

//     let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
//     let config: ConfigBase = serde_json::from_slice(&std::fs::read(config_file)?)?;
//     // let model = Model::Base(ModelBase::new(&config, vb)?);
//     let model = ModelBase::new(&config, vb)?;
//     Ok(model)
// }

pub async fn init_global_pipeline() -> Arc<RwLock<TextGeneration>> {
    let pipeline = build_pipeline().await.unwrap();
    Arc::new(RwLock::new(pipeline))
}

pub async fn build_pipeline() -> Result<TextGeneration> {
    let device = Device::new_cuda(0)?;
    let api = ApiBuilder::new().build()?;
    let repo = api.repo(Repo::with_revision(
        MODEL_ID.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let tokenizer_filename = repo.get("tokenizer.json").await?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    // let filenames = vec![repo.get("model.safetensors").await?];
    let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json").await?;
    let dtype = DType::BF16;
    let config_file = repo.get("config.json").await?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let config: ConfigBase = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let model = Model::Base(ModelBase::new(&config, vb)?);

    let pipeline = TextGeneration::new(model, tokenizer, 299792458, None, None, 1.1, 64, &device);

    Ok(pipeline)
}

pub enum Model {
    Base(ModelBase),
    Moe(ModelMoe),
}

impl Model {
    fn forward(&mut self, xs: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Moe(ref mut m) => m.forward(xs, s),
            Self::Base(ref mut m) => m.forward(xs, s),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            Model::Base(ref mut m) => m.clear_kv_cache(),
            Model::Moe(ref mut m) => m.clear_kv_cache(),
        }
    }
}
pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}
impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        let mut answer = "".to_string();
        self.tokenizer.clear();

        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?
                .unsqueeze(0)
                .context(format!("{}:{}", file!(), line!()))?;

            let logits = self.model.forward(&input, start_pos).context(format!(
                "{}:{}",
                file!(),
                line!()
            ))?;
            let logits = logits
                .squeeze(0)
                .context(format!("{}:{}", file!(), line!()))?
                .squeeze(0)
                .context(format!("{}:{}", file!(), line!()))?
                .to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits).context(format!(
                "{}:{}",
                file!(),
                line!()
            ))?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) =
                self.tokenizer
                    .next_token(next_token)
                    .context(format!("{}:{}", file!(), line!()))?
            {
                answer.push_str(t.as_str());
            }
        }

        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            answer.push_str(rest.as_str());
        }
        self.model.clear_kv_cache();

        Ok(answer)
    }
}

pub fn answer(question: &str, max_len: usize) -> Result<String> {
    match GLOBAL_PIPELINE
        .get()
        .unwrap()
        .write()
        .unwrap()
        .run(question, max_len)
    {
        Ok(s) => Ok(s),
        Err(e) => {
            log::error!("{:?}", e);
            return Err(e);
        }
    }
}

pub async fn hub_load_safetensors(
    repo: &hf_hub::api::tokio::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo
        .get(json_file)
        .await
        .map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        // None => candle_core::bail!("no weight map in {json_file:?}"),
        None => return Err(anyhow!("no weight map in {json_file:?}")),
        Some(serde_json::Value::Object(map)) => map,
        // Some(_) => candle_core::bail!("weight map in {json_file:?} is not a map"),
        Some(_) => return Err(anyhow!("weight map in {json_file:?} is not a map")),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    // let safetensors_files = safetensors_files
    //     .iter()
    //     .map(|v| repo.get(v).await.map_err(candle_core::Error::wrap))
    //     .collect::<Result<Vec<_>>>()?;

    let mut vec_paths = vec![];
    for f in safetensors_files {
        let p = repo.get(f.as_str()).await?;
        vec_paths.push(p);
    }
    Ok(vec_paths)
}
