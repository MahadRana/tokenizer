//! One full forward of Qwen2.5-0.5B-Instruct on CPU, plus a greedy loop that
//! recomputes the whole sequence for every new token.
//!
//! There is no KV cache. That is B2. The model borrows its weights (`&self`)
//! and the caller owns the token ids.

mod model;

pub use model::{Config, QwenModel};

use anyhow::Context;
use std::path::{Path, PathBuf};

/// Files hf-hub cached for `Qwen/Qwen2.5-0.5B-Instruct`.
pub struct QwenFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

/// Download `config.json`, `tokenizer.json`, and `model.safetensors`.
///
/// hf-hub 1.0's synchronous client is [`hf_hub::HFClientSync`]. A file that
/// is already in the hub cache is not downloaded again. The weights stay in
/// that cache; nothing in this repo stores them.
pub fn download_qwen() -> anyhow::Result<QwenFiles> {
    let client = hf_hub::HFClientSync::new().context("hf-hub client")?;
    let repo = client.model("Qwen", "Qwen2.5-0.5B-Instruct");
    let config = repo
        .download_file()
        .filename("config.json")
        .send()
        .context("config.json")?;
    let tokenizer = repo
        .download_file()
        .filename("tokenizer.json")
        .send()
        .context("tokenizer.json")?;
    let weights = repo
        .download_file()
        .filename("model.safetensors")
        .send()
        .context("model.safetensors")?;
    Ok(QwenFiles {
        config,
        tokenizer,
        weights,
    })
}

/// Load the model from paths already on disk. `weights` is a safetensors file
/// mapped as f32 on CPU.
pub fn load_model(config: &Path, weights: &Path) -> anyhow::Result<QwenModel> {
    QwenModel::load(config, weights)
}

/// Argmax one step and append it, `new_tokens` times.
///
/// Ties go to the lower id (`>` rather than `>=`). Each step runs the full
/// sequence, so the cost grows with the length. Twenty tokens of a short
/// prompt is the B1 loop.
pub fn greedy_tokens(
    model: &QwenModel,
    prompt: &[u32],
    new_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut ids = prompt.to_vec();
    let mut generated = Vec::with_capacity(new_tokens);
    for _ in 0..new_tokens {
        let next = model.next_token(&ids)?;
        generated.push(next);
        ids.push(next);
    }
    Ok(generated)
}
