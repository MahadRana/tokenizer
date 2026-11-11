//! Incremental decoding for Qwen2.5-0.5B-Instruct.
//!
//! [`QwenModel`] borrows its weights. A [`KvCache`] is passed per call, so
//! the model stays shareable. [`greedy_tokens`] still recomputes the whole
//! sequence; that is the no-cache loop the decode benchmark compares against.
//! [`generate`] is the cached loop.

mod generate;
mod model;
mod paged;
mod sample;
mod stream;

pub use generate::{
    Control, GenerateParams, Generated, SampleCursor, SampleNext, Sampled, StopReason, generate,
    generate_with,
};
pub use model::{Config, KvCache, QwenModel, tiny_model};
pub use paged::{
    AllocError, BLOCK_SIZE, BlockAllocator, BlockId, PagedCache, PagedKvPool, Release,
    blocks_for_len, prefix_block_hash,
};
pub use sample::{SampleError, greedy_token, sample_token};
pub use stream::StreamDecoder;

use anyhow::Context;
use std::path::{Path, PathBuf};

/// Files hf-hub cached for `Qwen/Qwen2.5-0.5B-Instruct`.
pub struct QwenFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
    pub generation_config: PathBuf,
}

/// Download `config.json`, `tokenizer.json`, `generation_config.json`, and
/// `model.safetensors`.
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
    let generation_config = repo
        .download_file()
        .filename("generation_config.json")
        .send()
        .context("generation_config.json")?;
    Ok(QwenFiles {
        config,
        tokenizer,
        weights,
        generation_config,
    })
}

/// Load the model from paths already on disk. `weights` is a safetensors file
/// mapped as f32 on CPU.
pub fn load_model(config: &Path, weights: &Path) -> anyhow::Result<QwenModel> {
    QwenModel::load(config, weights)
}

/// `eos_token_id` from a Hugging Face `generation_config.json`.
///
/// The field is either one integer or a list. Qwen2.5-0.5B-Instruct uses
/// `[151645, 151643]`.
pub fn parse_eos_token_ids(text: &str) -> anyhow::Result<Vec<u32>> {
    let value: serde_json::Value = serde_json::from_str(text)?;
    let eos = value
        .get("eos_token_id")
        .context("generation config has no eos_token_id")?;
    if let Some(id) = eos.as_u64() {
        let id = u32::try_from(id).context("eos id does not fit in u32")?;
        return Ok(vec![id]);
    }
    let list = eos
        .as_array()
        .context("eos_token_id is not an integer or a list")?;
    if list.is_empty() {
        anyhow::bail!("eos_token_id is empty");
    }
    let mut ids = Vec::with_capacity(list.len());
    for item in list {
        let id = item
            .as_u64()
            .context("eos_token_id entry is not an integer")?;
        ids.push(u32::try_from(id).context("eos id does not fit in u32")?);
    }
    Ok(ids)
}

/// Read [`parse_eos_token_ids`] from a file.
pub fn load_eos_token_ids(path: &Path) -> anyhow::Result<Vec<u32>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_eos_token_ids(&text)
}

/// Argmax one step and append it, `new_tokens` times. No KV cache.
///
/// Ties go to the lower id (`>` rather than `>=`). Each step runs the full
/// sequence. This is the B1 loop and the baseline the cached decoder is
/// measured against.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_eos_list_and_a_single_id() {
        let ids =
            parse_eos_token_ids(r#"{"eos_token_id":[151645,151643],"do_sample":true}"#).unwrap();
        assert_eq!(ids, vec![151645, 151643]);
        let ids = parse_eos_token_ids(r#"{"bos_token_id":1,"eos_token_id":151643}"#).unwrap();
        assert_eq!(ids, vec![151643]);
        assert!(parse_eos_token_ids(r#"{"bos_token_id":1}"#).is_err());
        assert!(parse_eos_token_ids(r#"{"eos_token_id":[]}"#).is_err());
    }
}
