//! `cargo run --release -p qwen-infer -- "The capital of France is"`
//!
//! Encodes with [`tokenizer::RankTokenizer::load_qwen`], runs 20 greedy steps
//! on CPU, and decodes those 20 ids. The prompt is not a chat template.
//! The root `tokenizer` binary is still the trainer: candle stays in this
//! package so `cargo test` on the core crate does not compile it.

use anyhow::Context;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(prompt) = args.next() else {
        eprintln!("usage: qwen-infer <prompt>");
        std::process::exit(1);
    };
    if args.next().is_some() {
        eprintln!("usage: qwen-infer <prompt>");
        std::process::exit(1);
    }

    let files = qwen_infer::download_qwen()?;
    let tokenizer_path = files
        .tokenizer
        .to_str()
        .context("tokenizer path is not utf-8")?;
    let tokenizer = tokenizer::RankTokenizer::load_qwen(tokenizer_path)?;
    let prompt_ids = tokenizer.encode(&prompt)?;
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt encoded to no tokens");
    }
    eprintln!("prompt ids ({}): {prompt_ids:?}", prompt_ids.len());

    let model = qwen_infer::load_model(&files.config, &files.weights)?;
    let started = Instant::now();
    let generated = qwen_infer::greedy_tokens(&model, &prompt_ids, 20)?;
    let elapsed = started.elapsed();
    let text = tokenizer.decode(&generated)?;
    println!("{text}");
    eprintln!("generated ids: {generated:?}");
    let secs = elapsed.as_secs_f64();
    eprintln!(
        "tokens/sec: {:.4} ({secs:.4} s for 20 tokens, full forward each step, no KV cache)",
        20.0 / secs
    );
    Ok(())
}
