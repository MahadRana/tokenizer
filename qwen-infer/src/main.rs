//! `cargo run --release -p qwen-infer -- "The capital of France is"`
//!
//! Encodes with [`tokenizer::RankTokenizer::load_qwen`], prefill once, then
//! decodes with a KV cache. The default is 20 greedy tokens (temperature 0),
//! stopping early on the EOS ids in `generation_config.json`. The prompt is
//! not a chat template. The root `tokenizer` binary is still the trainer.

use anyhow::Context;
use qwen_infer::{GenerateParams, StopReason};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args = Args::parse()?;
    let files = qwen_infer::download_qwen()?;
    let tokenizer_path = files
        .tokenizer
        .to_str()
        .context("tokenizer path is not utf-8")?;
    let tokenizer = tokenizer::RankTokenizer::load_qwen(tokenizer_path)?;
    let prompt_ids = tokenizer.encode(&args.prompt)?;
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt encoded to no tokens");
    }
    eprintln!("prompt ids ({}): {prompt_ids:?}", prompt_ids.len());

    let mut params = GenerateParams::greedy(args.max_tokens);
    params.temperature = args.temperature;
    params.top_p = args.top_p;
    params.seed = args.seed;
    params.eos = qwen_infer::load_eos_token_ids(&files.generation_config)?;
    params.stop = args.stop;
    eprintln!(
        "max_tokens {} temperature {} top_p {} seed {} eos {:?}",
        params.max_tokens, params.temperature, params.top_p, params.seed, params.eos
    );

    let model = qwen_infer::load_model(&files.config, &files.weights)?;
    let started = Instant::now();
    let generated = qwen_infer::generate(&model, &prompt_ids, &params, |id| {
        tokenizer.decode_bytes(&[id])
    })?;
    let elapsed = started.elapsed();
    println!("{}", generated.text);
    eprintln!("generated ids: {:?}", generated.tokens);
    eprintln!("stop: {:?}", generated.reason);
    let prefill = generated.prefill.as_secs_f64();
    let decode = generated.decode.as_secs_f64();
    let n = generated.tokens.len();
    if n == 0 || decode == 0.0 {
        eprintln!(
            "prefill {prefill:.4} s, decode {decode:.4} s, {n} tokens (KV cache, total {:.4} s)",
            elapsed.as_secs_f64()
        );
    } else {
        eprintln!(
            "prefill {prefill:.4} s, decode {decode:.4} s, {:.4} tok/s ({n} tokens, KV cache, total {:.4} s)",
            n as f64 / decode,
            elapsed.as_secs_f64()
        );
    }
    if generated.reason == StopReason::Eos {
        eprintln!("stopped on eos before max_tokens");
    }
    Ok(())
}

struct Args {
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
    stop: Vec<String>,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut prompt = None;
        let mut max_tokens = 20usize;
        let mut temperature = 0.0f32;
        let mut top_p = 1.0f32;
        let mut seed = 0u64;
        let mut stop = Vec::new();
        let mut raw = std::env::args().skip(1);
        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    usage();
                    std::process::exit(0);
                }
                "--max-tokens" => {
                    let value = next_value(&mut raw, &arg)?;
                    max_tokens = value
                        .parse()
                        .with_context(|| format!("--max-tokens {value}"))?;
                }
                "--temperature" => {
                    let value = next_value(&mut raw, &arg)?;
                    temperature = value
                        .parse()
                        .with_context(|| format!("--temperature {value}"))?;
                }
                "--top-p" => {
                    top_p = next_value(&mut raw, &arg)?
                        .parse()
                        .context("--top-p wants a number")?;
                }
                "--seed" => {
                    seed = next_value(&mut raw, &arg)?
                        .parse()
                        .context("--seed wants an integer")?;
                }
                "--stop" => stop.push(next_value(&mut raw, &arg)?),
                other if other.starts_with('-') => {
                    anyhow::bail!("unknown argument {other}");
                }
                other => {
                    if prompt.is_some() {
                        anyhow::bail!("only one prompt is accepted (got {other:?} as well)");
                    }
                    prompt = Some(other.to_string());
                }
            }
        }
        let Some(prompt) = prompt else {
            usage();
            std::process::exit(1);
        };
        Ok(Self {
            prompt,
            max_tokens,
            temperature,
            top_p,
            seed,
            stop,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    args.next().with_context(|| format!("{flag} needs a value"))
}

fn usage() {
    eprintln!(
        "usage: qwen-infer [--max-tokens N] [--temperature T] [--top-p P] [--seed N] [--stop STRING]... <prompt>"
    );
}
