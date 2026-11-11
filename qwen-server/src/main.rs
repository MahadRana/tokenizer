//! `cargo run --release -p qwen-server -- --bind 127.0.0.1:8080`
//!
//! Loads Qwen2.5-0.5B-Instruct from the hf-hub cache, applies the chat
//! template, and serves `POST /v1/chat/completions` plus `GET /v1/models`.
//! The prompt is not the raw string the `qwen-infer` CLI still takes.
//! `--max-batch-size` and `--max-waiting` set the iteration scheduler.

use std::net::SocketAddr;
use std::sync::Arc;

use qwen_server::ServerConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let options = parse_args();
    eprintln!("loading Qwen2.5-0.5B-Instruct (hf-hub cache, f32 CPU)");
    let files = qwen_infer::download_qwen()?;
    let tokenizer_path = files
        .tokenizer
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("tokenizer path is not utf-8"))?;
    let tokenizer = tokenizer::RankTokenizer::load_qwen(tokenizer_path)?;
    let eos = qwen_infer::load_eos_token_ids(&files.generation_config)?;
    let model = qwen_infer::load_model(&files.config, &files.weights)?;
    eprintln!(
        "eos ids {eos:?}  max_batch_size {}  max_waiting {}  num_blocks {}  prefix_cache {}",
        options.config.max_batch_size,
        options.config.max_waiting,
        options.config.num_blocks,
        options.config.prefix_cache
    );
    let server = qwen_server::serve_with(
        model,
        Arc::new(tokenizer),
        eos,
        options.bind,
        options.config,
    )
    .await?;
    eprintln!(
        "listening on http://{}/v1  model {}",
        server.addr(),
        qwen_server::MODEL_ID
    );
    tokio::signal::ctrl_c().await?;
    eprintln!("shutting down");
    server.shutdown().await;
    Ok(())
}

struct Options {
    bind: SocketAddr,
    config: ServerConfig,
}

fn parse_args() -> Options {
    let mut args = std::env::args().skip(1);
    let mut bind = "127.0.0.1:8080".to_string();
    let mut config = ServerConfig::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                eprintln!(
                    "usage: qwen-server [--bind 127.0.0.1:8080] [--max-batch-size 8] [--max-waiting 32] [--num-blocks 256] [--no-prefix-cache]"
                );
                std::process::exit(0);
            }
            "--bind" => {
                bind = args.next().unwrap_or_else(|| {
                    eprintln!("--bind needs an address");
                    std::process::exit(1);
                });
            }
            "--max-batch-size" => {
                config.max_batch_size = parse_usize("--max-batch-size", &mut args);
            }
            "--max-waiting" => {
                config.max_waiting = parse_usize("--max-waiting", &mut args);
            }
            "--num-blocks" => {
                config.num_blocks = parse_usize("--num-blocks", &mut args);
            }
            "--no-prefix-cache" => {
                config.prefix_cache = false;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(1);
            }
        }
    }
    let bind = match bind.parse() {
        Ok(addr) => addr,
        Err(err) => {
            eprintln!("bad --bind {bind}: {err}");
            std::process::exit(1);
        }
    };
    Options { bind, config }
}

fn parse_usize(flag: &str, args: &mut impl Iterator<Item = String>) -> usize {
    let raw = args.next().unwrap_or_else(|| {
        eprintln!("{flag} needs a number");
        std::process::exit(1);
    });
    match raw.parse::<usize>() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("bad {flag} {raw}: {err}");
            std::process::exit(1);
        }
    }
}
