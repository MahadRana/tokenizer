//! `cargo run --release -p qwen-server -- --bind 127.0.0.1:8080`
//!
//! Loads Qwen2.5-0.5B-Instruct from the hf-hub cache, applies the chat
//! template, and serves `POST /v1/chat/completions` plus `GET /v1/models`.
//! The prompt is not the raw string the `qwen-infer` CLI still takes.

use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bind = parse_bind();
    eprintln!("loading Qwen2.5-0.5B-Instruct (hf-hub cache, f32 CPU)");
    let files = qwen_infer::download_qwen()?;
    let tokenizer_path = files
        .tokenizer
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("tokenizer path is not utf-8"))?;
    let tokenizer = tokenizer::RankTokenizer::load_qwen(tokenizer_path)?;
    let eos = qwen_infer::load_eos_token_ids(&files.generation_config)?;
    let model = qwen_infer::load_model(&files.config, &files.weights)?;
    eprintln!("eos ids {eos:?}");
    let server = qwen_server::serve(model, Arc::new(tokenizer), eos, bind).await?;
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

fn parse_bind() -> SocketAddr {
    let mut args = std::env::args().skip(1);
    let mut bind = "127.0.0.1:8080".to_string();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                eprintln!("usage: qwen-server [--bind 127.0.0.1:8080]");
                std::process::exit(0);
            }
            "--bind" => {
                bind = args.next().unwrap_or_else(|| {
                    eprintln!("--bind needs an address");
                    std::process::exit(1);
                });
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(1);
            }
        }
    }
    match bind.parse() {
        Ok(addr) => addr,
        Err(err) => {
            eprintln!("bad --bind {bind}: {err}");
            std::process::exit(1);
        }
    }
}
