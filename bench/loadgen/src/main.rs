//! `cargo run --release -p loadgen -- --url http://127.0.0.1:8080/v1/chat/completions`
//!
//! Open-loop Poisson client. One warmup request is not included in the
//! histograms. The printed line is one sample, not a percentile study.

use loadgen::{LoadConfig, run_load};
use std::env;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let mut url = "http://127.0.0.1:8080/v1/chat/completions".to_string();
    let mut requests = 8usize;
    let mut rate = 1.0f64;
    let mut seed = 1u64;
    let mut prompt = "Hi".to_string();
    let mut max_tokens = 4u32;
    let mut model = "Qwen/Qwen2.5-0.5B-Instruct".to_string();
    let mut warmup = 1usize;
    let mut mix = false;
    let mut prefix = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                eprintln!(
                    "usage: loadgen [--url URL] [--requests 8] [--rate 1] [--seed 1] [--prompt Hi] [--max-tokens 4] [--warmup 1] [--model ID] [--workload mix|prefix]"
                );
                return ExitCode::SUCCESS;
            }
            "--url" => url = need(&mut args, "--url"),
            "--requests" => requests = parse_num(&mut args, "--requests"),
            "--rate" => rate = parse_f64(&mut args, "--rate"),
            "--seed" => seed = parse_num(&mut args, "--seed"),
            "--prompt" => prompt = need(&mut args, "--prompt"),
            "--max-tokens" => max_tokens = parse_num(&mut args, "--max-tokens"),
            "--model" => model = need(&mut args, "--model"),
            "--warmup" => warmup = parse_num(&mut args, "--warmup"),
            "--workload" => {
                let value = need(&mut args, "--workload");
                if value != "mix" && value != "prefix" {
                    eprintln!("--workload must be mix or prefix");
                    return ExitCode::from(2);
                }
                mix = value == "mix";
                prefix = value == "prefix";
            }
            other => {
                eprintln!("unknown argument {other}");
                return ExitCode::from(2);
            }
        }
    }
    let workload = if prefix {
        loadgen::Workload::Prefix
    } else if mix {
        loadgen::Workload::Mix
    } else {
        loadgen::Workload::Fixed { prompt, max_tokens }
    };
    let config = LoadConfig {
        url,
        requests,
        rate_per_sec: rate,
        seed,
        workload,
        model,
        warmup,
    };
    match run_load(config).await {
        Ok(report) => {
            println!(
                "ok {} err {} wall_s {:.4} output_tokens {} tok_s {:.4} req_s {:.4} ttft_p50_us {} ttft_p95_us {} ttft_p99_us {} itl_p50_us {} itl_p99_us {} e2e_p99_us {}",
                report.ok,
                report.errors,
                report.wall_secs,
                report.output_tokens,
                report.output_tokens_per_sec,
                report.requests_per_sec,
                fmt_opt(report.ttft_p50_us),
                fmt_opt(report.ttft_p95_us),
                fmt_opt(report.ttft_p99_us),
                fmt_opt(report.itl_p50_us),
                fmt_opt(report.itl_p99_us),
                fmt_opt(report.e2e_p99_us),
            );
            if report.errors > 0 || report.ok == 0 {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(err) => {
            eprintln!("loadgen: {err}");
            ExitCode::from(1)
        }
    }
}

fn need(args: &mut impl Iterator<Item = String>, flag: &str) -> String {
    args.next().unwrap_or_else(|| {
        eprintln!("{flag} needs a value");
        std::process::exit(2);
    })
}

fn parse_num<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, flag: &str) -> T {
    let raw = need(args, flag);
    raw.parse().unwrap_or_else(|_| {
        eprintln!("bad {flag} {raw}");
        std::process::exit(2);
    })
}

fn parse_f64(args: &mut impl Iterator<Item = String>, flag: &str) -> f64 {
    let raw = need(args, flag);
    raw.parse().unwrap_or_else(|_| {
        eprintln!("bad {flag} {raw}");
        std::process::exit(2);
    })
}

fn fmt_opt(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "na".to_string())
}
