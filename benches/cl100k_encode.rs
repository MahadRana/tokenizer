//! Throughput of cl100k `RankTokenizer::encode` on Pride and Prejudice.
//!
//! Needs `book.txt` in the crate root (gitignored) and the vocab cached at
//! `~/.cache/tokenizer/cl100k_base.tiktoken` (`scripts/fetch-cl100k.sh`).
//! Criterion reports throughput from [`criterion::Throughput::Bytes`].

use std::path::PathBuf;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tokenizer::RankTokenizer;

fn encode_book(c: &mut Criterion) {
    let path = cl100k_cache();
    let tok = RankTokenizer::load_cl100k(path.to_str().expect("cache path is utf-8"))
        .unwrap_or_else(|err| {
            panic!(
                "load {}: {err}\nrun scripts/fetch-cl100k.sh",
                path.display()
            )
        });
    let text = std::fs::read_to_string("book.txt")
        .expect("book.txt (Pride and Prejudice) is required; it is gitignored");

    let mut group = c.benchmark_group("cl100k_encode");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("book", |b| {
        b.iter(|| tok.encode(std::hint::black_box(&text)).expect("encode"));
    });
    group.finish();
}

fn cl100k_cache() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME is set");
    PathBuf::from(home).join(".cache/tokenizer/cl100k_base.tiktoken")
}

criterion_group!(benches, encode_book);
criterion_main!(benches);
