//! Arbitrary bytes → lossy UTF-8 → cl100k encode, compared to tiktoken-rs.
//!
//! `libfuzzer_sys::fuzz_target!` is the harness. The input is `&[u8]` because
//! libFuzzer mutates bytes, not Unicode. `String::from_utf8_lossy` is what
//! both encoders then see, including U+FFFD where the bytes were not UTF-8.
//! A mismatch or a panic is a crash; libFuzzer writes the bytes under
//! `fuzz/artifacts/`, which is gitignored. A crash worth keeping is copied
//! into a regression test by hand.

#![no_main]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use tiktoken_rs::CoreBPE;
use tokenizer::RankTokenizer;

const CL100K_URL: &str =
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken";
const CL100K_SHA256: &str = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";

struct Oracles {
    ours: RankTokenizer,
    theirs: CoreBPE,
}

fn oracles() -> &'static Oracles {
    static ORACLES: OnceLock<Oracles> = OnceLock::new();
    ORACLES.get_or_init(|| {
        let path = ensure_vocab();
        let ours = RankTokenizer::load_cl100k(path.to_str().expect("cache path is utf-8"))
            .unwrap_or_else(|err| panic!("load_cl100k({}): {err}", path.display()));
        let theirs = tiktoken_rs::cl100k_base()
            .unwrap_or_else(|err| panic!("tiktoken-rs cl100k_base: {err}"));
        Oracles { ours, theirs }
    })
}

fn cache_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME is set");
    PathBuf::from(home).join(".cache/tokenizer/cl100k_base.tiktoken")
}

fn ensure_vocab() -> PathBuf {
    let path = cache_path();
    if path.is_file() && file_sha256(&path) == CL100K_SHA256 {
        return path;
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create vocab cache dir");
    }
    let tmp = path.with_extension("partial");
    let status = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&tmp)
        .arg(CL100K_URL)
        .status()
        .unwrap_or_else(|err| panic!("could not run curl ({err}); run scripts/fetch-cl100k.sh"));
    assert!(status.success(), "curl failed to download cl100k: {status}");
    assert_eq!(
        file_sha256(&tmp),
        CL100K_SHA256,
        "cl100k download hash mismatch"
    );
    std::fs::rename(&tmp, &path).expect("rename cl100k download");
    path
}

fn file_sha256(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .unwrap_or_else(|err| panic!("sha256sum failed to start: {err}"));
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .expect("sha256sum output")
        .to_string()
}

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let pair = oracles();
    let ours = pair
        .ours
        .encode(&text)
        .unwrap_or_else(|err| panic!("encode ({} bytes): {err}", text.len()));
    let theirs = pair.theirs.encode_with_special_tokens(&text);
    if ours != theirs {
        let idx = ours
            .iter()
            .zip(theirs.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(ours.len().min(theirs.len()));
        panic!(
            "encode mismatch at token {idx}: ours {:?} theirs {:?} (ours {} ids, theirs {} ids, {} bytes)",
            ours.get(idx),
            theirs.get(idx),
            ours.len(),
            theirs.len(),
            text.len()
        );
    }
});
