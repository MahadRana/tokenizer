//! Fixture and book parity against Python tiktoken. The ids in
//! `tests/fixtures/cl100k_cases.jsonl` come from `scripts/gen_cl100k_cases.py`
//! (tiktoken 0.14.0). This module does not invent token ids.

use serde::Deserialize;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::rank::ensure_cl100k_file;
use crate::{RankTokenizer, TokenizerError};

#[derive(Debug, Deserialize)]
struct Case {
    text: String,
    specials: Specials,
    #[serde(default)]
    ids: Option<Vec<u32>>,
    #[serde(default)]
    disallowed: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Specials {
    All(String),
    Only(Vec<String>),
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cl100k_cases.jsonl")
}

fn read_cases() -> Vec<Case> {
    let path = fixture_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut cases = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let case: Case = serde_json::from_str(line)
            .unwrap_or_else(|err| panic!("{} line {}: {err}", path.display(), idx + 1));
        cases.push(case);
    }
    cases
}

fn loaded_cl100k() -> &'static RankTokenizer {
    static MODEL: OnceLock<RankTokenizer> = OnceLock::new();
    MODEL.get_or_init(|| {
        let path = ensure_cl100k_file();
        RankTokenizer::load_cl100k(path.to_str().expect("cache path is utf-8"))
            .unwrap_or_else(|err| panic!("load_cl100k({}): {err}", path.display()))
    })
}

const CL100K_SPECIALS: &[&str] = &[
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|endofprompt|>",
];

#[test]
fn cl100k_cases_fixture_covers_the_required_shapes() {
    let cases = read_cases();
    assert!(cases.len() >= 200, "got {} cases", cases.len());

    assert!(
        cases.iter().any(|case| {
            case.text.contains("'s") || case.text.contains("'LL") || case.text.contains("'ll")
        }),
        "missing a contraction"
    );
    assert!(
        cases.iter().any(|case| case.text.contains("12345")),
        "missing a digit run"
    );
    assert!(cases.iter().any(|case| case.text.contains("  ")));
    assert!(cases.iter().any(|case| case.text.contains('\t')));
    assert!(cases.iter().any(|case| case.text.contains('\n')));
    assert!(cases.iter().any(|case| case.text.contains("\r\n")));
    assert!(
        cases
            .iter()
            .any(|case| case.text.contains('🦀') || case.text.contains("👨‍👩‍👧‍👦"))
    );
    assert!(
        cases
            .iter()
            .any(|case| case.text.contains('日') || case.text.contains('你'))
    );
    assert!(
        cases
            .iter()
            .any(|case| case.text.chars().any(|ch| ch == '\u{0301}')),
        "missing a combining mark"
    );
    assert!(
        cases
            .iter()
            .any(|case| case.disallowed.as_deref() == Some("<|endoftext|>")),
        "missing a disallowed <|endoftext|>"
    );
    assert!(
        cases.iter().any(|case| {
            matches!(&case.specials, Specials::Only(list) if list.iter().any(|s| s == "<|endoftext|>"))
                && case.ids.is_some()
                && case.disallowed.is_none()
        }),
        "missing an allowed <|endoftext|>"
    );

    for (idx, case) in cases.iter().enumerate() {
        match (
            &case.specials,
            case.ids.is_some(),
            case.disallowed.is_some(),
        ) {
            (Specials::All(mode), true, false) => assert_eq!(mode, "all", "case {idx}"),
            (Specials::Only(_), true, false) | (Specials::Only(_), false, true) => {}
            _ => panic!("case {idx} is neither a success nor a disallowed special: {case:?}"),
        }
    }
}

#[test]
#[ignore = "compares every fixture id to cl100k; needs the cached vocab"]
fn encode_matches_tiktoken_fixtures() {
    let tok = loaded_cl100k();
    let cases = read_cases();
    assert!(cases.len() >= 200, "got {} cases", cases.len());
    let all_specials: HashSet<&str> = CL100K_SPECIALS.iter().copied().collect();

    for (idx, case) in cases.iter().enumerate() {
        match &case.specials {
            Specials::All(mode) => {
                assert_eq!(mode, "all", "case {idx}");
                let expected = case.ids.as_ref().expect("success case has ids");
                let got = tok
                    .encode(&case.text)
                    .unwrap_or_else(|err| panic!("case {idx}: {err}"));
                assert_eq!(got, *expected, "case {idx} text {:?}", case.text);
                assert_eq!(
                    tok.encode_with_special_tokens(&case.text, &all_specials)
                        .unwrap(),
                    got,
                    "case {idx}: encode and encode_with_special_tokens diverged"
                );
                assert_eq!(
                    tok.decode(&got).unwrap(),
                    case.text,
                    "case {idx} round trip"
                );
            }
            Specials::Only(allowed) => {
                let set: HashSet<&str> = allowed.iter().map(String::as_str).collect();
                match &case.disallowed {
                    Some(token) => {
                        let err = tok
                            .encode_with_special_tokens(&case.text, &set)
                            .expect_err(&format!("case {idx} should reject {token}"));
                        assert!(
                            matches!(err, TokenizerError::DisallowedSpecial(ref got) if got == token),
                            "case {idx}: {err}"
                        );
                    }
                    None => {
                        let expected = case.ids.as_ref().expect("success case has ids");
                        let got = tok
                            .encode_with_special_tokens(&case.text, &set)
                            .unwrap_or_else(|err| panic!("case {idx}: {err}"));
                        assert_eq!(got, *expected, "case {idx} text {:?}", case.text);
                        assert_eq!(
                            tok.decode(&got).unwrap(),
                            case.text,
                            "case {idx} round trip"
                        );
                    }
                }
            }
        }
    }

    // The regex changes these. Piece encoding is the A3 path and must not
    // be what `encode` returns.
    assert_ne!(
        tok.encode("999999").unwrap(),
        tok.encode_piece("999999"),
        "digit groups should not be one piece"
    );
    assert_ne!(tok.encode("  hello").unwrap(), tok.encode_piece("  hello"));
}

#[test]
#[ignore = "encodes Pride and Prejudice against tiktoken; needs book.txt, the vocab cache, and python tiktoken"]
fn encode_matches_tiktoken_on_book() {
    let tok = loaded_cl100k();
    let text =
        std::fs::read_to_string("book.txt").expect("book.txt is required for this ignored test");
    let out = std::env::temp_dir().join(format!("tokenizer-a4-book-{}.bin", std::process::id()));
    let output = std::process::Command::new("python3")
        .arg("scripts/gen_cl100k_cases.py")
        .arg("--book")
        .arg("book.txt")
        .arg("--ids-out")
        .arg(&out)
        .output()
        .unwrap_or_else(|err| panic!("could not run python3 ({err}); pip install tiktoken"));
    assert!(
        output.status.success(),
        "tiktoken book encode failed: {}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&out).unwrap_or_else(|err| panic!("read ids: {err}"));
    let _ = std::fs::remove_file(&out);
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "id file length {}", bytes.len());
    let expected: Vec<u32> = chunks.iter().copied().map(u32::from_le_bytes).collect();

    let got = tok.encode(&text).unwrap();
    if got != expected {
        let idx = got
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(expected.len()));
        panic!(
            "book mismatch at token {idx}: rust {:?} tiktoken {:?} (rust {} ids, tiktoken {} ids)",
            got.get(idx),
            expected.get(idx),
            got.len(),
            expected.len()
        );
    }
    assert_eq!(tok.decode(&got).unwrap(), text);
    eprintln!("book bytes {} tokens {}", text.len(), got.len());
}
