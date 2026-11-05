//! Load a Qwen2.5 `tokenizer.json` into [`RankTokenizer`](crate::RankTokenizer).
//!
//! The file is Hugging Face's tokenizers JSON (version 1.0), not a
//! `.tiktoken` vocab. Three things in it are not cl100k:
//!
//! - Vocab strings are GPT-2 `bytes_to_unicode` characters, not raw bytes.
//!   Space is `Ġ` (U+0120), not `0x20`. The loader unmaps those characters
//!   back to bytes before they go in the encoder.
//! - Merge priority is the order of the `merges` array. Token ids are not
//!   that order (the first merge is not id 0), so the rank map is separate
//!   from the id map. [`RankTokenizer`]'s piece encoder still picks the
//!   lowest rank, which is the same one-pair-at-a-time loop the `tokenizers`
//!   crate runs in `Word::merge_all`.
//! - The normalizer is NFC, and the pre-tokenizer is a regex split followed
//!   by a byte-level mapping. Added tokens are matched as literals first.
//!   Every added token in the Qwen2.5 file has `normalized: false`, so they
//!   are split out before NFC. A token with `normalized: true` is rejected
//!   rather than silently normalized.

use crate::RankTokenizer;
use crate::error::TokenizerError;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;

#[derive(Deserialize)]
struct TokenizerFile {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    normalizer: Option<Typed>,
    pre_tokenizer: PreTokenizer,
    model: BpeModel,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    /// Hugging Face defaults this to true when the field is absent. Qwen sets
    /// it false on every added token. `None` here means the field was absent.
    normalized: Option<bool>,
}

#[derive(Deserialize)]
struct Typed {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct PreTokenizer {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    pretokenizers: Vec<PreStep>,
}

#[derive(Deserialize)]
struct PreStep {
    #[serde(rename = "type")]
    kind: String,
    pattern: Option<RegexPattern>,
    behavior: Option<String>,
    invert: Option<bool>,
    use_regex: Option<bool>,
}

/// `{"Regex": "..."}` inside a Split pre-tokenizer.
#[derive(Deserialize)]
struct RegexPattern {
    #[serde(rename = "Regex")]
    regex: String,
}

#[derive(Deserialize)]
struct BpeModel {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    merges: Vec<String>,
}

fn parse_err(message: impl Into<String>) -> TokenizerError {
    TokenizerError::Parse {
        line: 0,
        message: message.into(),
    }
}

/// GPT-2 `bytes_to_unicode`, inverted: the unicode scalar in a vocab string
/// maps back to the original byte.
///
/// Printable ASCII and the latin-1 range `¡`–`¬`, `®`–`ÿ` map to themselves.
/// Every other byte is assigned `U+0100` upward, in byte order. Space (byte
/// 32) is the 33rd byte that misses that set, so it becomes U+0120 (`Ġ`).
fn bytes_to_unicode_reverse() -> HashMap<char, u8> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut codepoints: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    let mut next = 0u32;
    for byte in 0u16..=255 {
        let byte = byte as u8;
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + next);
            next += 1;
        }
    }
    bytes
        .into_iter()
        .zip(codepoints)
        .map(|(byte, code)| {
            (
                char::from_u32(code).expect("bytes_to_unicode codepoint fits in a char"),
                byte,
            )
        })
        .collect()
}

fn unmap(token: &str, rev: &HashMap<char, u8>) -> Result<Vec<u8>, TokenizerError> {
    let mut bytes = Vec::with_capacity(token.chars().count());
    for ch in token.chars() {
        let Some(&byte) = rev.get(&ch) else {
            return Err(parse_err(format!(
                "vocab char U+{ch:04X} is not a bytes_to_unicode scalar",
                ch = ch as u32
            )));
        };
        bytes.push(byte);
    }
    Ok(bytes)
}

pub(crate) fn parse_qwen_file(path: &str) -> Result<RankTokenizer, TokenizerError> {
    let text = fs::read_to_string(path)?;
    let file: TokenizerFile = serde_json::from_str(&text)?;
    parse_qwen(file)
}

fn parse_qwen(file: TokenizerFile) -> Result<RankTokenizer, TokenizerError> {
    let normalizer = file
        .normalizer
        .ok_or_else(|| parse_err("tokenizer.json has no normalizer"))?;
    if normalizer.kind != "NFC" {
        return Err(parse_err(format!(
            "expected an NFC normalizer, found {}",
            normalizer.kind
        )));
    }
    if file.pre_tokenizer.kind != "Sequence" {
        return Err(parse_err(format!(
            "expected a Sequence pre-tokenizer, found {}",
            file.pre_tokenizer.kind
        )));
    }
    let pattern = qwen_pattern(&file.pre_tokenizer)?;
    if file.model.kind != "BPE" {
        return Err(parse_err(format!(
            "expected a BPE model, found {}",
            file.model.kind
        )));
    }

    let rev = bytes_to_unicode_reverse();
    let mut encoder: HashMap<Vec<u8>, u32> = HashMap::with_capacity(file.model.vocab.len());
    let mut max_id = 0u32;
    for (token, id) in &file.model.vocab {
        let bytes = unmap(token, &rev)?;
        if encoder.contains_key(bytes.as_slice()) {
            return Err(parse_err(format!("duplicate vocab bytes for id {id}")));
        }
        encoder.insert(bytes, *id);
        max_id = max_id.max(*id);
    }
    for byte in 0..=u8::MAX {
        if !encoder.contains_key(std::slice::from_ref(&byte)) {
            return Err(parse_err(format!("missing single-byte token {byte}")));
        }
    }

    let mut ranks: HashMap<Vec<u8>, u32> = HashMap::with_capacity(file.model.merges.len());
    for (index, merge) in file.model.merges.iter().enumerate() {
        let Some((left, right)) = merge.split_once(' ') else {
            return Err(parse_err(format!("merge {index} has no space")));
        };
        if left.is_empty() || right.is_empty() || right.contains(' ') {
            return Err(parse_err(format!(
                "merge {index} is not two bytes_to_unicode tokens"
            )));
        }
        let left_bytes = unmap(left, &rev)?;
        let right_bytes = unmap(right, &rev)?;
        if !encoder.contains_key(left_bytes.as_slice())
            || !encoder.contains_key(right_bytes.as_slice())
        {
            return Err(parse_err(format!(
                "merge {index} names a token that is not in the vocab"
            )));
        }
        let mut merged = left_bytes;
        merged.extend_from_slice(&right_bytes);
        if !encoder.contains_key(merged.as_slice()) {
            return Err(parse_err(format!(
                "merge {index} concatenation is not in the vocab"
            )));
        }
        let rank =
            u32::try_from(index).map_err(|_| parse_err("merge index does not fit in u32"))?;
        if ranks.insert(merged, rank).is_some() {
            return Err(parse_err(format!("merge {index} repeats a byte string")));
        }
    }

    for added in &file.added_tokens {
        max_id = max_id.max(added.id);
    }
    let len = usize::try_from(max_id).map_err(|_| parse_err("token id does not fit in usize"))? + 1;
    let mut decoder = vec![None; len];
    let mut seen_ids = HashSet::with_capacity(encoder.len() + file.added_tokens.len());
    for (bytes, &id) in &encoder {
        let idx = usize::try_from(id).map_err(|_| parse_err("token id does not fit in usize"))?;
        if !seen_ids.insert(id) {
            return Err(parse_err(format!("duplicate token id {id}")));
        }
        decoder[idx] = Some(bytes.clone());
    }

    let mut specials = HashMap::with_capacity(file.added_tokens.len());
    for added in &file.added_tokens {
        // `tokenizers` treats a missing `normalized` as true. Those tokens
        // are extracted after NFC. This loader only implements the Qwen
        // file, where every added token is `normalized: false` and is
        // extracted first.
        if added.normalized.unwrap_or(true) {
            return Err(parse_err(format!(
                "added token {:?} is normalized; only normalized=false is supported",
                added.content
            )));
        }
        if added.content.is_empty() {
            return Err(parse_err("added token content is empty"));
        }
        let idx = usize::try_from(added.id)
            .map_err(|_| parse_err("added token id does not fit in usize"))?;
        if decoder[idx].is_some() {
            return Err(parse_err(format!(
                "added token id {} already belongs to a vocab token",
                added.id
            )));
        }
        decoder[idx] = Some(added.content.as_bytes().to_vec());
        if specials.insert(added.content.clone(), added.id).is_some() {
            return Err(parse_err(format!(
                "duplicate added token {:?}",
                added.content
            )));
        }
    }

    RankTokenizer::from_qwen_parts(encoder, decoder, ranks, specials, &pattern)
}

fn qwen_pattern(pre: &PreTokenizer) -> Result<String, TokenizerError> {
    let split = pre
        .pretokenizers
        .iter()
        .find(|step| step.kind == "Split")
        .ok_or_else(|| parse_err("pre-tokenizer has no Split step"))?;
    if split.behavior.as_deref() != Some("Isolated") {
        return Err(parse_err(format!(
            "expected Split behavior Isolated, found {:?}",
            split.behavior
        )));
    }
    if split.invert.unwrap_or(false) {
        return Err(parse_err("Split invert=true is not supported"));
    }
    let pattern = split
        .pattern
        .as_ref()
        .map(|pat| pat.regex.clone())
        .ok_or_else(|| parse_err("Split step has no Regex pattern"))?;
    let byte_level = pre
        .pretokenizers
        .iter()
        .find(|step| step.kind == "ByteLevel")
        .ok_or_else(|| parse_err("pre-tokenizer has no ByteLevel step"))?;
    if byte_level.use_regex.unwrap_or(false) {
        return Err(parse_err(
            "ByteLevel use_regex must be false; the Split step owns the regex",
        ));
    }
    Ok(pattern)
}

#[cfg(test)]
fn qwen_cache_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME is set");
    std::path::PathBuf::from(home).join(".cache/tokenizer/qwen2.5-0.5b-instruct/tokenizer.json")
}

/// Download Qwen2.5-0.5B-Instruct `tokenizer.json` into `~/.cache` when the
/// ignored test needs it. The file is not in git. A cached file is used only
/// when its sha256 matches the copy this milestone checked.
#[cfg(test)]
pub(crate) fn ensure_qwen_tokenizer() -> std::path::PathBuf {
    const URL: &str =
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json";
    const HASH: &str = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
    let path = qwen_cache_path();
    if path.is_file() && file_sha256(&path) == HASH {
        return path;
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let tmp = path.with_extension("partial");
    let status = std::process::Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&tmp)
        .arg(URL)
        .status()
        .unwrap_or_else(|err| panic!("could not run curl ({err})"));
    assert!(
        status.success(),
        "curl failed to download tokenizer.json: {status}"
    );
    assert_eq!(
        file_sha256(&tmp),
        HASH,
        "tokenizer.json download hash mismatch"
    );
    std::fs::rename(&tmp, &path).unwrap();
    path
}

#[cfg(test)]
fn file_sha256(path: &std::path::Path) -> String {
    let output = std::process::Command::new("sha256sum")
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    #[test]
    fn bytes_to_unicode_maps_the_known_bytes() {
        let rev = bytes_to_unicode_reverse();
        assert_eq!(rev.len(), 256);
        // Byte 0 is the first byte missing from the identity set, so it is
        // U+0100. Space is byte 32, the 33rd missing byte: 256 + 32 = 288.
        assert_eq!(rev[&'\u{0100}'], 0);
        assert_eq!(rev[&'\u{0120}'], b' ');
        assert_eq!(rev[&'!'], b'!');
        assert_eq!(rev[&'a'], b'a');
        let mut seen = HashSet::new();
        for byte in rev.values() {
            assert!(seen.insert(*byte), "duplicate byte {byte}");
        }
        assert_eq!(seen.len(), 256);
    }

    fn tiny_file() -> TokenizerFile {
        let rev = bytes_to_unicode_reverse();
        let mut of_byte = HashMap::new();
        for (&ch, &byte) in &rev {
            of_byte.insert(byte, ch);
        }
        let mut vocab = HashMap::new();
        for byte in 0..=u8::MAX {
            let mut token = String::new();
            token.push(of_byte[&byte]);
            vocab.insert(token, u32::from(byte));
        }
        // 'a' and 'b' are identity bytes. The merge result "ab" is id 256,
        // which is not the merge index (0). That is the Qwen shape: rank and
        // id are different numbers.
        vocab.insert("ab".to_string(), 256);
        let json = serde_json::json!({
            "added_tokens": [{
                "id": 300,
                "content": "<|endoftext|>",
                "normalized": false,
                "special": true
            }],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": "[\\s\\S]+"},
                        "behavior": "Isolated",
                        "invert": false
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "use_regex": false
                    }
                ]
            },
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": ["a b"]
            }
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn tiny_qwen_merges_by_rank_not_by_id_and_applies_nfc() {
        let tok = parse_qwen(tiny_file()).unwrap();
        assert_eq!(tok.encode("ab").unwrap(), vec![256]);
        assert_eq!(
            tok.encode("ba").unwrap(),
            vec![u32::from(b'b'), u32::from(b'a')]
        );
        assert_eq!(tok.encode("aba").unwrap(), vec![256, u32::from(b'a')]);
        assert_eq!(tok.decode(&[256]).unwrap(), "ab");

        // U+0301 is the combining acute. NFC folds "e" + acute into "é"
        // (UTF-8 c3 a9) before BPE. The two spellings are one encoding.
        let composed = tok.encode("é").unwrap();
        let decomposed = tok.encode("e\u{0301}").unwrap();
        assert_eq!(decomposed, composed);
        assert_ne!(
            composed,
            vec![u32::from(b'e'), 0xCC, 0x81],
            "decomposed bytes must not survive NFC"
        );
        assert_eq!(tok.decode(&composed).unwrap(), "é");

        let ids = tok.encode("a<|endoftext|>b").unwrap();
        assert_eq!(ids, vec![u32::from(b'a'), 300, u32::from(b'b')]);
        assert_eq!(tok.decode(&ids).unwrap(), "a<|endoftext|>b");
        let err = tok.decode(&[257]).unwrap_err();
        assert!(matches!(err, TokenizerError::UnknownToken(257)));
    }

    #[test]
    fn normalized_added_token_is_rejected() {
        let mut file = tiny_file();
        file.added_tokens[0].normalized = Some(true);
        let err = parse_qwen(file).unwrap_err();
        assert!(
            matches!(err, TokenizerError::Parse { line: 0, .. }),
            "{err}"
        );
        assert!(err.to_string().contains("normalized"), "{err}");
    }

    #[derive(Deserialize)]
    struct Case {
        text: String,
        ids: Vec<u32>,
    }

    fn fixture_cases() -> Vec<Case> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen_cases.jsonl");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        text.lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn qwen_cases_fixture_covers_the_required_shapes() {
        let cases = fixture_cases();
        assert!(cases.len() >= 200, "got {} cases", cases.len());
        assert!(
            cases
                .iter()
                .any(|case| case.text.contains("'s") || case.text.contains("'ll")),
            "missing a contraction"
        );
        assert!(
            cases.iter().any(|case| case.text.contains("12345")),
            "missing a digit run"
        );
        assert!(
            cases
                .iter()
                .any(|case| case.text.contains('\n') || case.text.contains('\t')),
            "missing whitespace"
        );
        assert!(
            cases
                .iter()
                .any(|case| case.text.contains("🦀") || case.text.contains('🙂')),
            "missing emoji"
        );
        assert!(
            cases
                .iter()
                .any(|case| case.text.contains('你') || case.text.contains('日')),
            "missing CJK"
        );
        assert!(
            cases.iter().any(|case| case.text.contains('\u{0301}')),
            "missing a combining mark"
        );
        assert!(
            cases.iter().any(|case| case.text.contains("<|endoftext|>")),
            "missing a special"
        );
        assert!(
            cases
                .iter()
                .any(|case| case.text == "The capital of France is"),
            "missing the generation prompt"
        );
    }

    fn loaded_qwen() -> &'static RankTokenizer {
        static MODEL: OnceLock<RankTokenizer> = OnceLock::new();
        MODEL.get_or_init(|| {
            let path = ensure_qwen_tokenizer();
            RankTokenizer::load_qwen(path.to_str().expect("cache path is utf-8"))
                .unwrap_or_else(|err| panic!("load_qwen({}): {err}", path.display()))
        })
    }

    #[test]
    #[ignore = "needs Qwen tokenizer.json in ~/.cache; curl downloads it on a hash miss"]
    fn qwen_tokenizer_matches_hf() {
        let tok = loaded_qwen();
        let cases = fixture_cases();
        assert!(cases.len() >= 200, "got {} cases", cases.len());
        for case in &cases {
            let got = tok.encode(&case.text).unwrap();
            assert_eq!(got, case.ids, "{:?}", case.text);
        }
    }
}
