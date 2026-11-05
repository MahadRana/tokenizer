//! Logits and greedy tokens against the Python dump.
//!
//! `scripts/dump_qwen_reference.py` writes the two fixtures. This test loads
//! the same weights through hf-hub (`DType::F32`, `Device::Cpu`) and checks
//! the last-position logits within 1e-3 max-abs, then 20 greedy ids.
//!
//! Ignored so `cargo test` does not download the model. Run:
//!
//! ```text
//! cargo test --release -p qwen-infer -- --ignored
//! ```

use std::path::PathBuf;

#[test]
#[ignore = "downloads Qwen2.5-0.5B-Instruct via hf-hub and runs a CPU forward"]
fn logits_and_greedy_match_transformers() {
    let files = qwen_infer::download_qwen().expect("hf-hub download");
    let tokenizer_path = files.tokenizer.to_str().expect("tokenizer path is utf-8");
    let tok = tokenizer::RankTokenizer::load_qwen(tokenizer_path).expect("load tokenizer");

    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures");
    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir.join("qwen_greedy.json")).unwrap(),
    )
    .unwrap();
    let reference = read_f32(&fixture_dir.join("qwen_last_logits.f32"));
    assert_eq!(reference.len(), meta.vocab, "logit file length");

    let prompt_ids = tok.encode(&meta.prompt).unwrap();
    assert_eq!(
        prompt_ids, meta.prompt_ids,
        "tokenizer drifted from the dump"
    );

    let model = qwen_infer::load_model(&files.config, &files.weights).expect("load model");
    let logits = model
        .last_logits(&prompt_ids)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(logits.len(), reference.len());
    let mut max_abs = 0.0f32;
    let mut at = 0usize;
    for (index, (got, want)) in logits.iter().zip(&reference).enumerate() {
        let diff = (got - want).abs();
        if diff > max_abs {
            max_abs = diff;
            at = index;
        }
    }
    eprintln!(
        "max abs logit diff {max_abs} at id {at} (got {}, reference {})",
        logits[at], reference[at]
    );
    assert!(
        max_abs <= 1e-3,
        "max abs {max_abs} at id {at}: got {} reference {}",
        logits[at],
        reference[at]
    );

    let generated = qwen_infer::greedy_tokens(&model, &prompt_ids, meta.new_tokens).unwrap();
    assert_eq!(generated, meta.greedy_ids);
}

#[derive(serde::Deserialize)]
struct Meta {
    prompt: String,
    prompt_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
    vocab: usize,
    new_tokens: usize,
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "logit file length {}", bytes.len());
    chunks.iter().copied().map(f32::from_le_bytes).collect()
}
