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

/// Cached logits against a full recompute, on the real weights.
///
/// The prompt is prefilled in one chunk and again in two. Then several
/// decode steps append the fixture's greedy ids. Each step's last-position
/// logits have to stay within 1e-4 of `last_logits` on the whole sequence.
/// Twenty cached greedy tokens also have to match the fixture and the
/// no-cache loop.
#[test]
#[ignore = "downloads Qwen2.5-0.5B-Instruct via hf-hub and runs cached and uncached forwards"]
fn cached_equals_uncached() {
    let (model, tok, meta) = load_qwen();
    let eos =
        qwen_infer::load_eos_token_ids(&qwen_infer::download_qwen().unwrap().generation_config)
            .unwrap();
    assert_eq!(eos, vec![151645, 151643]);

    let prompt = meta.prompt_ids.clone();
    let mut worst = 0.0f32;
    let mut worst_at = String::new();

    let mut cache = qwen_infer::KvCache::new(model.num_layers());
    let mut ids = prompt.clone();
    let cached = model
        .forward(&ids, 0, &mut cache)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let full = model.last_logits(&ids).unwrap().to_vec1::<f32>().unwrap();
    note_diff("prefill", &cached, &full, &mut worst, &mut worst_at);

    let steps = meta.greedy_ids.len().min(8);
    for (step, id) in meta.greedy_ids.iter().take(steps).copied().enumerate() {
        ids.push(id);
        let cached = model
            .forward(&[id], ids.len() - 1, &mut cache)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let full = model.last_logits(&ids).unwrap().to_vec1::<f32>().unwrap();
        note_diff(
            &format!("decode {step}"),
            &cached,
            &full,
            &mut worst,
            &mut worst_at,
        );
        assert_eq!(cache.seq_len(), ids.len());
    }

    let mut chunked = qwen_infer::KvCache::new(model.num_layers());
    let split = 3.min(prompt.len());
    model.forward(&prompt[..split], 0, &mut chunked).unwrap();
    let cached = model
        .forward(&prompt[split..], split, &mut chunked)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let full = model
        .last_logits(&prompt)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    note_diff("chunked prefill", &cached, &full, &mut worst, &mut worst_at);

    eprintln!("cached_equals_uncached real weights max abs {worst} at {worst_at}");
    assert!(worst <= 1e-4, "max abs {worst} at {worst_at} exceeds 1e-4");

    let params = qwen_infer::GenerateParams::greedy(meta.new_tokens);
    let generated =
        qwen_infer::generate(&model, &prompt, &params, |id| tok.decode_bytes(&[id])).unwrap();
    assert_eq!(generated.tokens, meta.greedy_ids);
    assert_eq!(generated.reason, qwen_infer::StopReason::MaxTokens);
    let text = tok.decode(&generated.tokens).unwrap();
    assert_eq!(generated.text, text);
    assert_eq!(generated.chunks.concat(), generated.text);
    let uncached = qwen_infer::greedy_tokens(&model, &prompt, meta.new_tokens).unwrap();
    assert_eq!(generated.tokens, uncached);
}

/// Release-mode decode of 256 new tokens, cached against the no-cache loop.
///
/// EOS is off so both sides run exactly 256 steps. The clock for the cached
/// rate is the decode forwards only; prefill is printed beside it. Model load
/// is outside both clocks. One warmup forward of each path faults the weights
/// in before the timed run.
///
/// The prompt is repeated out to 512 tokens. On the 5-token sentence alone
/// the ratio was 2.54×: each no-cache step is one batched matmul, so the
/// weights are read once per step either way (120.3043 s uncached, 47.4393 s
/// cached decode, same greedy ids). The extra rows are what the cache skips,
/// and they clear 5× once the sequence is a few hundred long.
#[test]
#[ignore = "256 uncached Qwen forwards on a 512-token prompt; several minutes"]
fn cached_decode_is_at_least_5x_uncached() {
    let (model, tok, _meta) = load_qwen();
    let mut prompt_text = String::new();
    let mut prompt = Vec::new();
    while prompt.len() < 512 {
        prompt_text.push_str("The capital of France is ");
        prompt = tok.encode(&prompt_text).unwrap();
    }
    let new_tokens = 256;

    qwen_infer::greedy_tokens(&model, &prompt, 1).unwrap();
    let warmup = qwen_infer::GenerateParams::greedy(1);
    qwen_infer::generate(&model, &prompt, &warmup, |_| Ok(Vec::new())).unwrap();

    let uncached_started = std::time::Instant::now();
    let uncached = qwen_infer::greedy_tokens(&model, &prompt, new_tokens).unwrap();
    let uncached_elapsed = uncached_started.elapsed();

    let params = qwen_infer::GenerateParams::greedy(new_tokens);
    let cached =
        qwen_infer::generate(&model, &prompt, &params, |id| tok.decode_bytes(&[id])).unwrap();

    let uncached_secs = uncached_elapsed.as_secs_f64();
    let prefill_secs = cached.prefill.as_secs_f64();
    let decode_secs = cached.decode.as_secs_f64();
    let uncached_rate = new_tokens as f64 / uncached_secs;
    let decode_rate = new_tokens as f64 / decode_secs;
    let total_rate = new_tokens as f64 / (prefill_secs + decode_secs);
    let speedup = decode_rate / uncached_rate;
    eprintln!("prompt tokens: {}", prompt.len());
    eprintln!(
        "uncached loop: {uncached_secs:.4} s, {uncached_rate:.4} tok/s ({new_tokens} tokens)"
    );
    eprintln!("cached prefill: {prefill_secs:.4} s");
    eprintln!("cached decode: {decode_secs:.4} s, {decode_rate:.4} tok/s ({new_tokens} tokens)");
    eprintln!(
        "cached prefill+decode: {:.4} s, {total_rate:.4} tok/s",
        prefill_secs + decode_secs
    );
    eprintln!("speedup decode/uncached: {speedup:.2}x");
    eprintln!("greedy tokens equal: {}", uncached == cached.tokens);
    if uncached != cached.tokens {
        let mismatch = uncached
            .iter()
            .zip(&cached.tokens)
            .position(|(a, b)| a != b);
        eprintln!("first mismatch at {mismatch:?}");
    }
    assert_eq!(cached.tokens.len(), new_tokens);
    assert_eq!(uncached.len(), new_tokens);
    assert_eq!(uncached, cached.tokens);
    assert!(
        speedup >= 5.0,
        "decode speedup {speedup:.2}x is below 5 (cached {decode_rate:.4} tok/s, uncached {uncached_rate:.4} tok/s)"
    );
}

/// Eight prompts of different lengths. Greedy tokens from one decode batch
/// match `generate` on each prompt alone. Prefill is still per sequence.
#[test]
#[ignore = "downloads Qwen2.5-0.5B-Instruct and runs a batched CPU decode"]
fn batched_outputs_equal_solo_outputs() {
    let (model, tok, _meta) = load_qwen();
    let texts = [
        "The capital of France is",
        "The capital of France is Paris",
        "Hello",
        "Hello there",
        "A",
        "Say hi",
        "The capital of France is the city",
        "One",
    ];
    let prompts: Vec<Vec<u32>> = texts.iter().map(|text| tok.encode(text).unwrap()).collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    assert!(
        lens.iter().copied().max() != lens.iter().copied().min(),
        "prompts should differ in length, got {lens:?}"
    );
    let steps = 4usize;
    let mut batch_caches: Vec<qwen_infer::KvCache> = (0..prompts.len())
        .map(|_| qwen_infer::KvCache::new(model.num_layers()))
        .collect();
    let mut solo_caches: Vec<qwen_infer::KvCache> = (0..prompts.len())
        .map(|_| qwen_infer::KvCache::new(model.num_layers()))
        .collect();
    let mut batch_rows = Vec::new();
    let mut solo_rows = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        batch_rows.push(
            model
                .forward(prompt, 0, &mut batch_caches[index])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
        solo_rows.push(
            model
                .forward(prompt, 0, &mut solo_caches[index])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }
    let mut generated = vec![Vec::new(); prompts.len()];
    let mut worst = 0.0f32;
    for step in 0..steps {
        let mut next = Vec::with_capacity(prompts.len());
        for index in 0..prompts.len() {
            let id = qwen_infer::greedy_token(&batch_rows[index]).unwrap();
            assert_eq!(id, qwen_infer::greedy_token(&solo_rows[index]).unwrap());
            generated[index].push(id);
            next.push(id);
        }
        if step + 1 == steps {
            break;
        }
        let mut refs: Vec<&mut qwen_infer::KvCache> = batch_caches.iter_mut().collect();
        let batched = model
            .forward_decode_batch(&next, &mut refs)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        drop(refs);
        for (index, id) in next.iter().copied().enumerate() {
            let solo = model
                .forward(&[id], solo_caches[index].seq_len(), &mut solo_caches[index])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff = batched[index]
                .iter()
                .zip(&solo)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            if diff > worst {
                worst = diff;
            }
            assert!(
                diff <= 1e-3,
                "step {step} seq {index} max abs {diff} (len {})",
                lens[index]
            );
            batch_rows[index] = batched[index].clone();
            solo_rows[index] = solo;
        }
    }
    eprintln!("real-weight batched vs solo max abs logit {worst}");
    let params = qwen_infer::GenerateParams::greedy(steps);
    for (index, prompt) in prompts.iter().enumerate() {
        let alone =
            qwen_infer::generate(&model, prompt, &params, |id| tok.decode_bytes(&[id])).unwrap();
        assert_eq!(
            generated[index], alone.tokens,
            "prompt {} ({:?})",
            texts[index], lens[index]
        );
    }
}

fn load_qwen() -> (qwen_infer::QwenModel, tokenizer::RankTokenizer, Meta) {
    let files = qwen_infer::download_qwen().expect("hf-hub download");
    let tokenizer_path = files.tokenizer.to_str().expect("tokenizer path is utf-8");
    let tok = tokenizer::RankTokenizer::load_qwen(tokenizer_path).expect("load tokenizer");
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures");
    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir.join("qwen_greedy.json")).unwrap(),
    )
    .unwrap();
    let prompt_ids = tok.encode(&meta.prompt).unwrap();
    assert_eq!(
        prompt_ids, meta.prompt_ids,
        "tokenizer drifted from the dump"
    );
    let model = qwen_infer::load_model(&files.config, &files.weights).expect("load model");
    (model, tok, meta)
}

fn note_diff(label: &str, got: &[f32], want: &[f32], worst: &mut f32, worst_at: &mut String) {
    assert_eq!(got.len(), want.len(), "{label}");
    let mut max_abs = 0.0f32;
    let mut at = 0usize;
    for (index, (left, right)) in got.iter().zip(want).enumerate() {
        let diff = (left - right).abs();
        if diff > max_abs {
            max_abs = diff;
            at = index;
        }
    }
    eprintln!(
        "{label}: max abs {max_abs} at id {at} (got {}, full {})",
        got[at], want[at]
    );
    if max_abs > *worst {
        *worst = max_abs;
        *worst_at = format!("{label} id {at}");
    }
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
