//! Generation loop: prefill once, then one cached token at a time.
//!
//! The model stays `&QwenModel`. The [`KvCache`] is local to the call, so two
//! sequences never share one. Stop strings are applied to the decoded text,
//! not to token ids, and a string that is still only a prefix is held back
//! from the chunks so it is not streamed and then deleted.

use crate::model::{KvCache, QwenModel};
use crate::sample::{self, SampleError};
use crate::stream::StreamDecoder;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Duration;
use tokenizer::TokenizerError;

/// How [`generate`] decided to return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// `max_tokens` new ids were sampled and none of them was EOS.
    MaxTokens,
    /// The sampled id was in [`GenerateParams::eos`]. That id is not in
    /// [`Generated::tokens`] and its text is not decoded.
    Eos,
    /// A stop string occurred in the decoded text. The string itself is not
    /// in [`Generated::text`] or in any streamed chunk.
    StopString,
}

/// Sampling and stop settings for one [`generate`] call.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Seed for [`StdRng`]. Temperature 0 does not draw, so the seed does not
    /// change a greedy run.
    pub seed: u64,
    /// Any of these ids ends the loop. Qwen2.5-0.5B-Instruct lists
    /// 151645 and 151643 in `generation_config.json`.
    pub eos: Vec<u32>,
    /// Literal strings. The match is on decoded UTF-8, so it can cross a
    /// token boundary. Empty strings are rejected.
    pub stop: Vec<String>,
}

impl GenerateParams {
    /// Greedy, no EOS, no stop strings. `top_p` is 1.
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
            eos: Vec::new(),
            stop: Vec::new(),
        }
    }
}

/// One finished generation.
#[derive(Debug, Clone)]
pub struct Generated {
    /// Sampled ids, excluding an EOS id.
    ///
    /// The id that completed a stop string is included. Characters from that
    /// id that belong to the stop string are not part of [`Self::text`].
    pub tokens: Vec<u32>,
    /// Decoded text with stop strings removed.
    pub text: String,
    /// Pieces yielded as soon as they were safe to show. Joining them is
    /// [`Self::text`]. A stop string never appears in a piece.
    pub chunks: Vec<String>,
    pub reason: StopReason,
    /// The prompt forward, before the first sampled id.
    pub prefill: Duration,
    /// The decode forwards, after prefill. Sampling itself is in this
    /// interval too; it is not the cost.
    pub decode: Duration,
}

/// Prefill `prompt` at position 0, then sample up to `params.max_tokens` new
/// ids. Each new id is one cached forward at `cache.seq_len()`.
///
/// `token_bytes` is `Tokenizer::decode_bytes` (or `RankTokenizer::decode_bytes`)
/// on a single id. The streaming decoder concatenates those bytes.
pub fn generate(
    model: &QwenModel,
    prompt: &[u32],
    params: &GenerateParams,
    mut token_bytes: impl FnMut(u32) -> Result<Vec<u8>, TokenizerError>,
) -> anyhow::Result<Generated> {
    validate(params)?;
    if prompt.is_empty() {
        anyhow::bail!("prompt encoded to no tokens");
    }

    let mut cache = KvCache::new(model.num_layers());
    let mut rng = StdRng::seed_from_u64(params.seed);
    let prefill_started = std::time::Instant::now();
    let mut logits = model.forward(prompt, 0, &mut cache)?;
    let prefill = prefill_started.elapsed();

    let mut decoder = StreamDecoder::new();
    let mut held = StopHold::default();
    let mut tokens = Vec::new();
    let mut chunks = Vec::new();
    let mut text = String::new();
    let decode_started = std::time::Instant::now();
    let mut reason = StopReason::MaxTokens;

    for _ in 0..params.max_tokens {
        let values = logits.to_vec1::<f32>()?;
        let id = sample::sample_token(&values, params.temperature, params.top_p, &mut rng)?;
        if params.eos.contains(&id) {
            reason = StopReason::Eos;
            break;
        }
        let bytes = token_bytes(id)?;
        let piece = decoder.push(&bytes);
        match held.push(&piece, &params.stop) {
            HoldOut::Stop(chunk) => {
                emit(&mut chunks, &mut text, &chunk);
                tokens.push(id);
                reason = StopReason::StopString;
                break;
            }
            HoldOut::Continue(chunk) => {
                emit(&mut chunks, &mut text, &chunk);
                tokens.push(id);
            }
        }
        let pos = cache.seq_len();
        logits = model.forward(&[id], pos, &mut cache)?;
    }

    if reason != StopReason::StopString {
        // The decoder may still be holding a split character. `finish` is the
        // end of the stream, so a replacement here is not a mid-stream guess.
        let flushed = decoder.finish();
        if !flushed.is_empty() {
            match held.push(&flushed, &params.stop) {
                HoldOut::Stop(chunk) => {
                    emit(&mut chunks, &mut text, &chunk);
                    reason = StopReason::StopString;
                }
                HoldOut::Continue(chunk) => {
                    emit(&mut chunks, &mut text, &chunk);
                    emit(&mut chunks, &mut text, &held.finish());
                }
            }
        } else {
            emit(&mut chunks, &mut text, &held.finish());
        }
    }
    let decode = decode_started.elapsed();

    Ok(Generated {
        tokens,
        text,
        chunks,
        reason,
        prefill,
        decode,
    })
}

fn validate(params: &GenerateParams) -> anyhow::Result<()> {
    if !params.temperature.is_finite() {
        return Err(SampleError::NonFiniteTemperature(params.temperature).into());
    }
    if params.temperature < 0.0 {
        return Err(SampleError::NegativeTemperature(params.temperature).into());
    }
    if !params.top_p.is_finite() {
        return Err(SampleError::NonFiniteTopP(params.top_p).into());
    }
    if params.top_p < 0.0 {
        return Err(SampleError::NegativeTopP(params.top_p).into());
    }
    if params.stop.iter().any(|stop| stop.is_empty()) {
        anyhow::bail!("stop strings must be non-empty");
    }
    Ok(())
}

fn emit(chunks: &mut Vec<String>, text: &mut String, piece: &str) {
    if piece.is_empty() {
        return;
    }
    chunks.push(piece.to_string());
    text.push_str(piece);
}

/// Decoded text, plus how much of it has already been handed to the caller.
///
/// A suffix that is a proper prefix of a stop string stays unemitted. The
/// next chunk either completes the stop (and the suffix is dropped) or shows
/// that the suffix was ordinary text (and it is released).
#[derive(Debug, Default)]
struct StopHold {
    full: String,
    emitted: usize,
}

enum HoldOut {
    Continue(String),
    Stop(String),
}

impl StopHold {
    fn push(&mut self, chunk: &str, stops: &[String]) -> HoldOut {
        self.full.push_str(chunk);
        if let Some(pos) = earliest_stop(&self.full, stops) {
            let emit = if pos > self.emitted {
                self.full[self.emitted..pos].to_string()
            } else {
                String::new()
            };
            self.emitted = pos;
            return HoldOut::Stop(emit);
        }
        let hold = prefix_hold(&self.full, stops);
        let safe = self.full.len() - hold;
        if safe < self.emitted {
            return HoldOut::Continue(String::new());
        }
        let emit = self.full[self.emitted..safe].to_string();
        self.emitted = safe;
        HoldOut::Continue(emit)
    }

    fn finish(&mut self) -> String {
        let emit = self.full[self.emitted..].to_string();
        self.emitted = self.full.len();
        emit
    }
}

fn earliest_stop(text: &str, stops: &[String]) -> Option<usize> {
    let mut found: Option<usize> = None;
    for stop in stops {
        if let Some(pos) = text.find(stop.as_str()) {
            found = Some(found.map_or(pos, |current| current.min(pos)));
        }
    }
    found
}

/// Longest suffix of `text` that is a proper prefix of some stop string.
fn prefix_hold(text: &str, stops: &[String]) -> usize {
    let mut hold = 0;
    for stop in stops {
        let max = text.len().min(stop.len().saturating_sub(1));
        for len in 1..=max {
            if !stop.is_char_boundary(len) {
                continue;
            }
            if text.len() >= len
                && text.is_char_boundary(text.len() - len)
                && text.ends_with(&stop[..len])
            {
                hold = hold.max(len);
            }
        }
    }
    hold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::patterned_model;
    use std::cell::Cell;

    fn scripted(bytes: &[u8]) -> impl FnMut(u32) -> Result<Vec<u8>, TokenizerError> + '_ {
        let cursor = Cell::new(0usize);
        move |_id| {
            let index = cursor.get();
            cursor.set(index + 1);
            let byte = bytes.get(index).copied().unwrap_or(b'?');
            Ok(vec![byte])
        }
    }

    #[test]
    fn stops_on_max_tokens() {
        let model = patterned_model(false);
        let params = GenerateParams::greedy(3);
        let out = generate(&model, &[1, 2, 3], &params, scripted(b"abcdefgh")).unwrap();
        assert_eq!(out.reason, StopReason::MaxTokens);
        assert_eq!(out.tokens.len(), 3);
        assert_eq!(out.text, "abc");
        assert_eq!(out.chunks.concat(), out.text);
        let uncached = crate::greedy_tokens(&model, &[1, 2, 3], 3).unwrap();
        assert_eq!(out.tokens, uncached);
    }

    #[test]
    fn stops_on_eos_tokens() {
        let model = patterned_model(false);
        let prompt = [4u32, 5, 6];
        let free = generate(
            &model,
            &prompt,
            &GenerateParams::greedy(6),
            scripted(b"abcdefghij"),
        )
        .unwrap();
        assert!(free.tokens.len() >= 3, "{:?}", free.tokens);

        let eos_id = free.tokens[0];
        let mut params = GenerateParams::greedy(6);
        // The id that actually matches is the second entry, so a single-id
        // check would not be enough.
        params.eos = vec![eos_id.wrapping_add(97), eos_id];
        let stopped = generate(&model, &prompt, &params, scripted(b"abcdefghij")).unwrap();
        assert_eq!(stopped.reason, StopReason::Eos);
        assert!(stopped.tokens.is_empty());
        assert!(stopped.text.is_empty());
        assert!(!stopped.tokens.contains(&eos_id));
        assert!(
            stopped
                .chunks
                .iter()
                .all(|chunk| !chunk.contains('\u{FFFD}'))
        );

        // Temperature 0 on this tiny net repeats one id, so a later EOS has to
        // be a different id. A warm seed produces one. The bytes stay scripted,
        // so the text is still the prefix before that id.
        params.temperature = 1.0;
        params.eos.clear();
        let mut picked = None;
        for seed in 0..16 {
            params.seed = seed;
            let varied = generate(&model, &prompt, &params, scripted(b"abcdefghij")).unwrap();
            let later_at = varied
                .tokens
                .iter()
                .enumerate()
                .skip(1)
                .find_map(|(index, id)| {
                    if varied.tokens[..index].contains(id) {
                        None
                    } else {
                        Some(index)
                    }
                });
            if let Some(later_at) = later_at {
                picked = Some((seed, later_at, varied.tokens));
                break;
            }
        }
        let (seed, later_at, tokens) = picked.expect("warm sample should emit a new id");
        params.seed = seed;
        let later = tokens[later_at];
        params.eos = vec![later];
        let partial = generate(&model, &prompt, &params, scripted(b"abcdefghij")).unwrap();
        assert_eq!(partial.reason, StopReason::Eos);
        assert_eq!(partial.tokens, tokens[..later_at]);
        let prefix: String = b"abcdefghij"[..later_at]
            .iter()
            .map(|byte| char::from(*byte))
            .collect();
        assert_eq!(partial.text, prefix);
        assert!(!partial.tokens.contains(&later));
        assert_eq!(partial.chunks.concat(), partial.text);
    }

    #[test]
    fn stop_string_is_excluded_even_when_it_spans_tokens() {
        let model = patterned_model(false);
        let prompt = [1u32, 2];
        // "hello world": the stop "lo " starts on the second `l` and covers
        // three byte tokens. "world" is a later alternative and must lose.
        let bytes: &'static [u8] = b"hello world";
        let mut params = GenerateParams::greedy(bytes.len());
        params.stop = vec!["world".to_string(), "lo ".to_string()];
        let out = generate(&model, &prompt, &params, scripted(bytes)).unwrap();
        assert_eq!(out.reason, StopReason::StopString);
        assert_eq!(out.text, "hel");
        assert!(!out.text.contains("lo "));
        assert!(!out.text.contains("world"));
        assert_eq!(out.chunks.concat(), out.text);
        for chunk in &out.chunks {
            assert!(!chunk.contains("lo "), "{chunk:?}");
            assert!(!chunk.contains("world"), "{chunk:?}");
        }

        // The stop is only a prefix when generation ends. Those bytes are
        // real output, not a stop, so max_tokens must release them.
        params.stop = vec!["abc".to_string()];
        params.max_tokens = 2;
        let flushed = generate(&model, &prompt, &params, scripted(b"abZZ")).unwrap();
        assert_eq!(flushed.reason, StopReason::MaxTokens);
        assert_eq!(flushed.text, "ab");

        // 🦀 is four byte tokens. Nothing before the emoji is the stop, and
        // the byte after it must not be emitted either.
        let mut emoji = b"a".to_vec();
        emoji.extend_from_slice("🦀".as_bytes());
        emoji.push(b'b');
        params.stop = vec!["🦀".to_string()];
        params.max_tokens = emoji.len();
        let cut = generate(&model, &prompt, &params, scripted(&emoji)).unwrap();
        assert_eq!(cut.reason, StopReason::StopString);
        assert_eq!(cut.text, "a");
        assert!(!cut.text.contains('🦀'));
        assert!(!cut.text.contains('b'));
        assert_eq!(cut.chunks.concat(), cut.text);
    }

    #[test]
    fn stop_hold_releases_a_suffix_that_was_not_the_stop() {
        // "aba" ends in a prefix of "abc". The next bytes are "bc", which
        // completes a stop that starts later, and the earlier "ab" stays.
        let mut held = StopHold::default();
        let stops = vec!["abc".to_string()];
        match held.push("aba", &stops) {
            HoldOut::Continue(text) => assert_eq!(text, "ab"),
            HoldOut::Stop(text) => panic!("stopped early on {text:?}"),
        }
        match held.push("bc", &stops) {
            HoldOut::Stop(text) => assert_eq!(text, ""),
            HoldOut::Continue(text) => panic!("missed the stop, emitted {text:?}"),
        }
        assert_eq!(held.full[..held.emitted], *"ab");
    }
}
