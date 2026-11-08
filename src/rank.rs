//! Rank-based byte-pair encoding, the model a `.tiktoken` file describes.
//!
//! [`crate::Tokenizer`] stores a merge list keyed by `(token_id, token_id)`.
//! cl100k has no merge list, only `bytes -> rank`. Encoding a pre-split piece
//! starts from single bytes and repeatedly merges the adjacent span whose
//! concatenated bytes have the lowest rank. Those are different lookups, so
//! this is a second type. A `rank(&left, &right)` trait would make the
//! merge-list encoder build byte strings it does not have, and it would
//! change `Tokenizer::encode`, which the existing tests pin down.
//!
//! Full-text [`RankTokenizer::encode`] splits special tokens out first, then
//! splits each remaining segment with the cl100k regex. Pieces never merge
//! across a regex boundary.

use crate::error::TokenizerError;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use std::collections::{HashMap, HashSet};
use std::fs;
use unicode_normalization::UnicodeNormalization;

/// `cl100k_base()["pat_str"]`, copied verbatim from tiktoken
/// `tiktoken_ext/openai_public.py` at commit
/// `3591ff175d6a80efbe4fcc7f0e219ddd4b8c52f1`
/// (<https://github.com/openai/tiktoken/blob/3591ff175d6a80efbe4fcc7f0e219ddd4b8c52f1/tiktoken_ext/openai_public.py>).
///
/// The same string is what tiktoken 0.14.0 stores on `Encoding._pat_str`.
/// `\s+(?!\S)` is a negative lookahead, so the `regex` crate rejects the
/// pattern. fancy-regex is the engine that accepts it, and it is the engine
/// tiktoken's own Rust core compiles this string with.
const CL100K_PAT: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

/// Mergeable tokens in `cl100k_base.tiktoken`. Ranks are `0..CL100K_VOCAB_LEN`.
const CL100K_VOCAB_LEN: usize = 100_256;

const CL100K_SPECIALS: &[(&str, u32)] = &[
    ("<|endoftext|>", 100_257),
    ("<|fim_prefix|>", 100_258),
    ("<|fim_middle|>", 100_259),
    ("<|fim_suffix|>", 100_260),
    ("<|endofprompt|>", 100_276),
];

/// One part boundary while encoding.
///
/// `start` is an index into the piece. `next_rank` is the rank of this part
/// concatenated with the next part, or `None` when that byte string is not in
/// the vocab. `None` is not a sentinel rank: a real token can never be confused
/// with "not mergeable".
struct Part {
    start: usize,
    next_rank: Option<u32>,
}

/// Byte-level BPE whose vocab is `bytes -> rank`.
///
/// Build one with [`RankTokenizer::load_cl100k`]. Special tokens live in
/// `specials` and are matched as a whole piece. They are never produced by
/// merging. The encoder is a [`HashMap<Vec<u8>, u32>`] looked up by `&[u8]`
/// (`Vec<u8>: Borrow<[u8]>`, so a lookup does not allocate a key). The decoder
/// is a [`Vec<Vec<u8>>`] indexed by rank. `pat` is the cl100k regex, compiled
/// once when the tokenizer is built. [`RankTokenizer::encode`] borrows it.
/// It is not compiled per call.
///
/// `fancy_regex::Regex` is `Send + Sync`. tiktoken still keeps one copy per
/// thread, because the matcher has internal contention. That split is left
/// for later; one regex on the struct is enough for a single caller.
#[derive(Debug)]
pub struct RankTokenizer {
    encoder: HashMap<Vec<u8>, u32>,
    /// `Some` is the raw bytes of that id. `None` is a hole: the id was never
    /// in the file. cl100k has no holes (`0..n` is filled). Qwen's decoder is
    /// sized to the highest id, and a missing slot is [`TokenizerError::UnknownToken`]
    /// rather than an empty byte string.
    decoder: Vec<Option<Vec<u8>>>,
    specials: HashMap<String, u32>,
    pat: fancy_regex::Regex,
    /// Merge priority for Qwen. The key is the merged token's raw bytes and
    /// the value is the index in `tokenizer.json` `merges` (lower merges
    /// first). `None` on cl100k, where the encoder value is both the id and
    /// the rank. Qwen's ids are not in merge order, so the two have to be
    /// stored apart or `encode_bpe` would merge the lowest id instead of the
    /// earliest merge.
    ranks: Option<HashMap<Vec<u8>, u32>>,
    /// Apply Unicode NFC to each ordinary segment before the regex. Qwen's
    /// `tokenizer.json` normalizer is NFC. cl100k leaves this off: tiktoken
    /// does not normalize, and turning it on would change the A4 ids.
    normalize_nfc: bool,
}

impl RankTokenizer {
    /// Load `cl100k_base.tiktoken` and register the five cl100k special tokens.
    ///
    /// Each line is `base64(bytes) rank` in the standard padded alphabet.
    /// The file has [`CL100K_VOCAB_LEN`] mergeable tokens, ranks `0..n` with
    /// no gaps. A bad line is [`TokenizerError::Parse`] or
    /// [`TokenizerError::Base64`], not a panic. A missing path is
    /// [`TokenizerError::Io`].
    pub fn load_cl100k(path: &str) -> Result<Self, TokenizerError> {
        let text = fs::read_to_string(path)?;
        let mut tok = Self::parse(&text)?;
        if tok.encoder.len() != CL100K_VOCAB_LEN {
            return Err(TokenizerError::Parse {
                line: 0,
                message: format!(
                    "expected {CL100K_VOCAB_LEN} tokens, found {}",
                    tok.encoder.len()
                ),
            });
        }
        for byte in 0..=u8::MAX {
            if !tok.encoder.contains_key(std::slice::from_ref(&byte)) {
                return Err(TokenizerError::Parse {
                    line: 0,
                    message: format!("missing single-byte token {byte}"),
                });
            }
        }
        tok.specials = cl100k_specials();
        Ok(tok)
    }

    /// Encode one pre-split piece.
    ///
    /// An exact special token is that one id. Anything else is rank-based BPE
    /// over the piece's bytes. The empty piece is not a token:
    /// `encode_piece("")` returns an empty vec (tiktoken's piece encoder
    /// panics on `""`; full-text `encode("")` is also empty, because the
    /// regex yields no matches). A special token inside a longer piece is
    /// ordinary text here. [`RankTokenizer::encode`] splits those out before
    /// it calls this.
    pub fn encode_piece(&self, piece: &str) -> Vec<u32> {
        if piece.is_empty() {
            return Vec::new();
        }
        if let Some(&id) = self.specials.get(piece) {
            return vec![id];
        }
        self.encode_bpe(piece.as_bytes())
    }

    /// Encode `text`, allowing every registered special token.
    ///
    /// Specials are split out first and each becomes one id. Each remaining
    /// segment is split with [`fancy_regex::Regex::find_iter`] on the regex
    /// stored in this tokenizer, and each match is passed to
    /// [`encode_piece`](Self::encode_piece). A search error from fancy-regex
    /// is returned. This matches tiktoken `encode(text, allowed_special="all")`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.encode_splitting(text, None)
    }

    /// Encode `text`, allowing only the specials in `allowed`.
    ///
    /// A registered special that is not in `allowed` is disallowed. If one
    /// appears anywhere in `text`, this returns
    /// [`TokenizerError::DisallowedSpecial`] and does not encode the text.
    /// That is tiktoken's default `disallowed_special="all"`: the disallowed
    /// set is every registered special minus `allowed`, and a hit is an
    /// error rather than a silent byte-pair encoding. Strings in `allowed`
    /// that are not registered specials are ignored.
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        allowed: &HashSet<&str>,
    ) -> Result<Vec<u32>, TokenizerError> {
        if let Some((_, token, _)) = self.earliest_special(text, |name| !allowed.contains(name)) {
            return Err(TokenizerError::DisallowedSpecial(token.to_string()));
        }
        self.encode_splitting(text, Some(allowed))
    }

    /// Concatenate the raw bytes of each id.
    ///
    /// A rank indexes `decoder`. A cl100k special id is resolved in `specials`.
    /// Any other id is [`TokenizerError::UnknownToken`].
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token) = self.bpe_token(id) {
                bytes.extend_from_slice(token);
            } else if let Some(text) = self.special_text(id) {
                bytes.extend_from_slice(text.as_bytes());
            } else {
                return Err(TokenizerError::UnknownToken(id));
            }
        }
        Ok(bytes)
    }

    /// [`decode_bytes`](Self::decode_bytes), then UTF-8. Invalid sequences are
    /// [`TokenizerError::InvalidUtf8`].
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        String::from_utf8(self.decode_bytes(ids)?).map_err(TokenizerError::from)
    }

    fn parse(text: &str) -> Result<Self, TokenizerError> {
        let mut entries: Vec<(usize, Vec<u8>, u32)> = Vec::new();
        let mut seen_ranks = HashSet::new();
        for (idx, line) in text.lines().enumerate() {
            let line_no = idx + 1;
            if line.is_empty() {
                return Err(TokenizerError::Parse {
                    line: line_no,
                    message: "empty line".into(),
                });
            }
            let Some((token_b64, rank_str)) = line.split_once(' ') else {
                return Err(TokenizerError::Parse {
                    line: line_no,
                    message: "missing space".into(),
                });
            };
            let bytes = STANDARD
                .decode(token_b64)
                .map_err(|source| TokenizerError::Base64 {
                    line: line_no,
                    source,
                })?;
            let rank: u32 = rank_str.parse().map_err(|_| TokenizerError::Parse {
                line: line_no,
                message: format!("bad rank {rank_str:?}"),
            })?;
            if !seen_ranks.insert(rank) {
                return Err(TokenizerError::Parse {
                    line: line_no,
                    message: format!("duplicate rank {rank}"),
                });
            }
            entries.push((line_no, bytes, rank));
        }

        let n = entries.len();
        let mut encoder = HashMap::with_capacity(n);
        let mut decoder = vec![None; n];
        for (line_no, bytes, rank) in entries {
            // Unique ranks, each `< n`, are exactly `0..n`. A hole shows up
            // as some rank that does not fit.
            let Some(idx) = usize::try_from(rank).ok().filter(|&i| i < n) else {
                return Err(TokenizerError::Parse {
                    line: line_no,
                    message: format!("rank {rank} is outside 0..{n}"),
                });
            };
            if encoder.contains_key(bytes.as_slice()) {
                return Err(TokenizerError::Parse {
                    line: line_no,
                    message: "duplicate token".into(),
                });
            }
            // The base64 decoder owns `bytes`. The map and the vec both need
            // those bytes, so one of them keeps the buffer and the other
            // takes a clone. The file `String` is not stored.
            decoder[idx] = Some(bytes.clone());
            encoder.insert(bytes, rank);
        }
        debug_assert!(decoder.iter().enumerate().all(|(idx, token)| {
            let Some(token) = token else {
                return false;
            };
            encoder.get(token.as_slice()).copied() == u32::try_from(idx).ok()
        }));

        Ok(Self {
            encoder,
            decoder,
            specials: HashMap::new(),
            // Compiled once per tokenizer, then borrowed by every `encode`.
            // `?` uses `From<fancy_regex::Error>`.
            pat: fancy_regex::Regex::new(CL100K_PAT)?,
            ranks: None,
            normalize_nfc: false,
        })
    }

    /// Load a Hugging Face `tokenizer.json` for Qwen2.5 (byte-level BPE).
    ///
    /// The file's merges and vocab are unmapped through GPT-2
    /// `bytes_to_unicode`. Added tokens are matched as literals before the
    /// pre-tokenizer regex. See [`crate::qwen`] for the format this accepts.
    pub fn load_qwen(path: &str) -> Result<Self, TokenizerError> {
        crate::qwen::parse_qwen_file(path)
    }

    /// Assemble a Qwen tokenizer from parts the JSON loader already checked.
    ///
    /// `ranks` is merge priority (lower first), keyed by the merged token's
    /// bytes. `encoder` is bytes to id, including the single-byte tokens and
    /// every merge result. `decoder[id]` is that id's bytes. `pattern` is the
    /// pre-tokenizer regex copied out of the file, compiled once here.
    pub(crate) fn from_qwen_parts(
        encoder: HashMap<Vec<u8>, u32>,
        decoder: Vec<Option<Vec<u8>>>,
        ranks: HashMap<Vec<u8>, u32>,
        specials: HashMap<String, u32>,
        pattern: &str,
    ) -> Result<Self, TokenizerError> {
        Ok(Self {
            encoder,
            decoder,
            specials,
            pat: fancy_regex::Regex::new(pattern)?,
            ranks: Some(ranks),
            normalize_nfc: true,
        })
    }

    /// Leftmost registered special for which `keep` is true.
    ///
    /// The returned `&str` borrows the map entry, not `text`, so the caller
    /// can copy out the start and the length and then drop the borrow before
    /// calling another `&self` method that needs the map. Two specials that
    /// start at the same byte are decided by length, longest first. That tie
    /// never fires for cl100k: its five specials are not prefixes of one
    /// another. See the comment on `str::find` below.
    fn earliest_special<'a>(
        &'a self,
        text: &str,
        mut keep: impl FnMut(&str) -> bool,
    ) -> Option<(usize, &'a str, u32)> {
        let mut best: Option<(usize, &'a str, u32)> = None;
        for (token, &id) in &self.specials {
            if token.is_empty() || !keep(token) {
                continue;
            }
            // `str::find` searches each special on its own, then this loop
            // keeps the leftmost hit and, on the same byte, the longer token.
            // That is only equivalent to tiktoken while the specials are not
            // prefixes of one another: two of them never match at the same
            // index, so the longest-wins tie is dead code for this vocab.
            // Specials that share a prefix need an alternation (one regex of
            // the escaped tokens) or a HashMap-style match (try the token map
            // at the candidate offset). tiktoken's collision order is that
            // match, not "longest string wins".
            let Some(pos) = text.find(token.as_str()) else {
                continue;
            };
            let better = match best {
                Some((best_pos, best_tok, _)) => {
                    pos < best_pos || (pos == best_pos && token.len() > best_tok.len())
                }
                None => true,
            };
            if better {
                best = Some((pos, token.as_str(), id));
            }
        }
        best
    }

    fn encode_splitting(
        &self,
        text: &str,
        allowed: Option<&HashSet<&str>>,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        let mut rest = text;
        loop {
            // Copy start/len/id out of the map borrow before `push_ordinary`
            // borrows `self` again. `token` is `&str` into `self.specials`.
            let next = self
                .earliest_special(rest, |name| match allowed {
                    Some(allowed) => allowed.contains(name),
                    None => true,
                })
                .map(|(start, token, id)| (start, token.len(), id));
            let Some((start, len, id)) = next else {
                self.push_ordinary(rest, &mut ids)?;
                break;
            };
            self.push_ordinary(&rest[..start], &mut ids)?;
            ids.push(id);
            rest = &rest[start + len..];
        }
        Ok(ids)
    }

    /// Regex-split `text` and append one `encode_piece` per match.
    ///
    /// `find_iter` yields `Result<Match<'t>, Error>`. `'t` is the lifetime of
    /// `text`: `Match` stores that `&str`, and `as_str` returns a subslice of
    /// it, not a borrow of the `Match` temporary. The collected `&str`s are
    /// valid for the rest of this call and no longer. `collect::<Result<..>>`
    /// turns the iterator of results into one `Result`, and `?` converts the
    /// fancy-regex error.
    fn push_ordinary(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokenizerError> {
        if self.normalize_nfc {
            // The collected `String` has to live for the pieces borrowed from
            // it. `text` here is the gap between specials, not the original
            // input: specials with `normalized: false` were already split out,
            // which is what Qwen's added tokens ask for.
            let text = text.nfc().collect::<String>();
            self.push_pieces(&text, ids)
        } else {
            self.push_pieces(text, ids)
        }
    }

    fn push_pieces(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), TokenizerError> {
        let pieces = if self.ranks.is_some() {
            // Qwen's Split behavior is `Isolated`: a gap the regex does not
            // cover is still a piece. cl100k's pattern is written to cover
            // every byte, and the existing encode only keeps matches.
            self.pieces_keeping_gaps(text)?
        } else {
            self.pat
                .find_iter(text)
                .map(|mat| mat.map(|mat| mat.as_str()))
                .collect::<Result<Vec<_>, _>>()?
        };
        debug_assert_eq!(
            pieces.concat(),
            text,
            "pre-tokenizer left a gap or overlap in {text:?}"
        );
        for piece in pieces {
            ids.extend(self.encode_piece(piece));
        }
        Ok(())
    }

    /// Regex matches plus the unmatched slices between them.
    ///
    /// `Match::start` and `end` are byte offsets into `text`. fancy-regex
    /// returns them on char boundaries. An empty match would not advance
    /// `last` and the loop would spin, so that is an error instead.
    fn pieces_keeping_gaps<'a>(&self, text: &'a str) -> Result<Vec<&'a str>, TokenizerError> {
        let mut pieces = Vec::new();
        let mut last = 0usize;
        for mat in self.pat.find_iter(text) {
            let mat = mat?;
            let start = mat.start();
            let end = mat.end();
            if start < last || end < start {
                return Err(TokenizerError::Parse {
                    line: 0,
                    message: "pre-tokenizer match went backwards".into(),
                });
            }
            if start > last {
                pieces.push(&text[last..start]);
            }
            if end == start {
                return Err(TokenizerError::Parse {
                    line: 0,
                    message: "pre-tokenizer produced an empty match".into(),
                });
            }
            pieces.push(&text[start..end]);
            last = end;
        }
        if last < text.len() {
            pieces.push(&text[last..]);
        }
        Ok(pieces)
    }

    /// Byte-pair encode with no special-token match.
    ///
    /// Parts are a `Vec` of boundaries. Each merge rescans that vec for the
    /// lowest rank. Pieces are short (a regex chunk, or one vocab token), so
    /// the scan stays in cache. A heap would be less code-local and not
    /// faster at these lengths.
    fn encode_bpe(&self, piece: &[u8]) -> Vec<u32> {
        if piece.is_empty() {
            return Vec::new();
        }
        if piece.len() == 1 {
            return vec![self.id_of(piece)];
        }

        let mut parts = Vec::with_capacity(piece.len() + 1);
        let mut best: Option<(u32, usize)> = None;
        for i in 0..piece.len() - 1 {
            let rank = self.rank_of(&piece[i..i + 2]);
            consider(&mut best, rank, parts.len());
            parts.push(Part {
                start: i,
                next_rank: rank,
            });
        }
        parts.push(Part {
            start: piece.len() - 1,
            next_rank: None,
        });
        parts.push(Part {
            start: piece.len(),
            next_rank: None,
        });

        while let Some((_, index)) = best {
            // Refresh the two pairs the merge touches, then drop the right
            // half. `rank_after_merge` reads `parts[index + 3]` and must run
            // before the removal: the element at `index + 1` is still there,
            // so `+ 3` skips it.
            if index > 0 {
                parts[index - 1].next_rank = self.rank_after_merge(&parts, piece, index - 1);
            }
            parts[index].next_rank = self.rank_after_merge(&parts, piece, index);
            parts.remove(index + 1);

            best = None;
            for (i, part) in parts.iter().enumerate().take(parts.len() - 1) {
                consider(&mut best, part.next_rank, i);
            }
        }

        debug_assert!(parts.len() >= 2);
        let mut ids = Vec::with_capacity(parts.len() - 1);
        for window in parts.windows(2) {
            ids.push(self.id_of(&piece[window[0].start..window[1].start]));
        }
        ids
    }

    /// Rank of `piece[parts[index].start .. parts[index + 3].start]`.
    ///
    /// Called before `parts.remove(merge_at + 1)`. The `+ 3` steps over the
    /// boundary that is about to disappear, so the slice is the merged part
    /// plus the part that will sit beside it afterwards.
    fn rank_after_merge(&self, parts: &[Part], piece: &[u8], index: usize) -> Option<u32> {
        let end_at = index.checked_add(3)?;
        if end_at >= parts.len() {
            return None;
        }
        let start = parts[index].start;
        let end = parts[end_at].start;
        // `piece[start..end]` has type `[u8]` (the `Index` output). Borrowing
        // it gives `&[u8]`, which is what `rank_of` takes and what
        // `HashMap<Vec<u8>, _>::get` accepts: the argument is `&Q` with
        // `Q = [u8]`, and `Vec<u8>: Borrow<[u8]>`. Passing `&bytes` when
        // `bytes` is already `&[u8]` would set `Q = &[u8]` and fail E0277,
        // because `Vec<u8>` does not implement `Borrow<&[u8]>`.
        self.rank_of(&piece[start..end])
    }

    fn rank_of(&self, bytes: &[u8]) -> Option<u32> {
        match &self.ranks {
            Some(ranks) => ranks.get(bytes).copied(),
            None => self.encoder.get(bytes).copied(),
        }
    }

    fn id_of(&self, bytes: &[u8]) -> u32 {
        // The id is always the encoder value. cl100k stores the rank in that
        // same slot, so `rank_of` and this agree there. Qwen does not: the
        // rank map is merge order, and calling `rank_of` here would emit the
        // merge index (0 for the first merge) instead of the vocab id.
        self.encoder
            .get(bytes)
            .copied()
            .unwrap_or_else(|| panic!("vocab is missing token bytes {bytes:?}"))
    }

    fn bpe_token(&self, id: u32) -> Option<&[u8]> {
        let idx = usize::try_from(id).ok()?;
        self.decoder.get(idx).and_then(|slot| slot.as_deref())
    }

    fn special_text(&self, id: u32) -> Option<&str> {
        self.specials
            .iter()
            .find(|&(_, &rank)| rank == id)
            .map(|(text, _)| text.as_str())
    }
}

fn consider(best: &mut Option<(u32, usize)>, rank: Option<u32>, index: usize) {
    let Some(rank) = rank else {
        return;
    };
    // Strict `<` keeps the leftmost pair when two spans share a rank.
    match best {
        Some((best_rank, _)) if rank < *best_rank => *best = Some((rank, index)),
        None => *best = Some((rank, index)),
        _ => {}
    }
}

fn cl100k_specials() -> HashMap<String, u32> {
    CL100K_SPECIALS
        .iter()
        .map(|(text, id)| (text.to_string(), *id))
        .collect()
}

/// Download `cl100k_base.tiktoken` into `~/.cache` when the tests need it.
///
/// The file is not in git. A cached file is used only when its sha256 matches
/// the hash tiktoken publishes.
#[cfg(test)]
pub(crate) fn ensure_cl100k_file() -> std::path::PathBuf {
    const URL: &str = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken";
    const HASH: &str = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";
    let path = cl100k_cache_path();
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
        .unwrap_or_else(|err| panic!("could not run curl ({err}); run scripts/fetch-cl100k.sh"));
    assert!(status.success(), "curl failed to download cl100k: {status}");
    assert_eq!(file_sha256(&tmp), HASH, "cl100k download hash mismatch");
    std::fs::rename(&tmp, &path).unwrap();
    path
}

#[cfg(test)]
fn cl100k_cache_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").expect("HOME is set");
    std::path::PathBuf::from(home).join(".cache/tokenizer/cl100k_base.tiktoken")
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
    use super::{CL100K_PAT, RankTokenizer, STANDARD, cl100k_specials};
    use crate::TokenizerError;
    use base64::Engine as _;
    use std::collections::{HashMap, HashSet};
    use std::error::Error;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    fn vocab(tokens: &[(&[u8], u32)]) -> RankTokenizer {
        let mut text = String::new();
        for (bytes, rank) in tokens {
            text.push_str(&STANDARD.encode(bytes));
            text.push(' ');
            text.push_str(&rank.to_string());
            text.push('\n');
        }
        RankTokenizer::parse(&text).unwrap()
    }

    #[test]
    fn parses_a_real_file_line() {
        // First line of cl100k_base.tiktoken. "!" is IQ== in standard base64.
        let tok = RankTokenizer::parse("IQ== 0\n").unwrap();
        assert_eq!(tok.decoder[0].as_deref(), Some(b"!".as_slice()));
        assert_eq!(tok.encode_piece("!"), vec![0]);
    }

    #[test]
    fn parses_crlf_and_a_trailing_newline() {
        let tok = RankTokenizer::parse("YQ== 0\r\nYg== 1\r\n").unwrap();
        assert_eq!(tok.encode_piece("ab"), vec![0, 1]);
        assert_eq!(tok.decode(&[0, 1]).unwrap(), "ab");
    }

    #[test]
    fn empty_piece_encodes_to_nothing() {
        // Pins encode_piece("") == []. tiktoken's `_encode_single_piece("")`
        // panics. Full-text encode("") is empty too: the regex yields nothing.
        let tok = vocab(&[(b"a", 0)]);
        assert_eq!(tok.encode_piece(""), Vec::<u32>::new());
        assert!(tok.encode_piece("").is_empty());
        assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
        assert_eq!(
            tok.encode_with_special_tokens("", &HashSet::new()).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(tok.decode(&[]).unwrap(), "");
    }

    #[test]
    fn cl100k_pattern_is_stored_once_on_the_tokenizer() {
        let tok = vocab(&[(b"a", 0)]);
        assert_eq!(tok.pat.as_str(), CL100K_PAT);
        assert!(CL100K_PAT.contains(r"\s+(?!\S)"));
        assert!(tok.pat.as_str().contains(r"(?i:[sdmt]|ll|ve|re)"));
        // A second encode borrows the same compiled regex. `as_str` is the
        // pattern the constructor compiled, not a fresh `Regex::new`.
        assert_eq!(tok.encode("a").unwrap(), vec![0]);
        assert_eq!(tok.pat.as_str(), CL100K_PAT);
    }

    #[test]
    fn regex_split_stops_a_merge_across_the_digit_boundary() {
        // `\p{N}{1,3}` cuts "12345" into "123" and "45". "234" crosses that
        // cut. As one piece it is the lowest-rank merge. After the split it
        // is not a candidate, so encode and encode_piece disagree.
        // Ranks are dense `0..n`. "234" is the lowest, and it crosses the
        // regex cut between "123" and "45".
        let tok = vocab(&[
            (b"234", 0),
            (b"23", 1),
            (b"34", 2),
            (b"123", 3),
            (b"45", 4),
            (b"12", 5),
            (b"1", 6),
            (b"2", 7),
            (b"3", 8),
            (b"4", 9),
            (b"5", 10),
        ]);
        assert_eq!(tok.encode_piece("12345"), vec![6, 0, 10]);
        assert_eq!(tok.encode("12345").unwrap(), vec![3, 4]);
        assert_eq!(tok.decode(&[3, 4]).unwrap(), "12345");
    }

    #[test]
    fn disallowed_special_is_an_error_not_bpe() {
        let mut tok = byte_vocab("hello <|endoftext|> <|fim_prefix|>");
        tok.specials = cl100k_specials();

        let err = tok
            .encode_with_special_tokens("hello <|endoftext|>", &HashSet::new())
            .unwrap_err();
        assert!(
            matches!(err, TokenizerError::DisallowedSpecial(ref token) if token == "<|endoftext|>"),
            "{err}"
        );
        // The BPE of the same string is not what the error path returns.
        let bpe = tok.encode_bpe("hello <|endoftext|>".as_bytes());
        assert!(bpe.len() > 1, "{bpe:?}");
        assert!(!bpe.contains(&100_257), "{bpe:?}");

        let only_eot = HashSet::from(["<|endoftext|>"]);
        let ids = tok
            .encode_with_special_tokens("hello <|endoftext|>", &only_eot)
            .unwrap();
        assert_eq!(*ids.last().unwrap(), 100_257);
        assert_eq!(tok.decode(&ids).unwrap(), "hello <|endoftext|>");
        assert_eq!(tok.encode("hello <|endoftext|>").unwrap(), ids);

        // An earlier allowed special does not hide a later disallowed one.
        let err = tok
            .encode_with_special_tokens("<|endoftext|> then <|fim_prefix|>", &only_eot)
            .unwrap_err();
        assert!(
            matches!(err, TokenizerError::DisallowedSpecial(ref token) if token == "<|fim_prefix|>"),
            "{err}"
        );

        let both = HashSet::from(["<|endoftext|>", "<|fim_prefix|>"]);
        let ids = tok
            .encode_with_special_tokens("<|endoftext|><|fim_prefix|>", &both)
            .unwrap();
        assert_eq!(ids, vec![100_257, 100_258]);
        assert_eq!(tok.decode(&ids).unwrap(), "<|endoftext|><|fim_prefix|>");

        // A string that is not a registered special is ordinary text.
        let allowed_unknown = HashSet::from(["not-a-special"]);
        let plain = tok
            .encode_with_special_tokens("hello", &allowed_unknown)
            .unwrap();
        assert_eq!(plain, tok.encode("hello").unwrap());
        assert_eq!(tok.decode(&plain).unwrap(), "hello");
    }

    #[test]
    fn single_byte_is_its_rank_not_the_byte_value() {
        // cl100k does not use the byte value as the id. 'a' is 0x61, rank 0 here.
        let tok = vocab(&[(b"a", 0)]);
        assert_eq!(tok.encode_piece("a"), vec![0]);
    }

    #[test]
    fn merges_the_lowest_rank_not_the_leftmost_pair() {
        let tok = vocab(&[(b"a", 0), (b"b", 1), (b"c", 2), (b"bc", 3), (b"ab", 4)]);
        assert_eq!(tok.encode_piece("abc"), vec![0, 3]);
    }

    #[test]
    fn merges_again_when_the_concatenation_is_in_the_vocab() {
        let tok = vocab(&[
            (b"a", 0),
            (b"b", 1),
            (b"c", 2),
            (b"abc", 3),
            (b"ab", 4),
            (b"bc", 5),
        ]);
        assert_eq!(tok.encode_piece("abc"), vec![3]);
    }

    #[test]
    fn stops_when_no_adjacent_concatenation_is_in_the_vocab() {
        let tok = vocab(&[(b"a", 0), (b"b", 1), (b"ab", 2)]);
        assert_eq!(tok.encode_piece("ba"), vec![1, 0]);
    }

    #[test]
    fn whole_piece_in_the_vocab_is_not_a_shortcut() {
        // "abc" has a rank, but neither pair does, so nothing merges.
        let tok = vocab(&[(b"a", 0), (b"b", 1), (b"c", 2), (b"abc", 3)]);
        assert_eq!(tok.encode_piece("abc"), vec![0, 1, 2]);
    }

    #[test]
    fn tie_on_rank_keeps_the_leftmost_pair() {
        let tok = vocab(&[(b"a", 0), (b"aa", 1)]);
        assert_eq!(tok.encode_piece("aaa"), vec![1, 0]);
    }

    #[test]
    fn overlapping_run_collapses_when_the_longer_token_ranks_lower() {
        let tok = vocab(&[(b"a", 0), (b"aaa", 1), (b"aa", 2)]);
        assert_eq!(tok.encode_piece("aaa"), vec![1]);
        assert_eq!(tok.decode_bytes(&[1]).unwrap(), b"aaa");
    }

    #[test]
    fn lookup_uses_the_slice_already_in_hand() {
        let mut map: HashMap<Vec<u8>, u32> = HashMap::new();
        map.insert(b"ab".to_vec(), 7);
        let piece: &[u8] = b"ab";
        assert_eq!(map.get(piece).copied(), Some(7));
    }

    #[test]
    fn decode_of_unknown_id_is_an_error() {
        let tok = vocab(&[(b"a", 0)]);
        let err = tok.decode(&[99]).unwrap_err();
        assert!(matches!(err, TokenizerError::UnknownToken(99)));
        assert_eq!(err.to_string(), "unknown token id 99");
    }

    #[test]
    fn decode_of_invalid_utf8_is_invalid_utf8() {
        let tok = vocab(&[(b"\xff", 0)]);
        assert_eq!(tok.decode_bytes(&[0]).unwrap(), vec![0xff]);
        assert!(matches!(
            tok.decode(&[0]).unwrap_err(),
            TokenizerError::InvalidUtf8(_)
        ));
    }

    #[test]
    fn special_is_a_literal_match_and_not_bpe() {
        let mut tok = byte_vocab("<|endoftext|>!");
        let bpe = tok.encode_piece("<|endoftext|>");
        assert!(bpe.len() > 1, "{bpe:?}");
        assert_ne!(bpe, vec![100_257]);

        tok.specials = cl100k_specials();
        assert_eq!(tok.encode_piece("<|endoftext|>"), vec![100_257]);
        assert_eq!(tok.decode(&[100_257]).unwrap(), "<|endoftext|>");
        // A longer piece is not the literal, so it still goes through BPE.
        let longer = tok.encode_piece("<|endoftext|>!");
        assert!(longer.len() > 1, "{longer:?}");
        assert_ne!(longer, vec![100_257]);
        // 100256 is the hole between the mergeable vocab and the specials.
        let err = tok.decode(&[100_256]).unwrap_err();
        assert!(matches!(err, TokenizerError::UnknownToken(100_256)));
    }

    fn byte_vocab(text: &str) -> RankTokenizer {
        let mut seen = HashSet::new();
        let mut lines = String::new();
        let mut rank = 0u32;
        for &byte in text.as_bytes() {
            if seen.insert(byte) {
                lines.push_str(&STANDARD.encode([byte]));
                lines.push(' ');
                lines.push_str(&rank.to_string());
                lines.push('\n');
                rank += 1;
            }
        }
        RankTokenizer::parse(&lines).unwrap()
    }

    #[test]
    fn missing_space_is_a_parse_error() {
        let err = RankTokenizer::parse("YQ== 0\nYg==\n").unwrap_err();
        assert!(matches!(
            err,
            TokenizerError::Parse {
                line: 2,
                ref message
            } if message == "missing space"
        ));
        assert!(err.source().is_none());
    }

    #[test]
    fn empty_line_is_a_parse_error() {
        let err = RankTokenizer::parse("YQ== 0\n\n").unwrap_err();
        assert!(matches!(
            err,
            TokenizerError::Parse { line: 2, ref message } if message == "empty line"
        ));
    }

    #[test]
    fn bad_base64_is_a_base64_error() {
        let err = RankTokenizer::parse("!!!! 0\n").unwrap_err();
        assert!(matches!(err, TokenizerError::Base64 { line: 1, .. }));
        assert!(err.source().is_some());
        assert!(err.to_string().contains("line 1"), "{err}");
    }

    #[test]
    fn bad_rank_is_a_parse_error() {
        for line in ["YQ== nope", "YQ== -1", "YQ== 1.5", "YQ== 999999999999"] {
            let err = RankTokenizer::parse(line).unwrap_err();
            assert!(
                matches!(err, TokenizerError::Parse { line: 1, ref message } if message.starts_with("bad rank")),
                "{line}: {err}"
            );
        }
    }

    #[test]
    fn duplicate_rank_is_a_parse_error() {
        let err = RankTokenizer::parse("YQ== 0\nYg== 0\n").unwrap_err();
        assert!(matches!(
            err,
            TokenizerError::Parse { line: 2, ref message } if message == "duplicate rank 0"
        ));
    }

    #[test]
    fn duplicate_token_is_a_parse_error() {
        let err = RankTokenizer::parse("YQ== 0\nYQ== 1\n").unwrap_err();
        assert!(matches!(
            err,
            TokenizerError::Parse { line: 2, ref message } if message == "duplicate token"
        ));
    }

    #[test]
    fn a_rank_gap_is_a_parse_error() {
        let err = RankTokenizer::parse("YQ== 0\nYg== 2\n").unwrap_err();
        assert!(matches!(
            err,
            TokenizerError::Parse { line: 2, ref message } if message.contains("outside")
        ));
        assert!(err.to_string().contains("rank 2"), "{err}");
    }

    #[test]
    fn load_cl100k_missing_file_is_an_io_error() {
        let err = RankTokenizer::load_cl100k("no-such-cl100k.tiktoken").unwrap_err();
        assert!(matches!(err, TokenizerError::Io(_)));
    }

    #[test]
    fn load_cl100k_rejects_the_wrong_token_count() {
        let path = std::env::temp_dir().join(format!(
            "tokenizer-a3-count-{}.tiktoken",
            std::process::id()
        ));
        std::fs::write(&path, "YQ== 0\nYg== 1\n").unwrap();
        let err = RankTokenizer::load_cl100k(path.to_str().unwrap()).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(err, TokenizerError::Parse { line: 0, .. }),
            "{err}"
        );
        assert!(err.to_string().contains("100256"), "{err}");
    }

    /// Pairs from `scripts/gen_cl100k_pieces.py` (tiktoken 0.14.0).
    /// Ordinary pieces are `Encoding._encode_single_piece` (no regex).
    /// Comments mark strings whose regex `encode()` differs.
    const PIECES: &[(&str, &[u32])] = &[
        ("hello", &[15339]),
        (" world", &[1917]),
        ("🦀", &[9468, 99, 222]),
        ("   ", &[262]),
        ("\n\n", &[271]),
        ("\t", &[197]),
        ("\r\n", &[319]),
        ("12345", &[4513, 1774]),
        ("999999", &[1484, 1484, 1484]), // encode() with the regex is [5500, 5500]
        ("0", &[15]),
        ("café", &[936, 59958]),
        ("日本語", &[9080, 22656, 45918, 252]),
        ("a", &[64]),
        ("  hello", &[256, 15339]), // encode() with the regex is [220, 24748]
        ("🙂", &[9468, 19044]),
        ("é", &[978]),
        ("Привет", &[54745, 28089, 8341]),
        ("你好", &[57668, 53901]),
        ("tokenization", &[5963, 2065]),
        ("\n", &[198]),
        ("aaaaaaaa", &[70540]),
        ("...", &[1131]),
        (
            "👨‍👩‍👧‍👦",
            &[
                9468, 239, 101, 378, 235, 9468, 239, 102, 378, 235, 9468, 239, 100, 378, 235, 9468,
                239, 99,
            ],
        ),
        (" ", &[220]),
        ("hello world", &[15339, 1917]),
        ("ABC", &[26484]),
        ("\t\t\t", &[573]),
        ("café déjà", &[936, 59958, 46939]),
    ];

    const SPECIALS: &[(&str, &[u32])] = &[
        ("<|endoftext|>", &[100257]),
        ("<|fim_prefix|>", &[100258]),
        ("<|fim_middle|>", &[100259]),
        ("<|fim_suffix|>", &[100260]),
        ("<|endofprompt|>", &[100276]),
    ];

    const SPECIALS_AS_BPE: &[(&str, &[u32])] = &[
        ("<|endoftext|>", &[27, 91, 8862, 728, 428, 91, 29]),
        ("<|fim_prefix|>", &[27, 91, 69, 318, 14301, 91, 29]),
        ("<|fim_middle|>", &[27, 91, 69, 318, 63680, 91, 29]),
        ("<|fim_suffix|>", &[27, 91, 69, 318, 38251, 91, 29]),
        ("<|endofprompt|>", &[27, 91, 408, 1073, 41681, 91, 29]),
    ];

    fn loaded_cl100k() -> &'static RankTokenizer {
        static MODEL: OnceLock<RankTokenizer> = OnceLock::new();
        MODEL.get_or_init(|| {
            let path = super::ensure_cl100k_file();
            RankTokenizer::load_cl100k(path.to_str().expect("cache path is utf-8"))
                .unwrap_or_else(|err| panic!("load_cl100k({}): {err}", path.display()))
        })
    }

    #[test]
    #[ignore = "needs cl100k_base.tiktoken in ~/.cache; curl downloads it, or run scripts/fetch-cl100k.sh"]
    fn encode_piece_matches_known_ids() {
        let tok = loaded_cl100k();
        assert_eq!(tok.encoder.len(), 100_256);
        assert_eq!(tok.decoder.len(), 100_256);
        assert_eq!(tok.specials.len(), 5);
        for (piece, ids) in PIECES {
            assert_eq!(tok.encode_piece(piece), *ids, "{piece:?}");
        }
        // Longer than 100 bytes. tiktoken's own Rust encoder switches to a
        // heap at that length; these ids are still `_encode_single_piece`
        // from scripts/gen_cl100k_pieces.py (tiktoken 0.14.0).
        let repeated_a = "a".repeat(120);
        assert_eq!(tok.encode_piece(&repeated_a), vec![70540; 15]);
        let sentence = "The quick brown fox jumps over the lazy dog. ".repeat(4);
        assert_eq!(
            tok.encode_piece(&sentence),
            [
                791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13, 578, 4062, 14198, 39935,
                35308, 927, 279, 16053, 5679, 13, 578, 4062, 14198, 39935, 35308, 927, 279, 16053,
                5679, 13, 578, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13, 220,
            ]
        );
        for (piece, ids) in SPECIALS {
            assert_eq!(tok.encode_piece(piece), *ids, "{piece:?}");
            assert_eq!(tok.decode(ids).unwrap(), *piece);
        }
        for (i, (piece, bpe_ids)) in SPECIALS_AS_BPE.iter().enumerate() {
            assert_eq!(tok.encode_bpe(piece.as_bytes()), *bpe_ids, "{piece:?}");
            assert_ne!(*bpe_ids, SPECIALS[i].1);
        }
    }

    #[test]
    #[ignore = "round-trips every cl100k token; needs the cached vocab and is timed in --release"]
    fn decode_encode_piece_round_trips_every_vocab_entry() {
        let tok = loaded_cl100k();
        let start = Instant::now();
        for (bytes, &rank) in &tok.encoder {
            let ids = tok.encode_bpe(bytes);
            assert!(
                ids.len() == 1 && ids[0] == rank,
                "token {bytes:?} rank {rank} encoded as {ids:?}"
            );
            assert_eq!(tok.decode_bytes(&ids).unwrap(), *bytes);
        }
        let elapsed = start.elapsed();
        eprintln!(
            "cl100k vocab round trip {elapsed:?} entries {}",
            tok.encoder.len()
        );
        if !cfg!(debug_assertions) {
            assert!(
                elapsed < Duration::from_secs(1),
                "round trip took {elapsed:?}, budget is 1s"
            );
        }
    }
}
