//! Rank-based byte-pair encoding, the model a `.tiktoken` file describes.
//!
//! [`crate::Tokenizer`] stores a merge list keyed by `(token_id, token_id)`.
//! cl100k has no merge list, only `bytes -> rank`. Encoding a pre-split piece
//! starts from single bytes and repeatedly merges the adjacent span whose
//! concatenated bytes have the lowest rank. Those are different lookups, so
//! this is a second type. A `rank(&left, &right)` trait would make the
//! merge-list encoder build byte strings it does not have, and it would
//! change `Tokenizer::encode`, which the existing tests pin down.

use crate::error::TokenizerError;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use std::collections::{HashMap, HashSet};
use std::fs;

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
/// is a [`Vec<Vec<u8>>`] indexed by rank.
#[derive(Debug)]
pub struct RankTokenizer {
    encoder: HashMap<Vec<u8>, u32>,
    decoder: Vec<Vec<u8>>,
    specials: HashMap<String, u32>,
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
    /// over the piece's bytes. A special token inside a longer piece is
    /// ordinary text here; splitting those out is a later step.
    pub fn encode_piece(&self, piece: &str) -> Vec<u32> {
        if let Some(&id) = self.specials.get(piece) {
            return vec![id];
        }
        self.encode_bpe(piece.as_bytes())
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
        let mut decoder = vec![Vec::new(); n];
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
            decoder[idx] = bytes.clone();
            encoder.insert(bytes, rank);
        }
        debug_assert!(decoder.iter().enumerate().all(|(idx, token)| {
            encoder.get(token.as_slice()).copied() == u32::try_from(idx).ok()
        }));

        Ok(Self {
            encoder,
            decoder,
            specials: HashMap::new(),
        })
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
        self.encoder.get(bytes).copied()
    }

    fn id_of(&self, bytes: &[u8]) -> u32 {
        self.rank_of(bytes)
            .unwrap_or_else(|| panic!("vocab is missing token bytes {bytes:?}"))
    }

    fn bpe_token(&self, id: u32) -> Option<&[u8]> {
        let idx = usize::try_from(id).ok()?;
        self.decoder.get(idx).map(Vec::as_slice)
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

#[cfg(test)]
mod tests {
    use super::{RankTokenizer, STANDARD, cl100k_specials};
    use crate::TokenizerError;
    use base64::Engine as _;
    use std::collections::{HashMap, HashSet};
    use std::error::Error;
    use std::path::PathBuf;
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
        assert_eq!(tok.decoder[0], b"!");
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
        let tok = vocab(&[(b"a", 0)]);
        assert_eq!(tok.encode_piece(""), Vec::<u32>::new());
        assert_eq!(tok.decode(&[]).unwrap(), "");
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
            let path = ensure_cl100k_file();
            RankTokenizer::load_cl100k(path.to_str().expect("cache path is utf-8"))
                .unwrap_or_else(|err| panic!("load_cl100k({}): {err}", path.display()))
        })
    }

    fn ensure_cl100k_file() -> PathBuf {
        const URL: &str =
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken";
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
            .unwrap_or_else(|err| {
                panic!("could not run curl ({err}); run scripts/fetch-cl100k.sh")
            });
        assert!(status.success(), "curl failed to download cl100k: {status}");
        assert_eq!(file_sha256(&tmp), HASH, "cl100k download hash mismatch");
        std::fs::rename(&tmp, &path).unwrap();
        path
    }

    fn cl100k_cache_path() -> PathBuf {
        let home = std::env::var("HOME").expect("HOME is set");
        PathBuf::from(home).join(".cache/tokenizer/cl100k_base.tiktoken")
    }

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
