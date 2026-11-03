mod codec;
mod error;
mod merge;
mod rank;
mod train;

#[cfg(test)]
mod diff_fuzz;
#[cfg(test)]
mod parity;

pub use error::TokenizerError;
pub use rank::RankTokenizer;
pub use train::counter::read_text;

use train::training::{Merges, Vocab};

/// Byte-level BPE tokenizer trained from text.
///
/// Merges are keyed by token-id pairs. OpenAI's cl100k vocab has no merge
/// list; load that with [`RankTokenizer`] and encode full text with
/// [`RankTokenizer::encode`]. The fields are private so the vocab
/// can move from a `HashMap` to a `Vec` indexed by id without changing callers.
/// Build one with [`Tokenizer::train`]. There is no [`Default`]: `train` is
/// what seeds the 256 single-byte tokens.
pub struct Tokenizer {
    merges: Merges,
    vocab: Vocab,
}

impl Tokenizer {
    /// Learn up to `num_merges` pair rules from `text`.
    ///
    /// Stops early when every remaining pair occurs once. The same text and
    /// limit always produce the same rules: ties go to the lower pair.
    pub fn train(text: &str, num_merges: u32) -> Self {
        let (merges, vocab) = train::training::train(text, num_merges);
        Self { merges, vocab }
    }

    /// How many pair rules were learned. Less than `num_merges` means training
    /// stopped because no pair occurred twice.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        codec::encode::encode(text, &self.merges)
    }

    /// Concatenate the raw bytes of each id.
    ///
    /// One token can be only part of a UTF-8 character, so callers that stream
    /// output need the bytes and have to buffer an incomplete sequence
    /// themselves. An id that was never learned is [`TokenizerError::UnknownToken`].
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        codec::decode::decode_bytes(ids, &self.vocab)
    }

    /// [`decode_bytes`](Self::decode_bytes), then UTF-8. Invalid sequences are
    /// [`TokenizerError::InvalidUtf8`], not a panic and not a replacement character.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        codec::decode::decode(ids, &self.vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_unknown_id_is_error() {
        let tok = Tokenizer::train("abc", 10);
        let err = tok.decode(&[999_999]).unwrap_err();
        assert!(matches!(err, TokenizerError::UnknownToken(999_999)));

        // A valid id in front of a bad one still fails, and does not panic.
        let err = tok.decode_bytes(&[97, 999_999]).unwrap_err();
        assert!(matches!(err, TokenizerError::UnknownToken(999_999)));
        assert_eq!(err.to_string(), "unknown token id 999999");
    }

    #[test]
    fn tokenizer_round_trip_matches_wikipedia_example() {
        let text = "aaabdaaabac";
        let tok = Tokenizer::train(text, 1000);
        assert_eq!(tok.merge_count(), 3);
        let ids = tok.encode(text);
        assert_eq!(ids, vec![258, 100, 258, 97, 99]);
        assert_eq!(tok.decode_bytes(&ids).unwrap(), text.as_bytes());
        assert_eq!(tok.decode(&ids).unwrap(), text);
    }

    #[test]
    fn decode_of_split_utf8_is_invalid_utf8() {
        let tok = Tokenizer::train("", 0);
        // 195 is the first byte of "é" (UTF-8 c3 a9) with the continuation missing.
        let err = tok.decode(&[195]).unwrap_err();
        assert!(matches!(err, TokenizerError::InvalidUtf8(_)));
        assert_eq!(tok.decode_bytes(&[195]).unwrap(), vec![195]);
    }
}
