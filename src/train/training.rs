use super::fast;
pub use super::{Merges, Vocab};

pub fn train(text: &str, num_merges: u32) -> (Merges, Vocab) {
    fast::train(text, num_merges)
}

/// The A1 trainer: recount every pair, then copy the sequence into a new `Vec`.
/// Kept as the oracle for [`tests::fast_training_matches_naive`].
#[cfg(test)]
pub fn train_naive(text: &str, num_merges: u32) -> (Merges, Vocab) {
    use super::counter;
    use crate::merge;

    let bytes = text.as_bytes();
    let mut seq: Vec<u32> = bytes.iter().map(|x| u32::from(*x)).collect();
    let mut merges: Merges = std::collections::HashMap::new();
    let mut vocab: Vocab = std::collections::HashMap::new();
    for b in 0..256 {
        vocab.insert(b, vec![b as u8]);
    }

    for i in 0..num_merges {
        let counts = counter::counts(&seq);
        let top = merge::top_pair(&counts);
        let id = 256 + i;
        match top {
            Some(((a, b), count)) if count >= 2 => {
                seq = merge::merge(&seq, (a, b), id);
                merges.insert((a, b), id);
                let mut v = vocab[&a].clone();
                v.extend_from_slice(&vocab[&b]);
                vocab.insert(id, v);
            }
            _ => break,
        }
    }
    (merges, vocab)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode::encode;
    use std::collections::HashMap;

    #[test]
    fn training_seeds_decoder_with_all_bytes() {
        let (merges, vocab) = train("abc", 1000);
        assert!(merges.is_empty());
        assert_eq!(vocab.len(), 256);
        for b in 0..=255u8 {
            assert_eq!(vocab[&u32::from(b)], vec![b]);
        }
    }

    #[test]
    fn training_on_empty_file_learns_nothing() {
        let (merges, vocab) = train("", 1000);
        assert!(merges.is_empty());
        assert_eq!(vocab.len(), 256);
    }

    #[test]
    fn training_merges_repeated_pair() {
        // "aaaa" -> [256, 256]; the remaining (256,256) occurs once, so stop
        let (merges, vocab) = train("aaaa", 1000);
        assert_eq!(merges, HashMap::from([((97, 97), 256)]));
        assert_eq!(vocab[&256], b"aa");
        assert_eq!(vocab.len(), 257);
    }

    #[test]
    fn training_builds_merges_on_earlier_merges() {
        // "abababab" -> [256; 4] -> [257, 257]
        let (merges, vocab) = train("abababab", 1000);
        assert_eq!(merges, HashMap::from([((97, 98), 256), ((256, 256), 257)]));
        assert_eq!(vocab[&256], b"ab");
        assert_eq!(vocab[&257], b"abab");
    }

    #[test]
    fn training_breaks_ties_with_lowest_pair() {
        // (97,98) and (99,100) both occur twice; the lower pair merges first,
        // so it gets the lower ID
        let (merges, _) = train("cdcdabab", 1000);
        assert_eq!(merges[&(97, 98)], 256);
        assert_eq!(merges[&(99, 100)], 257);
    }

    #[test]
    fn training_works_on_bytes_not_chars() {
        // "é" is the two bytes [195, 169]
        let (merges, vocab) = train("éé", 1000);
        assert_eq!(merges, HashMap::from([((195, 169), 256)]));
        assert_eq!(vocab[&256], "é".as_bytes());
    }

    #[test]
    fn training_matches_wikipedia_aaabdaaabac() {
        // Hand count, overlapping windows included:
        // aa x4 -> 256, then ab and (256,a) tie at 2 and the lower pair ab
        // wins -> 257, then (256,257) x2 -> 258, then every pair occurs once.
        let text = "aaabdaaabac";
        let (merges, _) = train(text, 1000);
        assert_eq!(
            merges,
            HashMap::from([((97, 97), 256), ((97, 98), 257), ((256, 257), 258),])
        );
        assert_eq!(encode(text, &merges), vec![258, 100, 258, 97, 99]);
    }

    #[test]
    fn training_respects_num_merges() {
        let (limited, _) = train("aaabdaaabac", 2);
        assert_eq!(limited.len(), 2);
        let (unlimited, _) = train("aaabdaaabac", 1000);
        assert!(unlimited.len() > limited.len());
    }

    fn varied_text(len: usize) -> String {
        // Deterministic ASCII with repeated "aaab" so many pairs tie and overlap.
        let mut state: u64 = 0xdecafbad;
        let mut text = String::with_capacity(len + 8);
        while text.len() < len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let byte = (state >> 33) as u8;
            text.push(char::from(b'a' + (byte % 26)));
            if byte.is_multiple_of(7) {
                text.push(' ');
            }
            if byte.is_multiple_of(11) {
                text.push_str("aaab");
            }
        }
        text
    }

    #[test]
    fn fast_training_matches_naive() {
        let samples = [
            ("", 10u32),
            ("abc", 10),
            ("aaaa", 10),
            ("aaa", 10),
            ("abababab", 10),
            ("cdcdabab", 10),
            ("aaabdaaabac", 1000),
            ("éééé", 10),
        ];
        for (text, num_merges) in samples {
            let (fast_merges, fast_vocab) = train(text, num_merges);
            let (naive_merges, naive_vocab) = train_naive(text, num_merges);
            assert_eq!(fast_merges, naive_merges, "merges differ for {text:?}");
            assert_eq!(fast_vocab, naive_vocab, "vocab differs for {text:?}");
        }

        for text in ["a".repeat(400), "ab".repeat(400), varied_text(4_000)] {
            let (fast_merges, fast_vocab) = train(&text, 200);
            let (naive_merges, naive_vocab) = train_naive(&text, 200);
            assert!(
                !fast_merges.is_empty(),
                "expected merges on a repetitive text"
            );
            assert_eq!(fast_merges, naive_merges);
            assert_eq!(fast_vocab, naive_vocab);
        }

        // Book-sized inputs are not in git. A few dozen kilobytes is enough to
        // exercise neighbor updates past the hand-checked examples, and it
        // still finishes quickly under the naive oracle.
        let mut corpus = String::new();
        let sentence = "the quick brown fox jumps over the lazy dog. aaabdaaabac ";
        while corpus.len() < 24 * 1024 {
            corpus.push_str(sentence);
            corpus.push_str("café ");
        }
        let (fast_merges, fast_vocab) = train(&corpus, 300);
        let (naive_merges, naive_vocab) = train_naive(&corpus, 300);
        assert!(fast_merges.len() > 50, "got {}", fast_merges.len());
        assert_eq!(fast_merges, naive_merges);
        assert_eq!(fast_vocab, naive_vocab);
    }

    #[test]
    #[ignore = "naive pass over Pride and Prejudice; needs book.txt and --release"]
    fn fast_training_matches_naive_on_book() {
        let text = match std::fs::read_to_string("book.txt") {
            Ok(text) => text,
            Err(err) => panic!("book.txt is required for this ignored test: {err}"),
        };
        let (fast_merges, fast_vocab) = train(&text, 1000);
        let (naive_merges, naive_vocab) = train_naive(&text, 1000);
        assert_eq!(fast_merges, naive_merges);
        assert_eq!(fast_vocab, naive_vocab);
        let tokens = crate::train::fast::trained_token_count(&text, 1000);
        assert!(tokens < text.len());
        eprintln!(
            "book bytes {} tokens {tokens} merges {}",
            text.len(),
            fast_merges.len()
        );
    }
}
