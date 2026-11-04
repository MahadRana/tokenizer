use super::counter;
use crate::merge;
use std::collections::HashMap;

pub type Merges = HashMap<(u32, u32), u32>;
pub type Vocab = HashMap<u32, Vec<u8>>;

pub fn train(text: &str, num_merges: u32) -> (Merges, Vocab) {
    let bytes = text.as_bytes();
    let mut seq: Vec<u32> = bytes.iter().map(|x| u32::from(*x)).collect();
    let mut merges: Merges = HashMap::new();
    let mut vocab: Vocab = HashMap::new();
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
            HashMap::from([
                ((97, 97), 256),
                ((97, 98), 257),
                ((256, 257), 258),
            ])
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
}
