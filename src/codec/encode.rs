use crate::merge;
use std::collections::HashMap;

pub fn encode(text: &str, encoder: &HashMap<(u32, u32), u32>) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut seq: Vec<u32> = bytes.iter().map(|x| u32::from(*x)).collect();
    if seq.len() <= 1 {
        return seq;
    }

    let mut found = true; 
    while found {
        found = false;
        let mut min_pair = (u32::MAX,u32::MAX);
        let mut min_id = u32::MAX;
        for i in 0..seq.len()-1 {
            let pair = (seq[i], seq[i+1]);
            if encoder.contains_key(&pair) {
                found = true;
                let id = encoder[&pair];
                if id < min_id {
                    min_id = id;
                    min_pair = pair;
                }
            }
        }
        if found {
            seq = merge::merge(&seq, min_pair, min_id);
        }
    }
    seq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_of_empty_text_is_empty() {
        let encoder = HashMap::from([((97, 98), 256)]);
        assert_eq!(encode("", &encoder), Vec::<u32>::new());
    }

    #[test]
    fn encode_with_no_merges_returns_raw_bytes() {
        assert_eq!(encode("abc", &HashMap::new()), vec![97, 98, 99]);
    }

    #[test]
    fn encode_of_single_byte_is_unchanged() {
        let encoder = HashMap::from([((97, 97), 256)]);
        assert_eq!(encode("a", &encoder), vec![97]);
    }

    #[test]
    fn encode_leaves_text_unchanged_when_no_pair_matches() {
        let encoder = HashMap::from([((120, 121), 256)]);
        assert_eq!(encode("abc", &encoder), vec![97, 98, 99]);
    }

    #[test]
    fn encode_applies_single_merge_everywhere() {
        let encoder = HashMap::from([((97, 98), 256)]);
        assert_eq!(encode("abcab", &encoder), vec![256, 99, 256]);
    }

    #[test]
    fn encode_chains_merges_built_on_earlier_merges() {
        // "abababab" -> [256; 4] -> [257, 257]
        let encoder = HashMap::from([((97, 98), 256), ((256, 256), 257)]);
        assert_eq!(encode("abababab", &encoder), vec![257, 257]);
        assert_eq!(encode("ababab", &encoder), vec![257, 256]);
    }

    #[test]
    fn encode_applies_lowest_id_first_not_leftmost() {
        // "abc" has both (a,b) and (b,c); whichever was learned first wins.
        // HashMap iteration order is not stable, so run it enough times that
        // a non-deterministic choice would show up.
        let ab_first = HashMap::from([((97, 98), 256), ((98, 99), 257)]);
        let bc_first = HashMap::from([((98, 99), 256), ((97, 98), 257)]);
        for _ in 0..50 {
            assert_eq!(encode("abc", &ab_first), vec![256, 99]);
            assert_eq!(encode("abc", &bc_first), vec![97, 256]);
        }
    }

    #[test]
    fn encode_consumes_overlapping_runs_left_to_right() {
        // "aaa" -> [256, 97] -> [257]
        let encoder = HashMap::from([((97, 97), 256), ((256, 97), 257)]);
        assert_eq!(encode("aaa", &encoder), vec![257]);
        assert_eq!(encode("aaaa", &encoder), vec![256, 256]);
    }

    #[test]
    fn encode_works_on_bytes_not_chars() {
        // "é" is the two bytes [195, 169]
        let encoder = HashMap::from([((195, 169), 256)]);
        assert_eq!(encode("éé", &encoder), vec![256, 256]);
    }
}
