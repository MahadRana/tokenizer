use std::collections::HashMap;

pub fn top_pair(counts: &HashMap<(u32, u32), usize>) -> Option<((u32, u32), usize)> {
    if counts.is_empty(){
        return None
    }
    let mut max_count = 0;
    let mut max_pair = (0,0);
    for (key, value) in counts {
        if (*value > max_count) || ((*value == max_count) && max_pair > *key) {
            max_count = *value;
            max_pair = *key;
        } 
    }
    Some((max_pair, max_count))
}

pub fn merge(seq: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut i = 0;
    let mut v: Vec<u32> = Vec::new(); 
    while i < seq.len() {
        if i < seq.len() -1 && pair == (seq[i], seq[i+1]) {
            v.push(new_id);
            i += 2;
        } else {
            v.push(seq[i]);
            i += 1;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts_of(pairs: &[((u32, u32), usize)]) -> HashMap<(u32, u32), usize> {
        pairs.iter().cloned().collect()
    }

    #[test]
    fn top_pair_of_empty_map_is_none() {
        assert_eq!(top_pair(&HashMap::new()), None);
    }

    #[test]
    fn top_pair_finds_single_entry() {
        let counts = counts_of(&[((3, 4), 7)]);
        assert_eq!(top_pair(&counts), Some(((3, 4), 7)));
    }

    #[test]
    fn top_pair_finds_highest_count() {
        let counts = counts_of(&[((1, 2), 3), ((5, 6), 9), ((7, 8), 4)]);
        assert_eq!(top_pair(&counts), Some(((5, 6), 9)));
    }

    #[test]
    fn top_pair_breaks_ties_with_lowest_pair() {
        // HashMap iteration order is not stable, so run it enough times that
        // a non-deterministic tie-break would show up.
        let counts = counts_of(&[((9, 9), 5), ((1, 1), 5), ((4, 4), 5)]);
        for _ in 0..50 {
            assert_eq!(top_pair(&counts), Some(((1, 1), 5)));
        }
    }

    #[test]
    fn top_pair_tie_break_compares_second_element() {
        let counts = counts_of(&[((2, 9), 6), ((2, 3), 6)]);
        assert_eq!(top_pair(&counts), Some(((2, 3), 6)));
    }

    #[test]
    fn merge_replaces_single_occurrence() {
        assert_eq!(merge(&[1, 2, 3], (1, 2), 256), vec![256, 3]);
    }

    #[test]
    fn merge_replaces_every_occurrence() {
        assert_eq!(merge(&[1, 2, 0, 1, 2], (1, 2), 256), vec![256, 0, 256]);
    }

    #[test]
    fn merge_consumes_overlapping_runs_left_to_right() {
        // [7,7,7] -> first two merge, the trailing 7 is left alone
        assert_eq!(merge(&[7, 7, 7], (7, 7), 256), vec![256, 7]);
        assert_eq!(merge(&[7, 7, 7, 7], (7, 7), 256), vec![256, 256]);
    }

    #[test]
    fn merge_at_start_and_end_of_sequence() {
        assert_eq!(merge(&[1, 2, 9], (1, 2), 256), vec![256, 9]);
        assert_eq!(merge(&[9, 1, 2], (1, 2), 256), vec![9, 256]);
    }

    #[test]
    fn merge_leaves_sequence_unchanged_when_pair_absent() {
        assert_eq!(merge(&[1, 2, 3], (8, 9), 256), vec![1, 2, 3]);
    }

    #[test]
    fn merge_does_not_match_a_split_pair() {
        // the 1 and 2 are adjacent only across a merge that already happened
        assert_eq!(merge(&[2, 1], (1, 2), 256), vec![2, 1]);
    }

    #[test]
    fn merge_of_empty_sequence_is_empty() {
        assert_eq!(merge(&[], (1, 2), 256), Vec::<u32>::new());
    }

    #[test]
    fn merge_of_single_element_is_unchanged() {
        assert_eq!(merge(&[1], (1, 2), 256), vec![1]);
    }
}
