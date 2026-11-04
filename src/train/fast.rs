//! Incremental BPE trainer.
//!
//! The A1 trainer recounts every pair and allocates a new `Vec` on every
//! merge, which is O(N) work per merge. This one keeps the counts and only
//! edits the pairs that touch a merged location.
//!
//! The sequence is a doubly linked list stored in parallel `Vec`s. The links
//! are indexes, and indexes are `Copy`, so merging never needs an aliased
//! `&mut` into the sequence. A `BinaryHeap` picks the most common pair. Stale
//! heap entries are left in place and ignored when their count no longer
//! matches the map (lazy deletion). The heap key is `(count, Reverse(pair))`,
//! so a tie still resolves to the lower pair, matching `merge::top_pair`.

use super::training::{Merges, Vocab};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

const NONE: usize = usize::MAX;

type Pair = (u32, u32);

struct Trainer {
    token: Vec<u32>,
    prev: Vec<usize>,
    next: Vec<usize>,
    alive: Vec<bool>,
    /// First live node. Read when walking the list (debug checks and the
    /// book-length test). Merging never needs it: the head node is never deleted.
    #[cfg_attr(not(any(test, debug_assertions)), allow(dead_code))]
    head: usize,
    counts: HashMap<Pair, i64>,
    /// Start indexes of each pair, including stale ones. A start is stale when
    /// that node is dead or its current tokens are no longer this pair.
    positions: HashMap<Pair, Vec<usize>>,
    heap: BinaryHeap<(i64, Reverse<Pair>)>,
    /// Pairs whose count changed during the merge currently being applied.
    /// Pushed onto the heap once, after that merge, at the final count.
    dirty: HashSet<Pair>,
    merges: Merges,
    vocab: Vec<Vec<u8>>,
}

impl Trainer {
    fn new(bytes: &[u8]) -> Self {
        let n = bytes.len();
        let token: Vec<u32> = bytes.iter().copied().map(u32::from).collect();
        let mut prev = vec![NONE; n];
        let mut next = vec![NONE; n];
        if n >= 2 {
            for i in 0..n - 1 {
                next[i] = i + 1;
                prev[i + 1] = i;
            }
        }

        let mut counts: HashMap<Pair, i64> = HashMap::new();
        let mut positions: HashMap<Pair, Vec<usize>> = HashMap::new();
        if n >= 2 {
            let mut i = 0;
            while next[i] != NONE {
                let j = next[i];
                let pair = (token[i], token[j]);
                *counts.entry(pair).or_insert(0) += 1;
                positions.entry(pair).or_default().push(i);
                i = j;
            }
        }

        let mut heap = BinaryHeap::with_capacity(counts.len());
        for (&pair, &count) in &counts {
            if count >= 2 {
                heap.push((count, Reverse(pair)));
            }
        }

        let mut vocab = Vec::with_capacity(256);
        for b in 0..256u32 {
            vocab.push(vec![b as u8]);
        }

        Self {
            token,
            prev,
            next,
            alive: vec![true; n],
            head: if n == 0 { NONE } else { 0 },
            counts,
            positions,
            heap,
            dirty: HashSet::new(),
            merges: HashMap::new(),
            vocab,
        }
    }

    fn run(&mut self, num_merges: u32) {
        #[cfg(debug_assertions)]
        let n = self.token.len();
        for i in 0..num_merges {
            let Some(pair) = self.pop_best() else {
                break;
            };
            let Some(id) = 256u32.checked_add(i) else {
                break;
            };
            self.merge_pair(pair, id);
            self.merges.insert(pair, id);
            let mut bytes = self.vocab[pair.0 as usize].clone();
            bytes.extend_from_slice(&self.vocab[pair.1 as usize]);
            debug_assert_eq!(self.vocab.len(), id as usize);
            self.vocab.push(bytes);

            // Full recount is the naive algorithm's cost, so only do it for
            // short inputs (and the first few merges of a long one). Release
            // builds strip this out.
            #[cfg(debug_assertions)]
            if i < 30 || n <= 32_768 {
                self.assert_consistent();
            }
        }
    }

    fn pop_best(&mut self) -> Option<Pair> {
        while let Some((count, Reverse(pair))) = self.heap.pop() {
            let current = self.counts.get(&pair).copied().unwrap_or(0);
            if current == count && count >= 2 {
                return Some(pair);
            }
        }
        None
    }

    /// Merge every non-overlapping occurrence of `pair`, left to right.
    ///
    /// This matches `merge::merge`: a run of the same token is consumed in
    /// steps of two, and a pair created by this merge is not merged again
    /// until a later rule. `new_id` is fresh, so the new neighbor pairs
    /// `(prev, new_id)` and `(new_id, next)` cannot equal `pair`.
    fn merge_pair(&mut self, pair: Pair, new_id: u32) {
        let mut locs = self.positions.remove(&pair).unwrap_or_default();
        locs.sort_unstable();
        locs.dedup();
        self.dirty.clear();

        for i in locs {
            if !self.alive[i] {
                continue;
            }
            let j = self.next[i];
            if j == NONE || !self.alive[j] {
                continue;
            }
            if self.token[i] != pair.0 || self.token[j] != pair.1 {
                continue;
            }

            let prev = self.prev[i];
            let nxt = self.next[j];
            self.sub(pair);
            if prev != NONE {
                let prev_tok = self.token[prev];
                self.sub((prev_tok, pair.0));
                self.add((prev_tok, new_id), prev);
            }
            if nxt != NONE {
                let next_tok = self.token[nxt];
                self.sub((pair.1, next_tok));
                self.add((new_id, next_tok), i);
            }

            self.token[i] = new_id;
            self.alive[j] = false;
            self.next[j] = NONE;
            self.prev[j] = NONE;
            self.next[i] = nxt;
            if nxt != NONE {
                self.prev[nxt] = i;
            }
        }

        debug_assert_eq!(self.counts.get(&pair).copied().unwrap_or(0), 0);

        let dirty: Vec<Pair> = self.dirty.drain().collect();
        for changed in dirty {
            if let Some(&count) = self.counts.get(&changed)
                && count >= 2
            {
                self.heap.push((count, Reverse(changed)));
            }
        }
    }

    fn add(&mut self, pair: Pair, pos: usize) {
        *self.counts.entry(pair).or_insert(0) += 1;
        self.positions.entry(pair).or_default().push(pos);
        self.dirty.insert(pair);
    }

    fn sub(&mut self, pair: Pair) {
        let next = match self.counts.get_mut(&pair) {
            Some(count) => {
                *count -= 1;
                *count
            }
            None => panic!("decrement of {pair:?} which is not in the count map"),
        };
        if next == 0 {
            self.counts.remove(&pair);
        } else {
            debug_assert!(next > 0, "negative count for {pair:?}");
        }
        self.dirty.insert(pair);
    }

    fn finish(self) -> (Merges, Vocab) {
        let mut vocab = HashMap::with_capacity(self.vocab.len());
        for (id, bytes) in self.vocab.into_iter().enumerate() {
            vocab.insert(id as u32, bytes);
        }
        (self.merges, vocab)
    }

    #[cfg(debug_assertions)]
    fn assert_consistent(&self) {
        let mut expected: HashMap<Pair, i64> = HashMap::new();
        if self.head != NONE {
            let mut i = self.head;
            loop {
                assert!(self.alive[i], "dead node {i} still linked");
                let j = self.next[i];
                if j == NONE {
                    break;
                }
                *expected.entry((self.token[i], self.token[j])).or_insert(0) += 1;
                assert_eq!(self.prev[j], i, "prev link broken at {j}");
                i = j;
            }
        }
        for (&pair, &count) in &expected {
            assert_eq!(
                self.counts.get(&pair).copied().unwrap_or(0),
                count,
                "count map missing {pair:?}"
            );
        }
        for (&pair, &count) in &self.counts {
            assert_eq!(
                expected.get(&pair).copied().unwrap_or(0),
                count,
                "count map has extra {pair:?}"
            );
        }
    }

    #[cfg(test)]
    fn token_count(&self) -> usize {
        if self.head == NONE {
            return 0;
        }
        let mut n = 0;
        let mut i = self.head;
        loop {
            n += 1;
            if self.next[i] == NONE {
                break;
            }
            i = self.next[i];
        }
        n
    }
}

pub fn train(text: &str, num_merges: u32) -> (Merges, Vocab) {
    let mut trainer = Trainer::new(text.as_bytes());
    trainer.run(num_merges);
    trainer.finish()
}

#[cfg(test)]
pub fn trained_token_count(text: &str, num_merges: u32) -> usize {
    let mut trainer = Trainer::new(text.as_bytes());
    trainer.run(num_merges);
    trainer.token_count()
}
