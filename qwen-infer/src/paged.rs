//! Block ids and the free list that hands them out.
//!
//! A paged KV cache is a fixed pool of blocks plus a table on each sequence.
//! This file is the allocator for that pool. The tensors, the `index_select`
//! gather, and the scheduler come next. They are not in this commit.
//!
//! [`BlockId`] is a newtype around `u32`. A token id or a token position is
//! also a `u32`, and passing one where a block id is required is E0308 until
//! the caller writes [`BlockId`] explicitly. The free list is a stack: `alloc`
//! pops, `free_block` pushes, both O(1). The block freed most recently is the
//! next one handed out.

/// Tokens in one physical block.
///
/// A sequence of length `len` uses [`blocks_for_len`] blocks and wastes
/// `blocks * BLOCK_SIZE - len` token slots. That waste is strictly less than
/// one block, which is what `fragmentation_is_bounded` checks.
pub const BLOCK_SIZE: usize = 16;

/// Physical block index. Not a token id and not a token position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(u32);

impl BlockId {
    /// The raw index. Callers that need a `u32` for a tensor index opt in here.
    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "block {}", self.0)
    }
}

/// Why [`BlockAllocator::free_block`] refused an id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// The id is already on the free stack.
    DoubleFree(BlockId),
    /// The id is outside `0..capacity`.
    Unknown(BlockId),
}

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocError::DoubleFree(id) => write!(f, "double free of {id}"),
            AllocError::Unknown(id) => write!(f, "{id} was not issued by this allocator"),
        }
    }
}

impl std::error::Error for AllocError {}

/// Stack of free physical blocks.
///
/// Blocks are one size, so a free count of `n` can always satisfy `alloc_n(n)`.
/// There is no external fragmentation. The only waste is the partial last
/// block of each sequence, which is under one block.
pub struct BlockAllocator {
    free: Vec<BlockId>,
    /// `true` while `id` sits on `free`. A second free is [`AllocError::DoubleFree`]
    /// and does not push again.
    is_free: Vec<bool>,
    capacity: u32,
}

impl BlockAllocator {
    /// `num_blocks` free ids, `0..num_blocks`. The first [`Self::alloc`] returns 0.
    pub fn new(num_blocks: u32) -> Self {
        let mut free = Vec::with_capacity(num_blocks as usize);
        for id in (0..num_blocks).rev() {
            free.push(BlockId(id));
        }
        Self {
            free,
            is_free: vec![true; num_blocks as usize],
            capacity: num_blocks,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Pop one id. `None` when every block is in use.
    pub fn alloc(&mut self) -> Option<BlockId> {
        let id = self.free.pop()?;
        self.is_free[id.index()] = false;
        Some(id)
    }

    /// Pop `n` ids, or pop none.
    ///
    /// A short free list does not move. `n == 0` returns an empty `Vec` and
    /// leaves the stack alone.
    pub fn alloc_n(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if self.free.len() < n {
            return None;
        }
        let start = self.free.len() - n;
        let mut ids = self.free.split_off(start);
        // `split_off` keeps stack order. `n` pops yield that slice reversed.
        ids.reverse();
        for id in &ids {
            self.is_free[id.index()] = false;
        }
        Some(ids)
    }

    /// Push `id` back. A second push of the same id is an error and does not
    /// grow the stack.
    pub fn free_block(&mut self, id: BlockId) -> Result<(), AllocError> {
        let Some(slot) = self.is_free.get_mut(id.index()) else {
            return Err(AllocError::Unknown(id));
        };
        if *slot {
            return Err(AllocError::DoubleFree(id));
        }
        *slot = true;
        self.free.push(id);
        Ok(())
    }
}

/// How many blocks a sequence of `len` tokens occupies.
///
/// Zero when `len` is zero. Otherwise `ceil(len / BLOCK_SIZE)`.
pub fn blocks_for_len(len: usize) -> usize {
    len.div_ceil(BLOCK_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_roundtrip() {
        let mut alloc = BlockAllocator::new(8);
        let mut ids = Vec::new();
        for _ in 0..8 {
            ids.push(alloc.alloc().expect("free block"));
        }
        assert!(alloc.alloc().is_none());
        assert_eq!(alloc.free_count(), 0);
        let mut seen: Vec<u32> = ids.iter().copied().map(BlockId::as_u32).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen, (0..8).collect::<Vec<_>>());

        // Push in alloc order. The stack returns the most recently freed id.
        for id in &ids {
            alloc.free_block(*id).unwrap();
        }
        assert_eq!(alloc.free_count(), 8);
        assert_eq!(alloc.alloc().unwrap(), *ids.last().unwrap());

        let mut one = BlockAllocator::new(8);
        let mut popped = Vec::new();
        for _ in 0..8 {
            popped.push(one.alloc().unwrap());
        }
        let mut batched = BlockAllocator::new(8);
        assert_eq!(batched.alloc_n(8).unwrap(), popped);
        assert!(batched.alloc_n(1).is_none());
        assert_eq!(batched.free_count(), 0);
    }

    #[test]
    fn no_double_free() {
        let mut alloc = BlockAllocator::new(2);
        let id = alloc.alloc().unwrap();
        alloc.free_block(id).unwrap();
        let err = alloc.free_block(id).unwrap_err();
        assert_eq!(err, AllocError::DoubleFree(id));
        // The rejected free did not push a second copy.
        assert_eq!(alloc.free_count(), 2);
        assert_eq!(alloc.alloc().unwrap(), id);
        let other = alloc.alloc().unwrap();
        assert_ne!(other, id);
        assert!(alloc.alloc().is_none());
        let err = alloc.free_block(BlockId(9)).unwrap_err();
        assert_eq!(err, AllocError::Unknown(BlockId(9)));
        assert_eq!(alloc.free_count(), 0);
    }

    #[test]
    fn oom_returns_none() {
        let mut alloc = BlockAllocator::new(2);
        assert!(alloc.alloc_n(3).is_none());
        assert_eq!(alloc.free_count(), 2, "a refused alloc_n must not pop");
        assert!(alloc.alloc().is_some());
        assert!(alloc.alloc().is_some());
        assert!(alloc.alloc().is_none());
        assert!(alloc.alloc_n(1).is_none());
        assert_eq!(alloc.free_count(), 0);
    }

    #[test]
    fn fragmentation_is_bounded() {
        let lengths = [
            1usize, 15, 16, 17, 31, 32, 33, 63, 64, 100, 127, 128, 200, 255, 256,
        ];
        let mut alloc = BlockAllocator::new(10_000);
        let mut held: Vec<Vec<BlockId>> = Vec::new();
        let mut waste = 0usize;
        for len in lengths {
            let blocks = blocks_for_len(len);
            let ids = alloc.alloc_n(blocks).expect("pool is larger than the sum");
            let gap = blocks * BLOCK_SIZE - len;
            assert!(gap < BLOCK_SIZE, "len {len} wastes {gap} token slots");
            waste += gap;
            held.push(ids);
        }
        assert!(
            waste < BLOCK_SIZE * lengths.len(),
            "waste {waste} is not under one block per sequence"
        );

        // Free every other sequence. The holes are whole blocks, so a later
        // request of exactly that many blocks still fits.
        let mut freed = 0usize;
        for (index, ids) in held.iter_mut().enumerate() {
            if index % 2 == 0 {
                freed += ids.len();
                for id in ids.drain(..) {
                    alloc.free_block(id).unwrap();
                }
            }
        }
        let again = alloc
            .alloc_n(freed)
            .expect("free blocks are not fragmented");
        assert_eq!(again.len(), freed);
        for id in again {
            alloc.free_block(id).unwrap();
        }
        for ids in held {
            for id in ids {
                alloc.free_block(id).unwrap();
            }
        }
        assert_eq!(alloc.free_count(), alloc.capacity());
    }
}
