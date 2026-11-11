//! Paged KV cache: a free-list of block ids and a fixed tensor pool.
//!
//! [`BlockId`] is a newtype around `u32`. A token id or a token position is
//! also a `u32`, and passing one where a block id is required is E0308 until
//! the caller writes [`BlockId`] explicitly. The free list is a stack: `alloc`
//! pops, `free_block` pushes, both O(1). The block freed most recently is the
//! next one handed out.
//!
//! The pool is one K tensor and one V tensor per layer, shaped
//! `(num_blocks, BLOCK_SIZE, kv_heads, head_dim)`. A sequence stores
//! [`PagedCache::block_table`], logical block to physical [`BlockId`]. It does
//! not borrow the pool. A `&mut` from each cache back to the pool would alias
//! the `Vec` of sequences (E0499). Indices are `Copy`.
//!
//! `Drop` for [`PagedCache`] cannot return those ids. It has no allocator.
//! Debug builds assert the table is empty. The scheduler calls
//! [`PagedCache::take_blocks`] and [`BlockAllocator::free_block`].

use std::collections::{HashMap, VecDeque};

use candle_core::{DType, Device, Result, Tensor, bail};

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

/// What [`BlockAllocator::free_block`] did with a live block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Release {
    /// Another sequence still holds this block. The KV stays.
    Shared,
    /// Refcount hit 0 and the block is a full prefix block. It stays out of
    /// the free stack so a later hash hit can reuse the KV. [`BlockAllocator::alloc`]
    /// evicts these, oldest first, before a caller has to preempt a live sequence.
    Cached,
    /// Refcount hit 0 and the block was private. It is back on the free stack.
    /// The caller should clear the KV.
    Freed,
}

/// Hash of one full block. `parent` is the previous block's hash, or 0 for
/// the first block. The token ids are mixed in, and a lookup still compares
/// the stored ids, so two different blocks that land on the same hash do not share.
pub fn prefix_block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64 ^ parent;
    for token in tokens {
        hash ^= u64::from(*token);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

struct PrefixEntry {
    id: BlockId,
    tokens: [u32; BLOCK_SIZE],
}

/// Stack of free physical blocks, plus a refcount and a prefix index.
///
/// `alloc` pops the free stack. A published full block whose refcount falls
/// to 0 is not pushed there: it stays in the prefix index until a later
/// `alloc` evicts it. Private blocks (anything not published) return to the
/// free stack at refcount 0, which is what the older tests do.
pub struct BlockAllocator {
    free: Vec<BlockId>,
    refcount: Vec<u32>,
    /// Hash of a published full block, if this id is in `index`.
    hash_of: Vec<Option<u64>>,
    capacity: u32,
    /// Published blocks with refcount 0, oldest at the front.
    cached: VecDeque<BlockId>,
    index: HashMap<u64, PrefixEntry>,
    peak_live: usize,
}

impl BlockAllocator {
    /// `num_blocks` free ids, `0..num_blocks`. The first [`Self::alloc`] returns 0.
    pub fn new(num_blocks: u32) -> Self {
        let mut free = Vec::with_capacity(num_blocks as usize);
        for id in (0..num_blocks).rev() {
            free.push(BlockId(id));
        }
        let n = num_blocks as usize;
        Self {
            free,
            refcount: vec![0; n],
            hash_of: vec![None; n],
            capacity: num_blocks,
            cached: VecDeque::new(),
            index: HashMap::new(),
            peak_live: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Ids on the free stack. Cached prefix blocks are not included.
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Free stack plus cached prefix blocks. Both can satisfy [`Self::alloc`].
    pub fn available(&self) -> usize {
        self.free.len() + self.cached.len()
    }

    pub fn cached_count(&self) -> usize {
        self.cached.len()
    }

    pub fn refcount(&self, id: BlockId) -> Option<u32> {
        self.refcount.get(id.index()).copied()
    }

    pub fn live_count(&self) -> usize {
        self.refcount.iter().filter(|count| **count > 0).count()
    }

    pub fn peak_live(&self) -> usize {
        self.peak_live
    }

    pub fn all_refcounts_zero(&self) -> bool {
        self.refcount.iter().all(|count| *count == 0)
    }

    /// Pop one id, from the free stack or by evicting the oldest cached block.
    /// `None` when every block has a positive refcount.
    pub fn alloc(&mut self) -> Option<BlockId> {
        if let Some(id) = self.free.pop() {
            self.refcount[id.index()] = 1;
            self.note_live();
            return Some(id);
        }
        let id = self.cached.pop_front()?;
        self.unindex(id);
        debug_assert_eq!(self.refcount[id.index()], 0);
        self.refcount[id.index()] = 1;
        self.note_live();
        Some(id)
    }

    /// Pop `n` ids, or pop none.
    ///
    /// A short pool does not move. `n == 0` returns an empty `Vec`. Cached
    /// prefix blocks are evicted after the free stack runs out.
    pub fn alloc_n(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if self.available() < n {
            return None;
        }
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(self.alloc().expect("available was checked"));
        }
        Some(ids)
    }

    /// Drop one reference. A second drop at refcount 0 is [`AllocError::DoubleFree`]
    /// and does not push the id again.
    pub fn free_block(&mut self, id: BlockId) -> std::result::Result<Release, AllocError> {
        let Some(slot) = self.refcount.get_mut(id.index()) else {
            return Err(AllocError::Unknown(id));
        };
        if *slot == 0 {
            return Err(AllocError::DoubleFree(id));
        }
        *slot -= 1;
        if *slot > 0 {
            return Ok(Release::Shared);
        }
        if self.hash_of[id.index()].is_some() {
            self.cached.push_back(id);
            Ok(Release::Cached)
        } else {
            self.free.push(id);
            Ok(Release::Freed)
        }
    }

    /// One more holder. A cached block leaves the LRU list. A free-stack id
    /// is not a valid target.
    pub fn incref(&mut self, id: BlockId) -> std::result::Result<(), AllocError> {
        let Some(slot) = self.refcount.get_mut(id.index()) else {
            return Err(AllocError::Unknown(id));
        };
        if *slot == 0 {
            let cached = self.cached.iter().position(|block| *block == id);
            let Some(pos) = cached else {
                return Err(AllocError::DoubleFree(id));
            };
            self.cached.remove(pos);
            *slot = 1;
            self.note_live();
            return Ok(());
        }
        *slot += 1;
        Ok(())
    }

    /// Record a full block under `prefix_block_hash(parent, tokens)`.
    ///
    /// A hash that already stores different token ids is left alone. The new
    /// block is not shared and is not indexed, so a collision cannot hand
    /// out the wrong KV.
    pub fn publish_full_block(
        &mut self,
        parent: u64,
        tokens: &[u32],
        id: BlockId,
    ) -> std::result::Result<(), AllocError> {
        if tokens.len() != BLOCK_SIZE {
            return Err(AllocError::Unknown(id));
        }
        if self.refcount.get(id.index()).copied().unwrap_or(0) == 0 {
            return Err(AllocError::Unknown(id));
        }
        self.publish_at_hash(prefix_block_hash(parent, tokens), tokens, id);
        Ok(())
    }

    /// Full blocks of `tokens` that are already published, in order.
    ///
    /// Each hit bumps that block's refcount. The walk stops at the first
    /// miss, so a prompt that diverges inside a block keeps only the earlier
    /// full blocks. The partial tail is never looked up.
    pub fn take_shared_prefix(&mut self, tokens: &[u32]) -> Vec<BlockId> {
        let mut parent = 0u64;
        let mut shared = Vec::new();
        let mut offset = 0;
        while offset + BLOCK_SIZE <= tokens.len() {
            let chunk = &tokens[offset..offset + BLOCK_SIZE];
            let Some(id) = self.find_full_block(parent, chunk) else {
                break;
            };
            if self.incref(id).is_err() {
                break;
            }
            shared.push(id);
            parent = prefix_block_hash(parent, chunk);
            offset += BLOCK_SIZE;
        }
        shared
    }

    /// Index every full block of `tokens`. `blocks[i]` holds tokens
    /// `[i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)`. A short tail is ignored.
    pub fn publish_prompt_blocks(&mut self, tokens: &[u32], blocks: &[BlockId]) {
        let full = tokens.len() / BLOCK_SIZE;
        let mut parent = 0u64;
        for index in 0..full {
            if index >= blocks.len() {
                break;
            }
            let chunk = &tokens[index * BLOCK_SIZE..(index + 1) * BLOCK_SIZE];
            let _ = self.publish_full_block(parent, chunk, blocks[index]);
            parent = prefix_block_hash(parent, chunk);
        }
    }

    /// A published block with these exact token ids, if the hash hits.
    pub fn find_full_block(&self, parent: u64, tokens: &[u32]) -> Option<BlockId> {
        if tokens.len() != BLOCK_SIZE {
            return None;
        }
        self.find_at_hash(prefix_block_hash(parent, tokens), tokens)
    }

    /// Drop every cached prefix block onto the free stack.
    ///
    /// Refcounts are already 0. Shutdown uses this so `free_count` covers
    /// the whole pool once no sequence is alive.
    pub fn reclaim_cached(&mut self) {
        while let Some(id) = self.cached.pop_front() {
            self.unindex(id);
            debug_assert_eq!(self.refcount[id.index()], 0);
            self.free.push(id);
        }
    }

    fn publish_at_hash(&mut self, hash: u64, tokens: &[u32], id: BlockId) -> bool {
        debug_assert_eq!(tokens.len(), BLOCK_SIZE);
        if let Some(existing) = self.index.get(&hash) {
            return existing.tokens.as_slice() == tokens && existing.id == id;
        }
        if self.hash_of[id.index()].is_some() {
            return false;
        }
        let mut stored = [0u32; BLOCK_SIZE];
        stored.copy_from_slice(tokens);
        self.index.insert(hash, PrefixEntry { id, tokens: stored });
        self.hash_of[id.index()] = Some(hash);
        true
    }

    fn find_at_hash(&self, hash: u64, tokens: &[u32]) -> Option<BlockId> {
        let entry = self.index.get(&hash)?;
        if entry.tokens.as_slice() == tokens {
            Some(entry.id)
        } else {
            None
        }
    }

    fn unindex(&mut self, id: BlockId) {
        if let Some(hash) = self.hash_of[id.index()].take()
            && self.index.get(&hash).is_some_and(|entry| entry.id == id)
        {
            self.index.remove(&hash);
        }
    }

    fn note_live(&mut self) {
        let live = self.live_count();
        self.peak_live = self.peak_live.max(live);
    }
}

/// How many blocks a sequence of `len` tokens occupies.
///
/// Zero when `len` is zero. Otherwise `ceil(len / BLOCK_SIZE)`.
pub fn blocks_for_len(len: usize) -> usize {
    len.div_ceil(BLOCK_SIZE)
}

/// One sequence's map from logical block to physical [`BlockId`].
///
/// `seq_len` is how many tokens are stored. The table may be a block longer
/// when the scheduler reserved the next token's slot before writing it.
/// Gather reads the whole table. The mask hides columns at and past `seq_len`.
pub struct PagedCache {
    block_table: Vec<BlockId>,
    seq_len: usize,
    n_layers: usize,
}

impl PagedCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            block_table: Vec::new(),
            seq_len: 0,
            n_layers,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.n_layers
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn block_count(&self) -> usize {
        self.block_table.len()
    }

    pub fn block_table(&self) -> &[BlockId] {
        &self.block_table
    }

    pub fn assign_blocks(&mut self, blocks: impl IntoIterator<Item = BlockId>) {
        self.block_table.extend(blocks);
    }

    /// How many tokens are already stored. Sharing sets this to the matched
    /// prefix so the next forward starts at the suffix.
    pub fn set_seq_len(&mut self, len: usize) {
        debug_assert!(
            len == 0 || blocks_for_len(len) <= self.block_table.len(),
            "seq_len {len} does not fit in {} blocks",
            self.block_table.len()
        );
        self.seq_len = len;
    }

    /// Detach every id and set `seq_len` to 0. The caller frees the ids.
    pub fn take_blocks(&mut self) -> Vec<BlockId> {
        self.seq_len = 0;
        std::mem::take(&mut self.block_table)
    }
}

impl Drop for PagedCache {
    fn drop(&mut self) {
        debug_assert!(
            self.block_table.is_empty(),
            "PagedCache dropped with {} blocks still assigned; free them on the allocator",
            self.block_table.len()
        );
    }
}

/// K and V for every layer, allocated once.
///
/// `k[layer]` and `v[layer]` are `(num_blocks, block_size, kv_heads, head_dim)`.
/// Writes use [`Tensor::slice_set`](candle_core::Tensor::slice_set). Reads use
/// [`Tensor::index_select`](candle_core::Tensor::index_select) on dimension 0.
/// <https://docs.rs/candle-core/0.11.1/candle_core/struct.Tensor.html#method.index_select>
pub struct PagedKvPool {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    zero_block: Tensor,
    num_blocks: usize,
    block_size: usize,
    kv_heads: usize,
    head_dim: usize,
    n_layers: usize,
    dtype: DType,
    device: Device,
}

impl PagedKvPool {
    pub fn new(
        num_blocks: usize,
        n_layers: usize,
        kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if num_blocks == 0 || n_layers == 0 || kv_heads == 0 || head_dim == 0 {
            bail!("paged kv pool has an empty dimension");
        }
        if u32::try_from(num_blocks).is_err() {
            bail!("num_blocks does not fit in u32");
        }
        let block_size = BLOCK_SIZE;
        let shape = (num_blocks, block_size, kv_heads, head_dim);
        let zero_block = Tensor::zeros((1, block_size, kv_heads, head_dim), dtype, device)?;
        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k.push(Tensor::zeros(shape, dtype, device)?);
            v.push(Tensor::zeros(shape, dtype, device)?);
        }
        Ok(Self {
            k,
            v,
            zero_block,
            num_blocks,
            block_size,
            kv_heads,
            head_dim,
            n_layers,
            dtype,
            device: device.clone(),
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn num_layers(&self) -> usize {
        self.n_layers
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Bytes of K and V. The one-block zero tensor used on free is not included.
    pub fn kv_bytes(&self) -> usize {
        let one = self.k.first().map(Tensor::elem_count).unwrap_or(0);
        one * self.dtype.size_in_bytes() * 2 * self.n_layers
    }

    /// Write `k` and `v` at absolute positions `pos..pos+seq`.
    ///
    /// Both tensors are `(1, kv_heads, seq, head_dim)`, which is the layout
    /// attention produces after RoPE. The pool stores a block as
    /// `(block, token, kv_heads, head_dim)`.
    pub fn write_layer(
        &self,
        layer: usize,
        table: &[BlockId],
        pos: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        if layer >= self.n_layers {
            bail!("kv pool layer {layer} is out of range ({})", self.n_layers);
        }
        write_tokens(
            &self.k[layer],
            self.block_size,
            self.kv_heads,
            self.head_dim,
            table,
            pos,
            k,
        )?;
        write_tokens(
            &self.v[layer],
            self.block_size,
            self.kv_heads,
            self.head_dim,
            table,
            pos,
            v,
        )
    }

    /// `(1, kv_heads, table.len() * block_size, head_dim)` in block-table order.
    pub fn gather_layer(&self, layer: usize, table: &[BlockId], keys: bool) -> Result<Tensor> {
        if layer >= self.n_layers {
            bail!("kv pool layer {layer} is out of range ({})", self.n_layers);
        }
        if table.is_empty() {
            bail!("no blocks to gather");
        }
        let pool = if keys { &self.k[layer] } else { &self.v[layer] };
        let ids: Vec<u32> = table.iter().copied().map(BlockId::as_u32).collect();
        let index = Tensor::from_vec(ids, (table.len(),), &self.device)?;
        let selected = pool.index_select(&index, 0)?;
        let (n, block_size, kv_heads, head_dim) = selected.dims4()?;
        selected
            .reshape((n * block_size, kv_heads, head_dim))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()
    }

    /// Zero one physical block in every layer before the id is reused.
    pub fn clear_block(&self, id: BlockId) -> Result<()> {
        let index = id.index();
        if index >= self.num_blocks {
            bail!("{id} is outside the pool ({})", self.num_blocks);
        }
        for layer in 0..self.n_layers {
            self.k[layer].slice_set(&self.zero_block, 0, index)?;
            self.v[layer].slice_set(&self.zero_block, 0, index)?;
        }
        Ok(())
    }
}

fn write_tokens(
    pool: &Tensor,
    block_size: usize,
    kv_heads: usize,
    head_dim: usize,
    table: &[BlockId],
    pos: usize,
    tokens: &Tensor,
) -> Result<()> {
    let (batch, heads, seq, dim) = tokens.dims4()?;
    if batch != 1 || heads != kv_heads || dim != head_dim {
        bail!(
            "kv write shape {:?}, pool heads {kv_heads} dim {head_dim}",
            tokens.shape()
        );
    }
    if seq == 0 {
        return Ok(());
    }
    let end = pos + seq;
    let need = blocks_for_len(end);
    if table.len() < need {
        bail!(
            "block table has {} blocks, write through token {end} needs {need}",
            table.len()
        );
    }
    let mut offset = 0;
    while offset < seq {
        let abs = pos + offset;
        let logical = abs / block_size;
        let within = abs % block_size;
        let take = (block_size - within).min(seq - offset);
        // (1, kv_heads, take, head_dim) -> (1, take, kv_heads, head_dim).
        let rows = tokens
            .narrow(2, offset, take)?
            .transpose(1, 2)?
            .contiguous()?;
        install_rows(pool, table[logical].index(), within, &rows)?;
        offset += take;
    }
    Ok(())
}

/// Splice `rows` at token offset `within` inside physical block `block`.
///
/// `rows` is `(1, take, kv_heads, head_dim)`. A full block replaces the
/// storage with one `slice_set`. A partial block is concatenated with the
/// head and tail already stored, then written back. The read is a copy, so
/// the write does not alias its source.
fn install_rows(pool: &Tensor, block: usize, within: usize, rows: &Tensor) -> Result<()> {
    let block_size = pool.dim(1)?;
    let take = rows.dim(1)?;
    let rows = rows.contiguous()?;
    if within == 0 && take == block_size {
        return pool.slice_set(&rows, 0, block);
    }
    let existing = pool.narrow(0, block, 1)?.contiguous()?;
    let mut parts = Vec::new();
    if within > 0 {
        parts.push(existing.narrow(1, 0, within)?.contiguous()?);
    }
    parts.push(rows);
    let tail = block_size - within - take;
    if tail > 0 {
        parts.push(existing.narrow(1, within + take, tail)?.contiguous()?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let updated = Tensor::cat(&refs, 1)?.contiguous()?;
    pool.slice_set(&updated, 0, block)
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

    /// B4 stores each sequence with `Tensor::cat` at its exact length. Paging
    /// rounds that length up to a block, so the same sequences take more
    /// bytes, not fewer. The waste is under one block per sequence.
    ///
    /// A max-length reservation is not this baseline. B4 does not reserve one.
    /// The ≥2× packing figure against that reservation is not a B4 result,
    /// and this test does not assert it.
    #[test]
    fn paged_bytes_exceed_exact_cat_by_less_than_one_block_per_sequence() {
        // Qwen2.5-0.5B: 24 layers, 2 kv heads, head dim 64, f32.
        // One token of K and V is 24 * 2 * 2 * 64 * 4 = 24576 bytes.
        const LAYERS: usize = 24;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        let bytes_per_token = LAYERS * 2 * KV_HEADS * HEAD_DIM * std::mem::size_of::<f32>();
        assert_eq!(bytes_per_token, 24_576);

        let lengths = [
            8usize, 16, 17, 24, 31, 32, 40, 48, 63, 64, 80, 96, 100, 127, 128, 160, 200, 255,
        ];
        // Repeat the list the same way the old count filled a 256-block pool,
        // so the published 51-sequence numbers stay checkable. The comparison
        // is still exact cat versus paged blocks for those sequences.
        let mut used_blocks = 0usize;
        let mut seq_lengths = Vec::new();
        while seq_lengths.len() < 10_000 {
            let len = lengths[seq_lengths.len() % lengths.len()];
            let blocks = blocks_for_len(len);
            if used_blocks + blocks > 256 {
                break;
            }
            used_blocks += blocks;
            seq_lengths.push(len);
        }
        let n = seq_lengths.len();
        let exact_tokens: usize = seq_lengths.iter().sum();
        let paged_tokens = used_blocks * BLOCK_SIZE;
        let waste = paged_tokens - exact_tokens;
        let exact_bytes = exact_tokens * bytes_per_token;
        let paged_bytes = paged_tokens * bytes_per_token;

        let pool = PagedKvPool::new(
            used_blocks,
            LAYERS,
            KV_HEADS,
            HEAD_DIM,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(pool.kv_bytes(), paged_bytes);
        // One exact-length K tensor. Times 2 (K and V) times the layer count
        // is what `Tensor::cat` holds for that sequence after the forward.
        let sample_len = seq_lengths[0];
        let exact_k = Tensor::zeros(
            (1, KV_HEADS, sample_len, HEAD_DIM),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            exact_k.elem_count() * std::mem::size_of::<f32>() * 2 * LAYERS,
            sample_len * bytes_per_token
        );

        eprintln!(
            "exact cat {exact_bytes} bytes, paged {paged_bytes} bytes, {n} sequences, waste {waste} token slots"
        );
        eprintln!(
            "paged/exact {:.4}; sequences are the same count, so this is not a 2x packing win",
            paged_bytes as f64 / exact_bytes as f64
        );
        assert!(
            paged_bytes > exact_bytes,
            "paged {paged_bytes} is not above exact cat {exact_bytes}"
        );
        assert!(
            waste < BLOCK_SIZE * n,
            "waste {waste} is not under one block per sequence"
        );
    }

    #[test]
    fn gather_reads_back_what_write_stored() {
        let kv_heads = 2usize;
        let head_dim = 4usize;
        let pool = PagedKvPool::new(4, 1, kv_heads, head_dim, DType::F32, &Device::Cpu).unwrap();
        let mut alloc = BlockAllocator::new(4);
        let mut cache = PagedCache::new(1);
        cache.assign_blocks(alloc.alloc_n(2).unwrap());

        let first = pattern_tokens(0, 10, kv_heads, head_dim);
        let second = pattern_tokens(10, 10, kv_heads, head_dim);
        pool.write_layer(0, cache.block_table(), 0, &first, &first)
            .unwrap();
        pool.write_layer(0, cache.block_table(), 10, &second, &second)
            .unwrap();

        let gathered = pool.gather_layer(0, cache.block_table(), true).unwrap();
        assert_eq!(gathered.dims(), &[1, kv_heads, 32, head_dim]);
        let flat = gathered
            .reshape((kv_heads, 32, head_dim))
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        for (head, head_row) in flat.iter().enumerate() {
            for (token, token_row) in head_row.iter().enumerate() {
                for (dim, got) in token_row.iter().enumerate() {
                    let want = if token < 20 {
                        pattern_value(token, head, dim)
                    } else {
                        0.0
                    };
                    assert_eq!(*got, want, "head {head} token {token} dim {dim}");
                }
            }
        }
        let vflat = pool
            .gather_layer(0, cache.block_table(), false)
            .unwrap()
            .reshape((kv_heads, 32, head_dim))
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(vflat[1][16][3], pattern_value(16, 1, 3));

        let id = cache.block_table()[1];
        pool.clear_block(id).unwrap();
        let cleared = pool
            .gather_layer(0, cache.block_table(), true)
            .unwrap()
            .reshape((kv_heads, 32, head_dim))
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(cleared[0][16][0], 0.0);
        assert_eq!(cleared[0][9][1], pattern_value(9, 0, 1));

        for id in cache.take_blocks() {
            alloc.free_block(id).unwrap();
        }
        assert_eq!(alloc.free_count(), alloc.capacity());
    }

    fn pattern_value(token: usize, head: usize, dim: usize) -> f32 {
        (token * 100 + head * 10 + dim) as f32
    }

    #[test]
    fn shared_prefix_uses_fewer_blocks_than_two_copies() {
        let prompt = vec![3u32; 32];
        let mut alloc = BlockAllocator::new(8);
        let first = alloc.alloc_n(2).unwrap();
        alloc.publish_prompt_blocks(&prompt, &first);
        let shared = alloc.take_shared_prefix(&prompt);
        assert_eq!(shared.len(), 2);
        assert_eq!(
            alloc.live_count(),
            2,
            "two private copies would take 4 blocks"
        );
        assert_eq!(alloc.refcount(first[0]), Some(2));
        assert_eq!(alloc.refcount(first[1]), Some(2));
        assert_eq!(alloc.peak_live(), 2);
        for id in first.into_iter().chain(shared) {
            alloc.free_block(id).unwrap();
        }
        assert!(alloc.all_refcounts_zero());
        assert_eq!(alloc.cached_count(), 2);
        alloc.reclaim_cached();
        assert_eq!(alloc.free_count(), alloc.capacity());
    }

    #[test]
    fn hash_collision_does_not_share() {
        let mut alloc = BlockAllocator::new(4);
        let id = alloc.alloc().unwrap();
        let other_id = alloc.alloc().unwrap();
        let stored = [1u32; BLOCK_SIZE];
        let other = [2u32; BLOCK_SIZE];
        assert!(alloc.publish_at_hash(7, &stored, id));
        assert!(!alloc.publish_at_hash(7, &other, other_id));
        assert_eq!(alloc.find_at_hash(7, &other), None);
        assert_eq!(alloc.find_at_hash(7, &stored), Some(id));
        assert_eq!(alloc.take_shared_prefix(&other), Vec::<BlockId>::new());
        alloc.free_block(id).unwrap();
        alloc.free_block(other_id).unwrap();
        alloc.reclaim_cached();
    }

    #[test]
    fn prompt_that_diverges_inside_a_block_shares_only_earlier_blocks() {
        let mut left = vec![4u32; 20];
        let mut right = left.clone();
        right[17] = 9;
        let mut alloc = BlockAllocator::new(4);
        let blocks = alloc.alloc_n(2).unwrap();
        alloc.publish_prompt_blocks(&left, &blocks);
        let shared = alloc.take_shared_prefix(&right);
        assert_eq!(shared, vec![blocks[0]]);
        assert_eq!(alloc.refcount(blocks[0]), Some(2));
        assert_eq!(alloc.refcount(blocks[1]), Some(1));
        assert!(alloc.find_full_block(0, &left[..16]).is_some());
        let parent = prefix_block_hash(0, &left[..16]);
        assert!(alloc.find_full_block(parent, &right[16..]).is_none());
        left[17] = 9;
        assert_eq!(left, right);
    }

    #[test]
    fn dropping_one_sharer_leaves_the_block_held() {
        let prompt = vec![8u32; 32];
        let mut alloc = BlockAllocator::new(4);
        let blocks = alloc.alloc_n(2).unwrap();
        alloc.publish_prompt_blocks(&prompt, &blocks);
        let shared = alloc.take_shared_prefix(&prompt);
        assert_eq!(alloc.free_block(blocks[0]).unwrap(), Release::Shared);
        assert_eq!(alloc.free_block(blocks[1]).unwrap(), Release::Shared);
        assert_eq!(alloc.refcount(shared[0]), Some(1));
        assert_eq!(alloc.live_count(), 2);
        assert!(alloc.alloc().is_some());
        assert_ne!(alloc.free_count(), alloc.capacity());
    }

    #[test]
    fn alloc_evicts_cached_blocks_before_it_returns_none() {
        let prompt = vec![5u32; 16];
        let mut alloc = BlockAllocator::new(1);
        let id = alloc.alloc().unwrap();
        alloc.publish_full_block(0, &prompt, id).unwrap();
        assert_eq!(alloc.free_block(id).unwrap(), Release::Cached);
        assert!(alloc.alloc().is_some());
        assert_eq!(alloc.live_count(), 1);
        assert!(alloc.find_full_block(0, &prompt).is_none());
        assert_eq!(alloc.cached_count(), 0);
    }

    fn pattern_tokens(pos: usize, seq: usize, kv_heads: usize, head_dim: usize) -> Tensor {
        let mut data = vec![0f32; kv_heads * seq * head_dim];
        for head in 0..kv_heads {
            for token in 0..seq {
                for dim in 0..head_dim {
                    let index = (head * seq + token) * head_dim + dim;
                    data[index] = pattern_value(pos + token, head, dim);
                }
            }
        }
        Tensor::from_vec(data, (1, kv_heads, seq, head_dim), &Device::Cpu).unwrap()
    }
}
