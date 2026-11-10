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
    pub fn free_block(&mut self, id: BlockId) -> std::result::Result<(), AllocError> {
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

    pub(crate) fn set_seq_len(&mut self, len: usize) {
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

    /// One pool, sized like 8 sequences each reserved at 512 tokens.
    /// Counting how many mixed lengths fit is arithmetic on block counts,
    /// checked against the tensor's `elem_count`. It is not a timed run.
    #[test]
    fn same_budget_fits_at_least_twice_the_sequences() {
        // Qwen2.5-0.5B: 24 layers, 2 kv heads, head dim 64, f32.
        // One token of K and V is 24 * 2 * 2 * 64 * 4 = 24576 bytes.
        const LAYERS: usize = 24;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 64;
        let bytes_per_token = LAYERS * 2 * KV_HEADS * HEAD_DIM * std::mem::size_of::<f32>();
        assert_eq!(bytes_per_token, 24_576);

        let max_len = 512usize;
        let b4_reserved = 8usize;
        let num_blocks = b4_reserved * max_len / BLOCK_SIZE;
        let pool = PagedKvPool::new(
            num_blocks,
            LAYERS,
            KV_HEADS,
            HEAD_DIM,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(pool.num_blocks(), 256);
        assert_eq!(pool.kv_bytes(), b4_reserved * max_len * bytes_per_token);

        let reserved =
            Tensor::zeros((1, KV_HEADS, max_len, HEAD_DIM), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            reserved.elem_count() * std::mem::size_of::<f32>() * 2 * LAYERS,
            max_len * bytes_per_token
        );

        let lengths = [
            8usize, 16, 17, 24, 31, 32, 40, 48, 63, 64, 80, 96, 100, 127, 128, 160, 200, 255,
        ];
        let longest = lengths.iter().copied().max().unwrap();
        let mut used_blocks = 0usize;
        let mut fitted = 0usize;
        let mut exact_tokens = 0usize;
        let mut paged_tokens = 0usize;
        loop {
            let len = lengths[fitted % lengths.len()];
            let blocks = blocks_for_len(len);
            if used_blocks + blocks > pool.num_blocks() {
                break;
            }
            used_blocks += blocks;
            exact_tokens += len;
            paged_tokens += blocks * BLOCK_SIZE;
            fitted += 1;
        }
        let b4_at_max_len = pool.kv_bytes() / (max_len * bytes_per_token);
        let b4_at_longest = pool.kv_bytes() / (longest * bytes_per_token);
        let ratio_max_len = fitted as f64 / b4_at_max_len as f64;
        let ratio_longest = fitted as f64 / b4_at_longest as f64;
        eprintln!(
            "paged budget {} bytes, {} blocks, fitted {fitted} mixed sequences",
            pool.kv_bytes(),
            pool.num_blocks()
        );
        eprintln!("b4 reserve {max_len}: {b4_at_max_len} sequences, paged/b4 {ratio_max_len:.4}");
        eprintln!(
            "b4 reserve longest {longest}: {b4_at_longest} sequences, paged/b4 {ratio_longest:.4}"
        );
        eprintln!(
            "same {fitted} sequences: exact cat {} bytes, paged {} bytes, waste {} token slots",
            exact_tokens * bytes_per_token,
            paged_tokens * bytes_per_token,
            paged_tokens - exact_tokens
        );
        assert!(fitted >= b4_at_max_len * 2, "fitted {fitted}");
        assert!(fitted >= b4_at_longest * 2, "fitted {fitted} vs longest");
        assert!(paged_tokens - exact_tokens < BLOCK_SIZE * fitted);
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
