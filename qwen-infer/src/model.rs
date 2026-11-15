//! Qwen2 decoder, written against candle-nn.
//!
//! `candle_transformers::models::qwen2` was the reference for the layer order
//! and the safetensors names. This file does not call it. Shapes below are
//! `(batch, seq, hidden)` unless a comment says otherwise. Qwen2.5-0.5B is
//! hidden 896, 24 layers, 14 query heads, 2 key/value heads, head dim 64.
//! Those numbers come from `config.json`, not from literals in the forward.
//!
//! [`QwenModel`] does not own a cache. [`QwenModel::last_logits`] recomputes
//! the whole sequence. [`QwenModel::forward`] takes a [`KvCache`] the caller
//! mutates. [`QwenModel::forward_decode_batch`] runs the one-token decode of
//! several of those caches in one matmul. Prefill stays on `forward`: prompts
//! have different lengths, and packing them is a separate problem.

use crate::paged::{BLOCK_SIZE, PagedCache, PagedKvPool, blocks_for_len};
use crate::sample::greedy_token;

use candle_core::{DType, Device, Result, Tensor, bail};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear, linear_no_bias, rms_norm,
};
use serde::Deserialize;
use std::path::Path;

/// Fields this forward reads from `config.json`. Unknown keys are ignored.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: String,
}

/// Contiguous per-layer K and V for one sequence.
///
/// `k` and `v` are one tensor per layer, shape `(1, kv_heads, seq, head_dim)`,
/// written after RoPE (V is not rotated). The layer vec starts empty and
/// grows with [`Tensor::cat`] on the sequence axis. Each step copies the
/// prefix, so a decode of length n copies O(n²) elements. That cost is the
/// reason a later milestone pages the cache instead of concatenating it.
///
/// The model is not inside this struct. A forward borrows the weights and
/// mutates only the cache it was handed.
pub struct KvCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    n_layers: usize,
}

impl KvCache {
    /// Empty cache for a model with `n_layers` blocks. `seq_len` is 0.
    pub fn new(n_layers: usize) -> Self {
        Self {
            k: Vec::with_capacity(n_layers),
            v: Vec::with_capacity(n_layers),
            n_layers,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.n_layers
    }

    /// Tokens already stored. Zero until the first forward returns.
    ///
    /// This reads layer 0. Every layer is appended in order, so a finished
    /// forward has the same length in each slot.
    pub fn seq_len(&self) -> usize {
        self.k.first().map(|tensor| tensor.dims()[2]).unwrap_or(0)
    }

    /// Store `k` and `v` for `layer`, or concatenate them onto what is there.
    ///
    /// Both tensors are made contiguous first. `cat` can copy a strided
    /// source, and the next matmul wants a contiguous K anyway.
    fn append(&mut self, layer: usize, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        if layer >= self.n_layers {
            bail!("kv cache layer {layer} is out of range ({})", self.n_layers);
        }
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        if self.k.len() == layer {
            self.k.push(k.clone());
            self.v.push(v.clone());
            return Ok((k, v));
        }
        if self.k.len() != self.n_layers || layer >= self.k.len() {
            bail!(
                "kv cache has {} of {} layers filled; cannot append layer {layer}",
                self.k.len(),
                self.n_layers
            );
        }
        let prev_k = self.k[layer].contiguous()?;
        let prev_v = self.v[layer].contiguous()?;
        let k = Tensor::cat(&[&prev_k, &k], 2)?;
        let v = Tensor::cat(&[&prev_v, &v], 2)?;
        self.k[layer] = k.clone();
        self.v[layer] = v.clone();
        Ok((k, v))
    }
}

/// `Some(window)` blocks keys more than `window` positions behind the query.
/// `None` is a plain causal mask. A key at the same position is allowed.
/// The blocked test is `key + window < query`, which is the inequality in
/// candle's Qwen2 mask (`j + sliding_window < i`). The 0.5B config sets
/// `use_sliding_window` to false, so generation passes `None`.
fn mask_value(query: usize, key: usize, window: Option<usize>) -> f32 {
    if key > query {
        return f32::NEG_INFINITY;
    }
    if let Some(window) = window
        && key + window < query
    {
        return f32::NEG_INFINITY;
    }
    0.0
}

/// Cos and sin for RoPE, shape `(max_seq, head_dim / 2)`.
///
/// `candle_nn::rotary_emb::rope` is the half-split form (the first half of
/// the head is rotated with the second half), which is what Qwen2 uses.
/// `rope_i` interleaves pairs and is a different model. `inv_freq` is
/// computed in f64 and then stored as f32, matching the candle reference:
/// `1 / theta^(i / dim)` for even `i`.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, head_dim: usize, dev: &Device) -> Result<Self> {
        let dim = head_dim;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| {
                let exponent = i as f64 / dim as f64;
                (1.0 / cfg.rope_theta.powf(exponent)) as f32
            })
            .collect();
        let freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, freq_len), dev)?.to_dtype(dtype)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(dtype)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// `xs` is `(batch, heads, seq, head_dim)` and must start at `pos`.
    fn apply(&self, xs: &Tensor, pos: usize) -> Result<Tensor> {
        let (_batch, _heads, seq, _dim) = xs.dims4()?;
        let cos = self.cos.narrow(0, pos, seq)?.contiguous()?;
        let sin = self.sin.narrow(0, pos, seq)?.contiguous()?;
        // rope() rejects a non-contiguous head axis. transpose() leaves one.
        candle_nn::rotary_emb::rope(&xs.contiguous()?, &cos, &sin)
    }

    /// RoPE for one new token per sequence, each at its own absolute position.
    ///
    /// `xs` is `(batch, heads, 1, head_dim)`. `positions[b]` is the absolute
    /// position of that row. Equal positions take the same path as [`Self::apply`]
    /// (one `(seq, freq)` table, shared across the batch). Different positions
    /// build a `(batch, 1, freq)` table. `candle_nn::rotary_emb::rope` treats a
    /// 3D cos/sin as per-batch: for one token the CPU kernel indexes
    /// `i_d + batch_index * (head_dim / 2)`. That is the layout `cat` produces
    /// here. Docs: <https://docs.rs/candle-nn/0.11.1/candle_nn/rotary_emb/fn.rope.html>
    /// (`rope_check_cs` accepts `(t, d)` or `(b, t, d)`).
    fn apply_positions(&self, xs: &Tensor, positions: &[usize]) -> Result<Tensor> {
        let (batch, _heads, seq, _dim) = xs.dims4()?;
        if seq != 1 {
            bail!("per-sequence RoPE is one token at a time, got seq {seq}");
        }
        if positions.len() != batch {
            bail!(
                "RoPE positions {} do not match batch {batch}",
                positions.len()
            );
        }
        if positions.is_empty() {
            bail!("no RoPE positions");
        }
        let table = self.cos.dim(0)?;
        for &pos in positions {
            if pos >= table {
                bail!("RoPE position {pos} is past the table ({table})");
            }
        }
        if positions.iter().all(|pos| *pos == positions[0]) {
            return self.apply(xs, positions[0]);
        }
        let freq = self.cos.dim(1)?;
        let mut cos_rows = Vec::with_capacity(batch);
        let mut sin_rows = Vec::with_capacity(batch);
        for &pos in positions {
            cos_rows.push(self.cos.narrow(0, pos, 1)?.contiguous()?);
            sin_rows.push(self.sin.narrow(0, pos, 1)?.contiguous()?);
        }
        let cos_refs: Vec<&Tensor> = cos_rows.iter().collect();
        let sin_refs: Vec<&Tensor> = sin_rows.iter().collect();
        let cos = Tensor::cat(&cos_refs, 0)?
            .reshape((batch, seq, freq))?
            .contiguous()?;
        let sin = Tensor::cat(&sin_refs, 0)?
            .reshape((batch, seq, freq))?
            .contiguous()?;
        candle_nn::rotary_emb::rope(&xs.contiguous()?, &cos, &sin)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inner = cfg.intermediate_size;
        // No bias on the SwiGLU projections. The safetensors file has none.
        Ok(Self {
            gate: linear_no_bias(hidden, inner, vb.pp("gate_proj"))?,
            up: linear_no_bias(hidden, inner, vb.pp("up_proj"))?,
            down: linear_no_bias(inner, hidden, vb.pp("down_proj"))?,
        })
    }

    /// `silu(gate(x)) * up(x)`, then `down`. SiLU is `x * sigmoid(x)`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = candle_nn::ops::silu(&self.gate.forward(xs)?)?;
        let up = self.up.forward(xs)?;
        self.down.forward(&(gated * up)?)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;
        // q/k/v have a bias. o_proj does not. Checked against the file header.
        Ok(Self {
            q_proj: linear(hidden, q_out, vb.pp("q_proj"))?,
            k_proj: linear(hidden, kv_out, vb.pp("k_proj"))?,
            v_proj: linear(hidden, kv_out, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(q_out, hidden, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        pos: usize,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = rope.apply(&q, pos)?;
        let k = rope.apply(&k, pos)?;
        // The cache keeps post-RoPE K. Decoding one new token must not rotate
        // the prefix a second time. V is not rotated either way.
        let (k, v) = if let Some(cache) = cache {
            cache.append(layer_idx, k, v)?
        } else {
            (k, v)
        };
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        // Scores are `(batch, heads, seq, seq)`. The scale is 1/sqrt(head_dim).
        // `affine` is the scalar multiply: a bare `f32 * Tensor` does not
        // implement `Mul` (that is the E0277 the roadmap mentions).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let scores = scores.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let mixed = weights.matmul(&v.contiguous()?)?;
        mixed
            .transpose(1, 2)?
            .reshape((batch, seq, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    /// One new token for every sequence in `caches`.
    ///
    /// `xs` is `(batch, 1, hidden)`. `positions[i]` is the absolute position
    /// of that token and the current [`KvCache::seq_len`] of `caches[i]`.
    /// K and V are rotated, appended to that sequence's own cache, then padded
    /// out to the longest cache in the batch so the attention matmul is one
    /// tensor. Padding keys are masked; they are not part of the softmax.
    fn forward_decode(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        caches: &mut [&mut KvCache],
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        if seq != 1 || batch != caches.len() || positions.len() != batch {
            bail!(
                "decode batch is (batch, 1) with one cache and one position each, got xs {:?}, {} caches, {} positions",
                xs.shape(),
                caches.len(),
                positions.len()
            );
        }
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = rope.apply_positions(&q, positions)?;
        let k = rope.apply_positions(&k, positions)?;
        // Post-RoPE K, same rule as the single-sequence path. Each cache stays
        // its own length. The stacked tensor below is only the attention view.
        for (index, cache) in caches.iter_mut().enumerate() {
            let k_i = k.narrow(0, index, 1)?;
            let v_i = v.narrow(0, index, 1)?;
            cache.append(layer_idx, k_i, v_i)?;
        }
        let k = stack_cached_kv(caches, layer_idx, true)?;
        let v = stack_cached_kv(caches, layer_idx, false)?;
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let scores = scores.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let mixed = weights.matmul(&v.contiguous()?)?;
        mixed
            .transpose(1, 2)?
            .reshape((batch, seq, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    /// One chunk whose K and V live in `pool`, addressed by `cache`'s block table.
    #[allow(clippy::too_many_arguments)]
    fn forward_paged(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        pos: usize,
        cache: &PagedCache,
        pool: &PagedKvPool,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (q, k, v) = self.project(xs)?;
        let q = rope.apply(&q, pos)?;
        let k = rope.apply(&k, pos)?;
        pool.write_layer(layer_idx, cache.block_table(), pos, &k, &v)?;
        let k = pool.gather_layer(layer_idx, cache.block_table(), true)?;
        let v = pool.gather_layer(layer_idx, cache.block_table(), false)?;
        self.attend(&q, &k, &v, mask)
    }

    /// One new token per sequence, written into that sequence's own blocks.
    #[allow(clippy::too_many_arguments)]
    fn forward_decode_paged(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        caches: &[&mut PagedCache],
        pool: &PagedKvPool,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        if seq != 1 || batch != caches.len() || positions.len() != batch {
            bail!(
                "paged decode batch is (batch, 1) with one cache and one position each, got xs {:?}, {} caches, {} positions",
                xs.shape(),
                caches.len(),
                positions.len()
            );
        }
        let (q, k, v) = self.project(xs)?;
        let q = rope.apply_positions(&q, positions)?;
        let k = rope.apply_positions(&k, positions)?;
        for (index, cache) in caches.iter().enumerate() {
            let k_i = k.narrow(0, index, 1)?;
            let v_i = v.narrow(0, index, 1)?;
            pool.write_layer(layer_idx, cache.block_table(), positions[index], &k_i, &v_i)?;
        }
        let mut keys = Vec::with_capacity(caches.len());
        let mut values = Vec::with_capacity(caches.len());
        for cache in caches {
            keys.push(pool.gather_layer(layer_idx, cache.block_table(), true)?);
            values.push(pool.gather_layer(layer_idx, cache.block_table(), false)?);
        }
        self.attend(&q, &stack_kv_batch(&keys)?, &stack_kv_batch(&values)?, mask)
    }

    /// Q, K, V in `(batch, heads, seq, head_dim)`. K and V still have `kv_heads`.
    fn project(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((batch, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        Ok((q, k, v))
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, _heads, seq, _) = q.dims4()?;
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, groups)?;
        let v = repeat_kv(v, groups)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let scores = scores.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let mixed = weights.matmul(&v.contiguous()?)?;
        mixed
            .transpose(1, 2)?
            .reshape((batch, seq, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

/// Pad `(1, heads, seq_i, dim)` on the sequence axis and stack on the batch axis.
fn stack_kv_batch(parts: &[Tensor]) -> Result<Tensor> {
    if parts.is_empty() {
        bail!("no kv tensors to stack");
    }
    let mut max_len = 0usize;
    for tensor in parts {
        max_len = max_len.max(tensor.dim(2)?);
    }
    let mut padded = Vec::with_capacity(parts.len());
    for tensor in parts {
        let len = tensor.dim(2)?;
        if len == max_len {
            padded.push(tensor.contiguous()?);
            continue;
        }
        let (batch, heads, _, dim) = tensor.dims4()?;
        let pad = Tensor::zeros(
            (batch, heads, max_len - len, dim),
            tensor.dtype(),
            tensor.device(),
        )?;
        padded.push(Tensor::cat(&[tensor, &pad], 2)?.contiguous()?);
    }
    let refs: Vec<&Tensor> = padded.iter().collect();
    Tensor::cat(&refs, 0)
}

/// Stack per-sequence K or V along the batch axis, padding the sequence axis
/// on the right with zeros out to the longest cache.
///
/// Each stored tensor is `(1, heads, seq_i, head_dim)`. The result is
/// `(batch, heads, max_seq, head_dim)`. Zeros are not a substitute for the
/// mask: the caller still writes `-inf` over `seq_i..max_seq`. The zeros only
/// give the matmul a rectangular layout.
fn stack_cached_kv(caches: &[&mut KvCache], layer: usize, keys: bool) -> Result<Tensor> {
    if caches.is_empty() {
        bail!("no caches to stack");
    }
    let mut max_len = 0usize;
    for cache in caches.iter() {
        let tensor = if keys {
            &cache.k[layer]
        } else {
            &cache.v[layer]
        };
        max_len = max_len.max(tensor.dim(2)?);
    }
    let mut parts = Vec::with_capacity(caches.len());
    for cache in caches.iter() {
        let tensor = if keys {
            &cache.k[layer]
        } else {
            &cache.v[layer]
        };
        let len = tensor.dim(2)?;
        if len == max_len {
            parts.push(tensor.contiguous()?);
            continue;
        }
        let (_batch, heads, _, dim) = tensor.dims4()?;
        let pad = Tensor::zeros(
            (1, heads, max_len - len, dim),
            tensor.dtype(),
            tensor.device(),
        )?;
        parts.push(Tensor::cat(&[tensor, &pad], 2)?.contiguous()?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 0)
}

/// Repeat each key/value head `n_rep` times along the head axis.
///
/// Input `(batch, kv_heads, seq, head_dim)`. Head `h` becomes heads
/// `h * n_rep .. (h + 1) * n_rep`, which is how grouped-query attention
/// lines the two KV heads up with the 14 query heads (repeat 7).
fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (batch, n_kv, seq, dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((batch, n_kv, n_rep, seq, dim))?
        .reshape((batch, n_kv * n_rep, seq, dim))?
        .contiguous()
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    post_norm: RmsNorm,
}

impl Layer {
    fn new(cfg: &Config, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, head_dim, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_norm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        pos: usize,
        cache: Option<&mut KvCache>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // `xs` is `&Tensor`. Copying the reference is not a copy of the
        // storage. `Tensor` itself is an `Arc` around that storage, so the
        // `+` below only clones the pointer when it needs an owned value.
        let residual = xs;
        let xs = self.input_norm.forward(xs)?;
        let xs = self.attn.forward(&xs, mask, rope, pos, cache, layer_idx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }

    fn forward_decode(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        caches: &mut [&mut KvCache],
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_norm.forward(xs)?;
        let xs = self
            .attn
            .forward_decode(&xs, mask, rope, positions, caches, layer_idx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_paged(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        pos: usize,
        cache: &PagedCache,
        pool: &PagedKvPool,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_norm.forward(xs)?;
        let xs = self
            .attn
            .forward_paged(&xs, mask, rope, pos, cache, pool, layer_idx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decode_paged(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        rope: &RotaryEmbedding,
        positions: &[usize],
        caches: &[&mut PagedCache],
        pool: &PagedKvPool,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_norm.forward(xs)?;
        let xs = self
            .attn
            .forward_decode_paged(&xs, mask, rope, positions, caches, pool, layer_idx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_norm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

enum Head {
    /// `lm_head.weight` is not in the file. Logits are `hidden @ embed^T`.
    Tied,
    Separate(Linear),
}

/// Weights plus the RoPE tables. Every forward takes `&self`: nothing here
/// is mutated. The token buffer and the [`KvCache`] live with the caller.
pub struct QwenModel {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rope: RotaryEmbedding,
    head: Head,
    cfg: Config,
    device: Device,
    dtype: DType,
}

impl QwenModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if cfg.hidden_act != "silu" {
            bail!("expected hidden_act silu, found {}", cfg.hidden_act);
        }
        if cfg.num_attention_heads == 0 || !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
        {
            bail!(
                "hidden_size {} does not divide into {} heads",
                cfg.hidden_size,
                cfg.num_attention_heads
            );
        }
        if cfg.num_key_value_heads == 0
            || !cfg
                .num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads)
        {
            bail!(
                "{} query heads is not a multiple of {} kv heads",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            );
        }
        if cfg.max_position_embeddings == 0 || cfg.num_hidden_layers == 0 {
            bail!("config has no layers or no positions");
        }
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        // `pp` borrows `vb` and returns a new builder. The mmap stays put.
        // Building the path as one string avoids holding two prefixes at once.
        let rope = RotaryEmbedding::new(vb.dtype(), cfg, head_dim, vb.device())?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            let layer = Layer::new(cfg, head_dim, vb.pp(format!("model.layers.{index}")))?;
            layers.push(layer);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let head = if cfg.tie_word_embeddings {
            Head::Tied
        } else {
            Head::Separate(linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?)
        };
        Ok(Self {
            embed,
            layers,
            norm,
            rope,
            head,
            cfg: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Map the safetensors file and build the model.
    ///
    /// The file on disk is bf16. `DType::F32` converts each tensor on read.
    /// bf16 on CPU is the slow path candle warns about, and the Python
    /// reference for the logit check is float32 as well.
    pub fn load(config_path: &Path, weights_path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(config_path)?;
        let cfg: Config = serde_json::from_reader(file)?;
        let device = Device::Cpu;
        // SAFETY: `from_mmaped_safetensors` is unsafe because the mapping
        // assumes the file is not truncated or rewritten while the tensors
        // exist. The caller only reads the hub cache.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
        Ok(Self::new(&cfg, vb)?)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `max_position_embeddings` from `config.json`.
    ///
    /// A request whose prompt plus `max_tokens` would walk past this is
    /// rejected before the engine thread sees it.
    pub fn max_position_embeddings(&self) -> usize {
        self.cfg.max_position_embeddings
    }

    pub fn head_dim(&self) -> usize {
        self.cfg.hidden_size / self.cfg.num_attention_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.cfg.num_key_value_heads
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Fixed KV pool for this model. `num_blocks` is shared by every sequence.
    pub fn kv_pool(&self, num_blocks: usize) -> Result<PagedKvPool> {
        PagedKvPool::new(
            num_blocks,
            self.num_layers(),
            self.num_kv_heads(),
            self.head_dim(),
            self.dtype,
            &self.device,
        )
    }

    /// Hidden states after the final RMSNorm, shape `(1, seq, hidden)`.
    ///
    /// No cache. The tokens are the whole context, and position `pos` is only
    /// the RoPE offset of the first id (the greedy path passes 0).
    fn forward_hidden(&self, ids: &[u32], pos: usize) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("no tokens");
        }
        self.check_pos(pos, ids.len())?;
        let input = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
        let mut xs = self.embed.forward(&input)?;
        let seq = ids.len();
        for (index, layer) in self.layers.iter().enumerate() {
            let window = self.window_for_layer(index);
            // Keys occupy the same positions as the queries. There is no
            // prefix outside this tensor.
            let mask = self.attention_mask(seq, seq, pos, pos, window)?;
            xs = layer.forward(&xs, &mask, &self.rope, pos, None, index)?;
        }
        self.norm.forward(&xs)
    }

    /// Same stack as [`Self::forward_hidden`], attending over `cache`.
    ///
    /// `ids` are the new tokens only. `pos` is the absolute position of
    /// `ids[0]` and must equal [`KvCache::seq_len`]: the cache is a prefix,
    /// not a window you can skip into. Prefill is `pos == 0` and `ids` the
    /// whole prompt. Decode is one id at `pos == cache.seq_len()`.
    fn forward_hidden_cached(
        &self,
        ids: &[u32],
        pos: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("no tokens");
        }
        if cache.num_layers() != self.layers.len() {
            bail!(
                "kv cache has {} layers, model has {}",
                cache.num_layers(),
                self.layers.len()
            );
        }
        if pos != cache.seq_len() {
            bail!(
                "pos {pos} does not match the kv cache length {}",
                cache.seq_len()
            );
        }
        self.check_pos(pos, ids.len())?;
        let input = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
        let mut xs = self.embed.forward(&input)?;
        let seq = ids.len();
        let kv_len = pos + seq;
        for (index, layer) in self.layers.iter().enumerate() {
            let window = self.window_for_layer(index);
            // Queries sit at `pos`. Keys are the cached prefix plus this
            // chunk, so they start at absolute position 0. A prefill with
            // `pos == 0` is the same mask as the uncached path. A one-token
            // decode (`seq == 1`) can see every key at `0..=pos` unless the
            // sliding window says otherwise.
            let mask = self.attention_mask(seq, kv_len, pos, 0, window)?;
            xs = layer.forward(&xs, &mask, &self.rope, pos, Some(cache), index)?;
        }
        self.norm.forward(&xs)
    }

    fn check_pos(&self, pos: usize, seq: usize) -> Result<()> {
        let end = pos.saturating_add(seq);
        if seq == 0 || end > self.cfg.max_position_embeddings {
            bail!(
                "positions {pos}..{end} exceed max_position_embeddings {}",
                self.cfg.max_position_embeddings
            );
        }
        Ok(())
    }

    /// `None` unless this layer is past `max_window_layers` and the config
    /// turned the sliding window on. Qwen2.5-0.5B has the flag off.
    fn window_for_layer(&self, index: usize) -> Option<usize> {
        if self.cfg.use_sliding_window && index >= self.cfg.max_window_layers {
            Some(self.cfg.sliding_window)
        } else {
            None
        }
    }

    /// Additive mask, shape `(1, 1, q_len, kv_len)`, so it broadcasts over heads.
    ///
    /// Query `q` is absolute position `query_pos + q`. Key `k` is absolute
    /// position `key_pos + k`. The uncached path passes the same origin for
    /// both, which is a square causal mask. The cached path passes
    /// `key_pos == 0` because the cache holds the prefix.
    fn attention_mask(
        &self,
        q_len: usize,
        kv_len: usize,
        query_pos: usize,
        key_pos: usize,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let mut values = Vec::with_capacity(q_len * kv_len);
        for q in 0..q_len {
            let query = query_pos + q;
            for k in 0..kv_len {
                let key = key_pos + k;
                values.push(mask_value(query, key, window));
            }
        }
        Tensor::from_vec(values, (1, 1, q_len, kv_len), &self.device)?.to_dtype(self.dtype)
    }

    /// Logits at the last position of a full recompute, shape `(vocab,)`.
    pub fn last_logits(&self, ids: &[u32]) -> Result<Tensor> {
        let hidden = self.forward_hidden(ids, 0)?;
        self.logits_from_last(&hidden)
    }

    /// Logits at the last position of `ids`, using `cache`, shape `(vocab,)`.
    ///
    /// `ids` is the chunk to run now, not the whole context. See
    /// [`Self::forward_hidden_cached`].
    pub fn forward(&self, ids: &[u32], pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let hidden = self.forward_hidden_cached(ids, pos, cache)?;
        self.logits_from_last(&hidden)
    }

    /// Decode one new token for each cache, in one forward.
    ///
    /// `ids[i]` is the token appended to `caches[i]`. Its RoPE position is
    /// that cache's length before the call, which is also the absolute
    /// position [`Self::forward`] would use. Caches may differ in length.
    /// They are padded on the sequence axis and the extra keys are masked
    /// with [`f32::NEG_INFINITY`]. The returned tensor is `(batch, vocab)`,
    /// one row of last-position logits per sequence.
    ///
    /// This does not prefill. A new sequence still runs [`Self::forward`] on
    /// its prompt alone. Doing that per sequence is the known inefficiency:
    /// the prompt matmul does not share a weight read with the decode batch
    /// or with the other prompts. Iteration-level scheduling still overlaps
    /// the *decode* steps, which is the bulk of a long generation.
    pub fn forward_decode_batch(&self, ids: &[u32], caches: &mut [&mut KvCache]) -> Result<Tensor> {
        if ids.is_empty() || ids.len() != caches.len() {
            bail!(
                "decode batch wants one token per cache, got {} tokens and {} caches",
                ids.len(),
                caches.len()
            );
        }
        let mut positions = Vec::with_capacity(ids.len());
        for cache in caches.iter() {
            if cache.num_layers() != self.layers.len() {
                bail!(
                    "kv cache has {} layers, model has {}",
                    cache.num_layers(),
                    self.layers.len()
                );
            }
            let pos = cache.seq_len();
            if pos == 0 {
                bail!("decode batch cache is empty; prefill with forward first");
            }
            self.check_pos(pos, 1)?;
            positions.push(pos);
        }
        let input = Tensor::from_slice(ids, (ids.len(), 1), &self.device)?;
        let mut xs = self.embed.forward(&input)?;
        // Length after this token is appended. Every layer appends one, so the
        // mask is the same shape throughout. `positions[i] + 1 == lengths[i]`.
        let lengths: Vec<usize> = positions.iter().map(|pos| pos + 1).collect();
        for (index, layer) in self.layers.iter().enumerate() {
            let window = self.window_for_layer(index);
            let mask = self.decode_batch_mask(&lengths, &positions, window)?;
            xs = layer.forward_decode(&xs, &mask, &self.rope, &positions, caches, index)?;
        }
        let hidden = self.norm.forward(&xs)?;
        self.logits_from_rows(&hidden)
    }

    /// [`Self::forward`] writing into a [`PagedKvPool`].
    ///
    /// `pos` must equal `cache.seq_len()`. The block table must already contain
    /// [`blocks_for_len`]`(pos + ids.len())` blocks. This does not allocate.
    /// Columns at and past the new length are masked. That covers the unused
    /// tail of the last block.
    pub fn forward_paged(
        &self,
        ids: &[u32],
        pos: usize,
        cache: &mut PagedCache,
        pool: &PagedKvPool,
    ) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("no tokens");
        }
        self.check_paged(cache, pool)?;
        if pos != cache.seq_len() {
            bail!(
                "pos {pos} does not match the paged cache length {}",
                cache.seq_len()
            );
        }
        self.check_pos(pos, ids.len())?;
        let end = pos + ids.len();
        let need = blocks_for_len(end);
        if cache.block_count() < need {
            bail!(
                "paged cache has {} blocks, positions {pos}..{end} need {need}",
                cache.block_count()
            );
        }
        let input = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
        let mut xs = self.embed.forward(&input)?;
        let seq = ids.len();
        let kv_len = cache.block_count() * BLOCK_SIZE;
        for (index, layer) in self.layers.iter().enumerate() {
            let window = self.window_for_layer(index);
            let mask = self.paged_mask(seq, kv_len, pos, end, window)?;
            xs = layer.forward_paged(&xs, &mask, &self.rope, pos, cache, pool, index)?;
        }
        cache.set_seq_len(end);
        let hidden = self.norm.forward(&xs)?;
        self.logits_from_last(&hidden)
    }

    /// [`Self::forward_decode_batch`] for paged caches that share `pool`.
    pub fn forward_decode_batch_paged(
        &self,
        ids: &[u32],
        caches: &mut [&mut PagedCache],
        pool: &PagedKvPool,
    ) -> Result<Tensor> {
        if ids.is_empty() || ids.len() != caches.len() {
            bail!(
                "paged decode batch wants one token per cache, got {} tokens and {} caches",
                ids.len(),
                caches.len()
            );
        }
        if pool.num_layers() != self.layers.len() || pool.block_size() != BLOCK_SIZE {
            bail!(
                "paged pool has {} layers and block size {}, model has {} layers",
                pool.num_layers(),
                pool.block_size(),
                self.layers.len()
            );
        }
        let mut positions = Vec::with_capacity(ids.len());
        for cache in caches.iter() {
            self.check_paged(cache, pool)?;
            let pos = cache.seq_len();
            if pos == 0 {
                bail!("paged decode cache is empty; prefill with forward_paged first");
            }
            self.check_pos(pos, 1)?;
            let need = blocks_for_len(pos + 1);
            if cache.block_count() < need {
                bail!(
                    "paged cache has {} blocks, decode at {pos} needs {need}",
                    cache.block_count()
                );
            }
            positions.push(pos);
        }
        let input = Tensor::from_slice(ids, (ids.len(), 1), &self.device)?;
        let mut xs = self.embed.forward(&input)?;
        let lengths: Vec<usize> = positions.iter().map(|pos| pos + 1).collect();
        for (index, layer) in self.layers.iter().enumerate() {
            let window = self.window_for_layer(index);
            let max_kv = caches
                .iter()
                .map(|cache| cache.block_count() * BLOCK_SIZE)
                .max()
                .unwrap_or(0);
            let mask = self.paged_decode_mask(&lengths, &positions, max_kv, window)?;
            xs = layer
                .forward_decode_paged(&xs, &mask, &self.rope, &positions, caches, pool, index)?;
        }
        for cache in caches.iter_mut() {
            let len = cache.seq_len() + 1;
            cache.set_seq_len(len);
        }
        let hidden = self.norm.forward(&xs)?;
        self.logits_from_rows(&hidden)
    }

    fn check_paged(&self, cache: &PagedCache, pool: &PagedKvPool) -> Result<()> {
        if cache.num_layers() != self.layers.len() || pool.num_layers() != self.layers.len() {
            bail!(
                "paged cache has {} layers, pool has {}, model has {}",
                cache.num_layers(),
                pool.num_layers(),
                self.layers.len()
            );
        }
        Ok(())
    }

    fn paged_decode_mask(
        &self,
        lengths: &[usize],
        positions: &[usize],
        kv_len: usize,
        window: Option<usize>,
    ) -> Result<Tensor> {
        if kv_len == 0 || lengths.len() != positions.len() {
            bail!("paged decode mask has no keys");
        }
        let mut values = Vec::with_capacity(lengths.len() * kv_len);
        for (b, &len) in lengths.iter().enumerate() {
            let query = positions[b];
            for key in 0..kv_len {
                let value = if key >= len {
                    f32::NEG_INFINITY
                } else {
                    mask_value(query, key, window)
                };
                values.push(value);
            }
        }
        Tensor::from_vec(values, (lengths.len(), 1, 1, kv_len), &self.device)?.to_dtype(self.dtype)
    }

    fn paged_mask(
        &self,
        q_len: usize,
        kv_len: usize,
        query_pos: usize,
        valid_len: usize,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let mut values = Vec::with_capacity(q_len * kv_len);
        for q in 0..q_len {
            let query = query_pos + q;
            for key in 0..kv_len {
                let value = if key >= valid_len {
                    f32::NEG_INFINITY
                } else {
                    mask_value(query, key, window)
                };
                values.push(value);
            }
        }
        Tensor::from_vec(values, (1, 1, q_len, kv_len), &self.device)?.to_dtype(self.dtype)
    }

    /// Additive mask `(batch, 1, 1, max_kv)` for a one-token decode batch.
    ///
    /// Query `b` sits at absolute position `positions[b]`. Real keys are
    /// `0..lengths[b]` (the cache prefix plus the token just appended). Indexes
    /// at or past `lengths[b]` are padding and get [`f32::NEG_INFINITY`], which
    /// softmax turns into a zero weight. Real keys use [`mask_value`], so a
    /// sliding window blocks the far past the same way the single-sequence
    /// mask does.
    fn decode_batch_mask(
        &self,
        lengths: &[usize],
        positions: &[usize],
        window: Option<usize>,
    ) -> Result<Tensor> {
        let max_len = lengths.iter().copied().max().unwrap_or(0);
        if max_len == 0 || lengths.len() != positions.len() {
            bail!("decode batch mask has no keys");
        }
        let mut values = Vec::with_capacity(lengths.len() * max_len);
        for (b, &len) in lengths.iter().enumerate() {
            let query = positions[b];
            for key in 0..max_len {
                let value = if key >= len {
                    f32::NEG_INFINITY
                } else {
                    mask_value(query, key, window)
                };
                values.push(value);
            }
        }
        Tensor::from_vec(values, (lengths.len(), 1, 1, max_len), &self.device)?.to_dtype(self.dtype)
    }

    /// Last-position logits for every row, shape `(batch, vocab)`.
    fn logits_from_rows(&self, hidden: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = hidden.dims3()?;
        let last = hidden
            .narrow(1, seq - 1, 1)?
            .reshape((batch, self.cfg.hidden_size))?;
        match &self.head {
            Head::Tied => {
                let weight = self.embed.embeddings().t()?;
                last.matmul(&weight)
            }
            Head::Separate(head) => last.apply(head),
        }
    }

    fn logits_from_last(&self, hidden: &Tensor) -> Result<Tensor> {
        let seq = hidden.dim(1)?;
        // `(1, seq, hidden)` -> `(1, hidden)`. candle's `matmul` does not
        // contract a 3D left-hand side with a 2D weight (the error is
        // `lhs: [1, 1, hidden], rhs: [hidden, vocab]`). Drop the seq axis
        // first. The weight transpose is a view of the embedding matrix.
        let last = hidden
            .narrow(1, seq - 1, 1)?
            .reshape((1, self.cfg.hidden_size))?;
        let logits = match &self.head {
            Head::Tied => {
                let weight = self.embed.embeddings().t()?;
                last.matmul(&weight)?
            }
            Head::Separate(head) => last.apply(head)?,
        };
        let vocab = self.cfg.vocab_size;
        logits.reshape((vocab,))
    }

    /// Greedy id for the next position, recomputing the whole sequence.
    /// Equal logits keep the lower index.
    pub fn next_token(&self, ids: &[u32]) -> Result<u32> {
        let logits = self.last_logits(ids)?;
        let values = logits.to_vec1::<f32>()?;
        greedy_token(&values).map_err(|err| candle_core::Error::Msg(err.to_string()))
    }
}

/// Two-layer model with a fixed pattern in every tensor. Zeros hide a wrong
/// mask, because V is zero and the attention output is zero either way.
///
/// `max_position_embeddings` is 32 here. [`tiny_model`] is the same net with
/// a longer table, for a chat-templated prompt in the server tests.
#[cfg(test)]
pub(crate) fn patterned_model(sliding: bool) -> QwenModel {
    build_tiny(sliding, 32)
}

/// Patterned two-layer net for protocol tests. Not a Qwen checkpoint.
///
/// Hidden 16, vocab 32, `max_position_embeddings` 2048. Token ids have to
/// be below 32. The server tests encode a chat string into that range.
#[doc(hidden)]
pub fn tiny_model(sliding: bool) -> QwenModel {
    build_tiny(sliding, 2048)
}

fn build_tiny(sliding: bool, max_position_embeddings: usize) -> QwenModel {
    let cfg = Config {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings,
        sliding_window: 3,
        max_window_layers: 0,
        tie_word_embeddings: true,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: sliding,
        hidden_act: "silu".into(),
    };
    let device = Device::Cpu;
    let mut tensors = std::collections::HashMap::new();
    {
        let mut put = |name: String, shape: Vec<usize>, norm: bool| {
            let n: usize = shape.iter().product();
            let salt = name
                .bytes()
                .fold(0.2f32, |acc, byte| acc + f32::from(byte) * 0.001);
            let values: Vec<f32> = (0..n)
                .map(|i| {
                    let wave = (i as f32 * 0.173 + salt).sin();
                    if norm { 1.0 + 0.02 * wave } else { 0.05 * wave }
                })
                .collect();
            tensors.insert(
                name,
                Tensor::from_vec(values, shape.as_slice(), &device).expect("pattern tensor"),
            );
        };
        put(
            "model.embed_tokens.weight".to_string(),
            vec![cfg.vocab_size, cfg.hidden_size],
            false,
        );
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;
        for layer in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            put(
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![q_out, cfg.hidden_size],
                false,
            );
            put(
                format!("{prefix}.self_attn.q_proj.bias"),
                vec![q_out],
                false,
            );
            put(
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_out, cfg.hidden_size],
                false,
            );
            put(
                format!("{prefix}.self_attn.k_proj.bias"),
                vec![kv_out],
                false,
            );
            put(
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv_out, cfg.hidden_size],
                false,
            );
            put(
                format!("{prefix}.self_attn.v_proj.bias"),
                vec![kv_out],
                false,
            );
            put(
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![cfg.hidden_size, q_out],
                false,
            );
            put(
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![cfg.intermediate_size, cfg.hidden_size],
                false,
            );
            put(
                format!("{prefix}.mlp.up_proj.weight"),
                vec![cfg.intermediate_size, cfg.hidden_size],
                false,
            );
            put(
                format!("{prefix}.mlp.down_proj.weight"),
                vec![cfg.hidden_size, cfg.intermediate_size],
                false,
            );
            put(
                format!("{prefix}.input_layernorm.weight"),
                vec![cfg.hidden_size],
                true,
            );
            put(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![cfg.hidden_size],
                true,
            );
        }
        put("model.norm.weight".to_string(), vec![cfg.hidden_size], true);
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    QwenModel::new(&cfg, vb).expect("patterned model")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_blocks_the_future_and_a_window_blocks_the_far_past() {
        assert_eq!(mask_value(0, 0, None), 0.0);
        assert_eq!(mask_value(2, 1, None), 0.0);
        assert_eq!(mask_value(1, 2, None), f32::NEG_INFINITY);
        // Distance 5 with window 4 is blocked. Distance 4 is kept.
        assert_eq!(mask_value(5, 0, Some(4)), f32::NEG_INFINITY);
        assert_eq!(mask_value(5, 1, Some(4)), 0.0);
        assert_eq!(mask_value(5, 5, Some(4)), 0.0);
        assert_eq!(mask_value(5, 6, Some(4)), f32::NEG_INFINITY);
    }

    fn tiny_config(sliding: bool) -> Config {
        Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 8,
            sliding_window: 2,
            max_window_layers: 0,
            tie_word_embeddings: true,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: sliding,
            hidden_act: "silu".into(),
        }
    }

    #[test]
    fn zero_weights_forward_returns_finite_vocab_logits() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = QwenModel::new(&tiny_config(false), vb).unwrap();
        let logits = model.last_logits(&[1, 2, 3]).unwrap();
        assert_eq!(logits.dims(), &[32]);
        let values = logits.to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite()), "{values:?}");
        // Every logit is the same only because every weight is zero. The
        // argmax of a tie is the first id.
        assert_eq!(model.next_token(&[1, 2, 3]).unwrap(), 0);
    }

    #[test]
    fn sliding_window_flag_is_read_from_the_config() {
        let device = Device::Cpu;
        let off =
            QwenModel::new(&tiny_config(false), VarBuilder::zeros(DType::F32, &device)).unwrap();
        let on =
            QwenModel::new(&tiny_config(true), VarBuilder::zeros(DType::F32, &device)).unwrap();
        assert_eq!(off.window_for_layer(0), None);
        // max_window_layers is 0, so layer 0 is inside the window when the
        // flag is on. The 0.5B file has the flag off and max_window_layers 21.
        assert_eq!(on.window_for_layer(0), Some(2));
    }

    fn max_abs(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn cached_equals_uncached() {
        for sliding in [false, true] {
            let model = patterned_model(sliding);
            let mut ids = vec![1u32, 4, 7, 2, 9];
            let extra = [3u32, 8, 1, 6];
            let mut cache = KvCache::new(model.num_layers());
            let mut worst = 0.0f32;

            let cached = model
                .forward(&ids, 0, &mut cache)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let full = model.last_logits(&ids).unwrap().to_vec1::<f32>().unwrap();
            let diff = max_abs(&cached, &full);
            assert!(diff <= 1e-4, "prefill sliding {sliding} max abs {diff}");
            worst = worst.max(diff);

            for (step, id) in extra.iter().copied().enumerate() {
                ids.push(id);
                let pos = ids.len() - 1;
                let cached = model
                    .forward(&[id], pos, &mut cache)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let full = model.last_logits(&ids).unwrap().to_vec1::<f32>().unwrap();
                let diff = max_abs(&cached, &full);
                assert!(
                    diff <= 1e-4,
                    "sliding {sliding} decode step {step} max abs {diff}"
                );
                worst = worst.max(diff);
                assert_eq!(cache.seq_len(), ids.len());
            }
            eprintln!("cached_equals_uncached sliding={sliding} max abs {worst}");

            // Second chunk starts at pos > 0. Its last logit is the logit of
            // the full prompt, and one more decode step still matches.
            let prompt = [1u32, 4, 7, 2, 9];
            let mut cache = KvCache::new(model.num_layers());
            model.forward(&prompt[..2], 0, &mut cache).unwrap();
            let cached = model
                .forward(&prompt[2..], 2, &mut cache)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let full = model
                .last_logits(&prompt)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff = max_abs(&cached, &full);
            assert!(
                diff <= 1e-4,
                "chunked prefill sliding {sliding} max abs {diff}"
            );
            let mut extended = prompt.to_vec();
            extended.push(extra[0]);
            let cached = model
                .forward(&[extra[0]], prompt.len(), &mut cache)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let full = model
                .last_logits(&extended)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff = max_abs(&cached, &full);
            assert!(
                diff <= 1e-4,
                "chunked decode sliding {sliding} max abs {diff}"
            );
        }
    }

    /// Eight prompts, different lengths, greedy. Each row of the decode batch
    /// matches that prompt run alone, including under the sliding window.
    #[test]
    fn batched_outputs_equal_solo_outputs() {
        let prompts: Vec<Vec<u32>> = vec![
            vec![1],
            vec![2, 3],
            vec![4, 5, 6],
            vec![7, 8, 1, 2],
            vec![3, 3, 3, 3, 3],
            vec![9, 1],
            vec![2, 2, 4, 6, 8, 1],
            vec![5],
        ];
        let steps = 4usize;
        for sliding in [false, true] {
            let model = patterned_model(sliding);
            let mut worst = 0.0f32;
            let mut solo_cache = Vec::new();
            let mut solo_logits = Vec::new();
            let mut solo_ids = Vec::new();
            let mut batch_cache = Vec::new();
            let mut batch_logits = Vec::new();
            for prompt in &prompts {
                let mut cache = KvCache::new(model.num_layers());
                let logits = model
                    .forward(prompt, 0, &mut cache)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                solo_cache.push(cache);
                solo_logits.push(logits);
                solo_ids.push(prompt.clone());

                let mut cache = KvCache::new(model.num_layers());
                let logits = model
                    .forward(prompt, 0, &mut cache)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                batch_cache.push(cache);
                batch_logits.push(logits);
            }
            for (i, (batch, solo)) in batch_logits.iter().zip(&solo_logits).enumerate() {
                let diff = max_abs(batch, solo);
                assert!(
                    diff <= 1e-4,
                    "prefill seq {i} sliding {sliding} max abs {diff}"
                );
                worst = worst.max(diff);
            }

            let mut generated = vec![Vec::new(); prompts.len()];
            for step in 0..steps {
                let mut next = Vec::with_capacity(prompts.len());
                for i in 0..prompts.len() {
                    let id = crate::greedy_token(&batch_logits[i]).unwrap();
                    assert_eq!(
                        id,
                        crate::greedy_token(&solo_logits[i]).unwrap(),
                        "sliding {sliding} step {step} seq {i}"
                    );
                    generated[i].push(id);
                    solo_ids[i].push(id);
                    next.push(id);
                }
                if step + 1 == steps {
                    break;
                }
                let mut refs: Vec<&mut KvCache> = batch_cache.iter_mut().collect();
                let rows = model
                    .forward_decode_batch(&next, &mut refs)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                drop(refs);
                for i in 0..prompts.len() {
                    let solo = model
                        .forward(&[next[i]], solo_cache[i].seq_len(), &mut solo_cache[i])
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let full = model
                        .last_logits(&solo_ids[i])
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let diff = max_abs(&rows[i], &solo);
                    let diff_full = max_abs(&solo, &full);
                    assert!(
                        diff <= 1e-4,
                        "sliding {sliding} decode step {step} seq {i} batch vs solo {diff}"
                    );
                    assert!(
                        diff_full <= 1e-4,
                        "sliding {sliding} decode step {step} seq {i} solo vs full {diff_full}"
                    );
                    worst = worst.max(diff).max(diff_full);
                    batch_logits[i] = rows[i].clone();
                    solo_logits[i] = solo;
                    assert_eq!(batch_cache[i].seq_len(), solo_cache[i].seq_len());
                }
            }
            let params = crate::GenerateParams::greedy(steps);
            for (i, prompt) in prompts.iter().enumerate() {
                let alone = crate::generate(&model, prompt, &params, |_| Ok(Vec::new())).unwrap();
                assert_eq!(
                    generated[i], alone.tokens,
                    "sliding {sliding} seq {i} vs generate"
                );
            }
            eprintln!("batched_outputs_equal_solo_outputs sliding={sliding} max abs {worst}");
        }
    }

    /// After one joint step, one sequence leaves. The other two stay on the
    /// path they would have walked alone. This is the mask when the batch
    /// changes shape between iterations.
    #[test]
    fn leaving_the_batch_does_not_change_the_sequences_that_stay() {
        let model = patterned_model(true);
        let prompts = [vec![1u32, 2, 3, 4, 5, 6], vec![7, 8], vec![9, 3, 1]];
        let mut solo_cache = Vec::new();
        let mut solo_logits = Vec::new();
        let mut batch_cache = Vec::new();
        let mut batch_logits = Vec::new();
        for prompt in &prompts {
            let mut cache = KvCache::new(model.num_layers());
            solo_logits.push(
                model
                    .forward(prompt, 0, &mut cache)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
            solo_cache.push(cache);
            let mut cache = KvCache::new(model.num_layers());
            batch_logits.push(
                model
                    .forward(prompt, 0, &mut cache)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
            batch_cache.push(cache);
        }
        let first: Vec<u32> = batch_logits
            .iter()
            .map(|row| crate::greedy_token(row).unwrap())
            .collect();
        let mut refs: Vec<&mut KvCache> = batch_cache.iter_mut().collect();
        let rows = model
            .forward_decode_batch(&first, &mut refs)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        drop(refs);
        for i in 0..prompts.len() {
            let solo = model
                .forward(&[first[i]], solo_cache[i].seq_len(), &mut solo_cache[i])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff = max_abs(&rows[i], &solo);
            assert!(diff <= 1e-4, "joint step seq {i} max abs {diff}");
            solo_logits[i] = solo;
            batch_logits[i] = rows[i].clone();
        }
        // Sequence 1 is finished. The next iteration is the other two only.
        let stay = [0usize, 2];
        let next: Vec<u32> = stay
            .iter()
            .map(|&i| crate::greedy_token(&batch_logits[i]).unwrap())
            .collect();
        let mut kept: Vec<&mut KvCache> = batch_cache
            .iter_mut()
            .enumerate()
            .filter(|(index, _)| stay.contains(index))
            .map(|(_, cache)| cache)
            .collect();
        // `filter` keeps source order, which is `stay` (0 then 2), so `kept`
        // lines up with `next`.
        let rows = model
            .forward_decode_batch(&next, &mut kept)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for ((row, &index), &id) in rows.iter().zip(&stay).zip(&next) {
            let solo = model
                .forward(&[id], solo_cache[index].seq_len(), &mut solo_cache[index])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff = max_abs(row, &solo);
            assert!(
                diff <= 1e-4,
                "after drop, seq {index} diverged from solo by {diff}"
            );
        }
    }

    /// Mixed lengths, including a prompt that ends on a block boundary and one
    /// that ends one token past it. Paged logits match the contiguous cache.
    /// A second chunk starts in the middle of a block.
    #[test]
    fn paged_equals_contiguous() {
        let prompts: Vec<Vec<u32>> = vec![
            vec![1],
            vec![2, 3, 4],
            vec![5; 15],
            vec![6; 16],
            vec![7; 17],
            vec![8; 31],
            vec![9; 32],
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        ];
        let steps = 4usize;
        for sliding in [false, true] {
            // `patterned_model` stops at position 32. A 32-token prompt plus
            // a decode step needs the longer table `tiny_model` builds.
            let model = tiny_model(sliding);
            let pool = model.kv_pool(64).unwrap();
            let mut hold = Hold {
                alloc: crate::paged::BlockAllocator::new(64),
                caches: Vec::new(),
            };
            let mut contig = Vec::new();
            let mut rows = Vec::new();
            let mut worst = 0.0f32;
            for prompt in &prompts {
                let mut paged = PagedCache::new(model.num_layers());
                paged.assign_blocks(hold.alloc.alloc_n(blocks_for_len(prompt.len())).unwrap());
                let mut plain = KvCache::new(model.num_layers());
                let paged_row = model
                    .forward_paged(prompt, 0, &mut paged, &pool)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let plain_row = model
                    .forward(prompt, 0, &mut plain)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let diff = max_abs(&paged_row, &plain_row);
                assert!(
                    diff <= 1e-4,
                    "prefill len {} sliding {sliding} max abs {diff}",
                    prompt.len()
                );
                worst = worst.max(diff);
                hold.caches.push(paged);
                contig.push(plain);
                rows.push(plain_row);
            }
            for step in 0..steps - 1 {
                let next: Vec<u32> = rows
                    .iter()
                    .map(|row| crate::greedy_token(row).unwrap())
                    .collect();
                for cache in hold.caches.iter_mut() {
                    let need = blocks_for_len(cache.seq_len() + 1);
                    if cache.block_count() < need {
                        let extra = hold.alloc.alloc_n(need - cache.block_count()).unwrap();
                        cache.assign_blocks(extra);
                    }
                }
                let mut refs: Vec<&mut PagedCache> = hold.caches.iter_mut().collect();
                let paged_rows = model
                    .forward_decode_batch_paged(&next, &mut refs, &pool)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                drop(refs);
                let mut plain_refs: Vec<&mut KvCache> = contig.iter_mut().collect();
                let plain_rows = model
                    .forward_decode_batch(&next, &mut plain_refs)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                for index in 0..prompts.len() {
                    let diff = max_abs(&paged_rows[index], &plain_rows[index]);
                    assert!(
                        diff <= 1e-4,
                        "sliding {sliding} decode step {step} seq {index} len {} max abs {diff}",
                        prompts[index].len()
                    );
                    worst = worst.max(diff);
                    rows[index] = plain_rows[index].clone();
                }
            }

            // Token 10 is the middle of a 16-token block. The second forward
            // continues that block and spills into the next one.
            let prompt: Vec<u32> = (1..21).collect();
            let mut paged = PagedCache::new(model.num_layers());
            paged.assign_blocks(hold.alloc.alloc_n(blocks_for_len(prompt.len())).unwrap());
            let mut plain = KvCache::new(model.num_layers());
            model
                .forward_paged(&prompt[..10], 0, &mut paged, &pool)
                .unwrap();
            model.forward(&prompt[..10], 0, &mut plain).unwrap();
            let paged_row = model
                .forward_paged(&prompt[10..], 10, &mut paged, &pool)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let plain_row = model
                .forward(&prompt[10..], 10, &mut plain)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let full = model
                .last_logits(&prompt)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let diff = max_abs(&paged_row, &plain_row);
            let diff_full = max_abs(&paged_row, &full);
            assert!(
                diff <= 1e-4,
                "chunked paged vs contiguous sliding {sliding} max abs {diff}"
            );
            assert!(
                diff_full <= 1e-4,
                "chunked paged vs full sliding {sliding} max abs {diff_full}"
            );
            worst = worst.max(diff).max(diff_full);
            hold.caches.push(paged);
            eprintln!("paged_equals_contiguous sliding={sliding} max abs {worst}");
        }
    }

    /// Returns blocks before the caches drop, including when an assertion panics.
    struct Hold {
        alloc: crate::paged::BlockAllocator,
        caches: Vec<PagedCache>,
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            for cache in &mut self.caches {
                for id in cache.take_blocks() {
                    self.alloc.free_block(id).expect("free block");
                }
            }
        }
    }
}
