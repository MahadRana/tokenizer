//! Qwen2 decoder, written against candle-nn.
//!
//! `candle_transformers::models::qwen2` was the reference for the layer order
//! and the safetensors names. This file does not call it. Shapes below are
//! `(batch, seq, hidden)` unless a comment says otherwise. Qwen2.5-0.5B is
//! hidden 896, 24 layers, 14 query heads, 2 key/value heads, head dim 64.
//! Those numbers come from `config.json`, not from literals in the forward.

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
    ) -> Result<Tensor> {
        // `xs` is `&Tensor`. Copying the reference is not a copy of the
        // storage. `Tensor` itself is an `Arc` around that storage, so the
        // `+` below only clones the pointer when it needs an owned value.
        let residual = xs;
        let xs = self.input_norm.forward(xs)?;
        let xs = self.attn.forward(&xs, mask, rope, pos)?;
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

/// Weights plus the RoPE tables. Forward takes `&self`: nothing here is
/// mutated. The token buffer lives with the caller.
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

    /// Hidden states after the final RMSNorm, shape `(1, seq, hidden)`.
    fn forward_hidden(&self, ids: &[u32], pos: usize) -> Result<Tensor> {
        if ids.is_empty() {
            bail!("no tokens");
        }
        let input = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
        let mut xs = self.embed.forward(&input)?;
        let seq = ids.len();
        for (index, layer) in self.layers.iter().enumerate() {
            let window = self.window_for_layer(index);
            let mask = self.attention_mask(seq, window)?;
            xs = layer.forward(&xs, &mask, &self.rope, pos)?;
        }
        self.norm.forward(&xs)
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

    /// Additive mask, shape `(1, 1, seq, seq)`, so it broadcasts over heads.
    fn attention_mask(&self, seq: usize, window: Option<usize>) -> Result<Tensor> {
        let mut values = Vec::with_capacity(seq * seq);
        for query in 0..seq {
            for key in 0..seq {
                values.push(mask_value(query, key, window));
            }
        }
        Tensor::from_vec(values, (1, 1, seq, seq), &self.device)?.to_dtype(self.dtype)
    }

    /// Logits at the last position, shape `(vocab,)`.
    pub fn last_logits(&self, ids: &[u32]) -> Result<Tensor> {
        let hidden = self.forward_hidden(ids, 0)?;
        let seq = ids.len();
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

    /// Greedy id for the next position. Equal logits keep the lower index.
    pub fn next_token(&self, ids: &[u32]) -> Result<u32> {
        let logits = self.last_logits(ids)?;
        let values = logits.to_vec1::<f32>()?;
        let mut best_at = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (index, value) in values.iter().copied().enumerate() {
            if value > best {
                best = value;
                best_at = index;
            }
        }
        u32::try_from(best_at).map_err(|_| candle_core::Error::Msg("token id overflow".to_string()))
    }
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
}
