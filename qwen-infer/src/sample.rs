//! Temperature and nucleus sampling.
//!
//! `temperature == 0` is greedy argmax. Ties keep the lower id, the same
//! rule as [`crate::QwenModel::next_token`]. Anything warmer divides the
//! centered logits and draws once from `StdRng`. Nucleus (`top_p`) walks
//! that order until the cumulative probability reaches the cutoff, and
//! always keeps the first token: a tiny `top_p` is still a distribution.

use rand::Rng;
use rand::RngExt;

/// Why [`sample_token`] refused the arguments.
#[derive(Debug, Clone, PartialEq)]
pub enum SampleError {
    EmptyLogits,
    NegativeTemperature(f32),
    NonFiniteTemperature(f32),
    NegativeTopP(f32),
    NonFiniteTopP(f32),
    IdOverflow,
}

impl std::fmt::Display for SampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleError::EmptyLogits => write!(f, "no logits to sample"),
            SampleError::NegativeTemperature(value) => {
                write!(f, "temperature {value} is negative")
            }
            SampleError::NonFiniteTemperature(value) => {
                write!(f, "temperature {value} is not finite")
            }
            SampleError::NegativeTopP(value) => write!(f, "top_p {value} is negative"),
            SampleError::NonFiniteTopP(value) => write!(f, "top_p {value} is not finite"),
            SampleError::IdOverflow => write!(f, "token id does not fit in u32"),
        }
    }
}

impl std::error::Error for SampleError {}

/// Greedy id. Equal logits keep the lower index (`>` rather than `>=`).
pub fn greedy_token(logits: &[f32]) -> Result<u32, SampleError> {
    if logits.is_empty() {
        return Err(SampleError::EmptyLogits);
    }
    let mut best_at = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best {
            best = value;
            best_at = index;
        }
    }
    u32::try_from(best_at).map_err(|_| SampleError::IdOverflow)
}

/// Draw one token.
///
/// `temperature == 0` ignores `top_p` and returns [`greedy_token`]. `top_p >= 1`
/// keeps the whole vocabulary. `top_p` below the top token's mass keeps that
/// token anyway. Logits are shifted by their max before `exp`.
///
/// The candidate order is `sort_unstable_by` with [`f32::total_cmp`]. `f32`
/// does not implement `Ord` (NaN), which is E0277 if you sort the values
/// directly. Equal logits then fall through to the lower index, so the order
/// is total and the draw does not depend on the unstable sort's tie break.
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    rng: &mut impl Rng,
) -> Result<u32, SampleError> {
    if logits.is_empty() {
        return Err(SampleError::EmptyLogits);
    }
    if !temperature.is_finite() {
        return Err(SampleError::NonFiniteTemperature(temperature));
    }
    if temperature < 0.0 {
        return Err(SampleError::NegativeTemperature(temperature));
    }
    if !top_p.is_finite() {
        return Err(SampleError::NonFiniteTopP(top_p));
    }
    if top_p < 0.0 {
        return Err(SampleError::NegativeTopP(top_p));
    }
    if temperature == 0.0 {
        return greedy_token(logits);
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weight = Vec::with_capacity(logits.len());
    for &logit in logits {
        let shifted = (logit - max_logit) / temperature;
        weight.push(shifted.exp());
    }

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&i, &j| logits[j].total_cmp(&logits[i]).then(i.cmp(&j)));

    let cutoff = if top_p >= 1.0 {
        order.len()
    } else {
        let sum: f32 = weight.iter().sum();
        if sum == 0.0 || !sum.is_finite() {
            return greedy_token(logits);
        }
        let mut cumulative = 0.0;
        let mut keep = 1;
        for (k, &index) in order.iter().enumerate() {
            cumulative += weight[index] / sum;
            keep = k + 1;
            if cumulative >= top_p {
                break;
            }
        }
        keep
    };

    let kept = &order[..cutoff];
    let kept_sum: f32 = kept.iter().map(|&index| weight[index]).sum();
    if kept_sum == 0.0 || !kept_sum.is_finite() {
        return greedy_token(logits);
    }
    let mut threshold = rng.random::<f32>() * kept_sum;
    for &index in kept {
        let mass = weight[index];
        if threshold < mass {
            return u32::try_from(index).map_err(|_| SampleError::IdOverflow);
        }
        threshold -= mass;
    }
    let last = *kept.last().ok_or(SampleError::EmptyLogits)?;
    u32::try_from(last).map_err(|_| SampleError::IdOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{GenerateParams, generate};
    use crate::model::patterned_model;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn draw(logits: &[f32], temperature: f32, top_p: f32, seed: u64) -> u32 {
        let mut rng = StdRng::seed_from_u64(seed);
        sample_token(logits, temperature, top_p, &mut rng).unwrap()
    }

    fn sequence(logits: &[f32], temperature: f32, top_p: f32, seed: u64, n: usize) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| sample_token(logits, temperature, top_p, &mut rng).unwrap())
            .collect()
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits = [0.2, 1.0, 0.5, 0.7, 0.1, 0.9, 0.4, 0.8];
        let first = sequence(&logits, 1.0, 0.9, 7, 32);
        let again = sequence(&logits, 1.0, 0.9, 7, 32);
        assert_eq!(first, again);
        let other = sequence(&logits, 1.0, 0.9, 8, 32);
        assert_ne!(
            first, other,
            "two seeds on a non-degenerate distribution produced the same ids"
        );

        // The loop has to consume the rng. A peaked model can still tie two
        // seeds for a few tokens, so this is a longer greedy-temperature run.
        let model = patterned_model(false);
        let prompt = [1u32, 2, 3, 4];
        let mut warm = GenerateParams::greedy(24);
        warm.temperature = 1.1;
        warm.top_p = 1.0;
        warm.seed = 11;
        let bytes = |id: u32| Ok(vec![u8::try_from(id).unwrap_or(0)]);
        let a = generate(&model, &prompt, &warm, bytes).unwrap();
        let b = generate(&model, &prompt, &warm, bytes).unwrap();
        assert_eq!(a.tokens, b.tokens);
        warm.seed = 12;
        let c = generate(&model, &prompt, &warm, bytes).unwrap();
        assert_ne!(a.tokens, c.tokens);
    }

    #[test]
    fn top_p_1_temp_0_equals_greedy() {
        // Two ids share the max. `>` keeps the lower one.
        let logits = [0.2, 1.5, 1.5, -3.0, 0.0];
        assert_eq!(greedy_token(&logits).unwrap(), 1);
        assert_eq!(draw(&logits, 0.0, 1.0, 99), 1);
        assert_eq!(draw(&logits, 0.0, 1.0, 1), draw(&logits, 0.0, 1.0, 2));
        // A seed must not move the greedy choice, and top_p above 1 is the
        // full distribution, which at temperature 0 is still argmax.
        assert_eq!(draw(&logits, 0.0, 1.5, 4), 1);

        let model = patterned_model(false);
        let prompt = [1u32, 2, 3];
        let uncached = crate::greedy_tokens(&model, &prompt, 5).unwrap();
        let params = GenerateParams::greedy(5);
        let cached = generate(&model, &prompt, &params, |_| Ok(Vec::new())).unwrap();
        assert_eq!(cached.tokens, uncached);
        let mut other_seed = params;
        other_seed.seed = 99;
        let again = generate(&model, &prompt, &other_seed, |_| Ok(Vec::new())).unwrap();
        assert_eq!(again.tokens, uncached);
    }

    #[test]
    fn top_p_keeps_at_least_the_top_token_and_full_mass_keeps_everyone() {
        let logits = [0.0, 2.0, 0.5, -1.0];
        for seed in 0..30 {
            assert_eq!(draw(&logits, 1.0, 1e-6, seed), 1);
            assert_eq!(draw(&logits, 0.7, 0.0, seed), 1);
        }
        // Flat logits, top_p 1. Several seeds must not all collapse to one id.
        let flat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let wide: Vec<u32> = (0..40).map(|seed| draw(&flat, 1.0, 1.0, seed)).collect();
        let mut distinct = wide.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(
            distinct.len() > 1,
            "top_p 1 on flat logits only produced {wide:?}"
        );
        // top_p > 1 takes the same branch as 1, so the first draw matches.
        assert_eq!(draw(&logits, 0.8, 1.0, 3), draw(&logits, 0.8, 4.0, 3));
    }

    #[test]
    fn sample_rejects_empty_and_non_finite_params() {
        let mut rng = StdRng::seed_from_u64(1);
        assert!(matches!(
            sample_token(&[], 1.0, 1.0, &mut rng),
            Err(SampleError::EmptyLogits)
        ));
        assert!(matches!(
            sample_token(&[1.0], -0.1, 1.0, &mut rng),
            Err(SampleError::NegativeTemperature(_))
        ));
        assert!(matches!(
            sample_token(&[1.0], f32::NAN, 1.0, &mut rng),
            Err(SampleError::NonFiniteTemperature(_))
        ));
        assert!(matches!(
            sample_token(&[1.0], 1.0, -0.2, &mut rng),
            Err(SampleError::NegativeTopP(_))
        ));
        assert!(matches!(
            sample_token(&[1.0], f32::INFINITY, 1.0, &mut rng),
            Err(SampleError::NonFiniteTemperature(_))
        ));
    }
}
