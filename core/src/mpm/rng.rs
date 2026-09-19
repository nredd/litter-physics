//! Minimal deterministic pseudo-random generator (`SplitMix64`).
//!
//! Used only for particle placement jitter so that identical seeds reproduce
//! identical fixtures. The full generator state is a single `u64` that is stored
//! in checkpoints.
//!
//! References: <https://doi.org/10.1145/2714064.2660195> (Steele, Lea, Flood 2014)

/// `SplitMix64` generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from a seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Current state, for checkpointing.
    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// Restore a generator from a checkpointed state.
    #[must_use]
    pub const fn from_state(state: u64) -> Self {
        Self { state }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)` with 53 bits of resolution.
    #[allow(clippy::cast_precision_loss)] // 53-bit mantissa is exact by construction.
    pub fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact values by construction.
mod tests {
    use super::SplitMix64;

    #[test]
    fn is_deterministic_and_bounded() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..1000 {
            let x = a.next_unit();
            assert_eq!(x, b.next_unit());
            assert!((0.0..1.0).contains(&x));
        }
        assert_eq!(a.state(), b.state());
        let restored = SplitMix64::from_state(a.state());
        assert_eq!(restored, a);
    }
}
