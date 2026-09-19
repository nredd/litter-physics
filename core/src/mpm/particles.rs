//! Material point storage (structure of arrays).
//!
//! Particles carry a stable `id` so that output frames keep identity across the
//! periodic cache-locality sort performed by the solver.

use nalgebra::{Matrix3, Vector3};

/// Structure-of-arrays particle storage for a single material.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticleSet {
    /// Stable particle identifiers (unique, dense from zero at creation).
    pub id: Vec<u32>,
    /// Positions in m.
    pub position: Vec<Vector3<f64>>,
    /// Velocities in m/s.
    pub velocity: Vec<Vector3<f64>>,
    /// APIC affine velocity matrix `C` in 1/s.
    pub affine: Vec<Matrix3<f64>>,
    /// Elastic deformation gradient (paste). Identity for liquids.
    pub deformation: Vec<Matrix3<f64>>,
    /// Volume ratio `J` (liquid). One for paste, where `det(F)` carries it.
    pub volume_ratio: Vec<f64>,
    /// Kirchhoff stress `P F^T` from the last constitutive update, in Pa.
    pub kirchhoff: Vec<Matrix3<f64>>,
    /// Accumulated equivalent plastic strain (dimensionless).
    pub plastic_strain: Vec<f64>,
    /// Particle mass in kg (constant).
    pub mass: Vec<f64>,
    /// Reference volume in m^3 (constant).
    pub volume0: Vec<f64>,
}

impl ParticleSet {
    /// Create an empty set with reserved capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            id: Vec::with_capacity(capacity),
            position: Vec::with_capacity(capacity),
            velocity: Vec::with_capacity(capacity),
            affine: Vec::with_capacity(capacity),
            deformation: Vec::with_capacity(capacity),
            volume_ratio: Vec::with_capacity(capacity),
            kirchhoff: Vec::with_capacity(capacity),
            plastic_strain: Vec::with_capacity(capacity),
            mass: Vec::with_capacity(capacity),
            volume0: Vec::with_capacity(capacity),
        }
    }

    /// Append a stress-free particle.
    pub fn push(
        &mut self,
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        mass: f64,
        volume0: f64,
    ) {
        let id = u32::try_from(self.id.len()).unwrap_or(u32::MAX);
        self.id.push(id);
        self.position.push(position);
        self.velocity.push(velocity);
        self.affine.push(Matrix3::zeros());
        self.deformation.push(Matrix3::identity());
        self.volume_ratio.push(1.0);
        self.kirchhoff.push(Matrix3::zeros());
        self.plastic_strain.push(0.0);
        self.mass.push(mass);
        self.volume0.push(volume0);
    }

    /// Number of particles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.id.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// Total mass in kg (fixed-order sum).
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        self.mass.iter().sum()
    }

    /// Total linear momentum in kg m/s (fixed-order sum).
    #[must_use]
    pub fn linear_momentum(&self) -> Vector3<f64> {
        self.mass
            .iter()
            .zip(&self.velocity)
            .fold(Vector3::zeros(), |acc, (m, v)| acc + v * *m)
    }

    /// Point-mass angular momentum about the origin in kg m^2/s. The APIC affine
    /// contribution (`m_p * axial(C_p D_p)`) is intentionally omitted.
    #[must_use]
    pub fn angular_momentum(&self) -> Vector3<f64> {
        self.mass
            .iter()
            .zip(&self.velocity)
            .zip(&self.position)
            .fold(Vector3::zeros(), |acc, ((m, v), x)| acc + x.cross(v) * *m)
    }

    /// Kinetic energy in J (fixed-order sum).
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        self.mass
            .iter()
            .zip(&self.velocity)
            .map(|(m, v)| 0.5 * m * v.norm_squared())
            .sum()
    }

    /// Reorder all arrays by the given permutation (`order[k]` is the old index
    /// of the new `k`-th particle).
    pub fn permute(&mut self, order: &[usize]) {
        fn apply<T: Clone>(values: &mut Vec<T>, order: &[usize]) {
            let reordered: Vec<T> = order.iter().map(|&i| values[i].clone()).collect();
            *values = reordered;
        }
        apply(&mut self.id, order);
        apply(&mut self.position, order);
        apply(&mut self.velocity, order);
        apply(&mut self.affine, order);
        apply(&mut self.deformation, order);
        apply(&mut self.volume_ratio, order);
        apply(&mut self.kirchhoff, order);
        apply(&mut self.plastic_strain, order);
        apply(&mut self.mass, order);
        apply(&mut self.volume0, order);
    }

    /// Remove particles at the given sorted, unique indices, returning the
    /// removed mass in kg.
    pub fn remove_sorted(&mut self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }
        let removed_mass: f64 = indices.iter().map(|&i| self.mass[i]).sum();
        let keep: Vec<usize> = (0..self.len())
            .filter(|i| indices.binary_search(i).is_err())
            .collect();
        self.permute(&keep);
        removed_mass
    }

    /// Verify every stored value is finite and array lengths agree.
    ///
    /// # Errors
    ///
    /// `String` naming the first inconsistency.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.len();
        let lengths = [
            ("position", self.position.len()),
            ("velocity", self.velocity.len()),
            ("affine", self.affine.len()),
            ("deformation", self.deformation.len()),
            ("volume_ratio", self.volume_ratio.len()),
            ("kirchhoff", self.kirchhoff.len()),
            ("plastic_strain", self.plastic_strain.len()),
            ("mass", self.mass.len()),
            ("volume0", self.volume0.len()),
        ];
        for (name, len) in lengths {
            if len != n {
                return Err(format!(
                    "particle array `{name}` has length '{len}', expected '{n}'"
                ));
            }
        }
        let mut seen = self.id.clone();
        seen.sort_unstable();
        if seen.windows(2).any(|w| w[0] == w[1]) {
            return Err("particle ids are not unique".to_string());
        }
        for i in 0..n {
            let finite = self.position[i].iter().all(|v| v.is_finite())
                && self.velocity[i].iter().all(|v| v.is_finite())
                && self.affine[i].iter().all(|v| v.is_finite())
                && self.deformation[i].iter().all(|v| v.is_finite())
                && self.kirchhoff[i].iter().all(|v| v.is_finite())
                && self.volume_ratio[i].is_finite()
                && self.plastic_strain[i].is_finite()
                && self.mass[i].is_finite()
                && self.volume0[i].is_finite();
            if !finite {
                return Err(format!("particle id '{}' has non-finite state", self.id[i]));
            }
            if self.mass[i] <= 0.0 || self.volume0[i] <= 0.0 {
                return Err(format!(
                    "particle id '{}' has non-positive mass or volume",
                    self.id[i]
                ));
            }
            if self.volume_ratio[i] <= 0.0 || self.plastic_strain[i] < 0.0 {
                return Err(format!(
                    "particle id '{}' has invalid ratio/plastic state",
                    self.id[i]
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact values by construction.
mod tests {
    use super::ParticleSet;
    use nalgebra::Vector3;

    fn sample() -> ParticleSet {
        let mut set = ParticleSet::with_capacity(3);
        set.push(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            1.0,
            1.0,
        );
        set.push(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            2.0,
            1.0,
        );
        set.push(
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 3.0),
            3.0,
            1.0,
        );
        set
    }

    #[test]
    fn sums_are_exact_for_small_sets() {
        let set = sample();
        assert_eq!(set.total_mass(), 6.0);
        assert_eq!(set.linear_momentum(), Vector3::new(1.0, 4.0, 9.0));
        assert_eq!(set.kinetic_energy(), 0.5 + 4.0 + 13.5);
        assert_eq!(set.angular_momentum(), Vector3::new(9.0, 0.0, 4.0));
    }

    #[test]
    fn permute_and_remove_keep_identity() {
        let mut set = sample();
        set.permute(&[2, 0, 1]);
        assert_eq!(set.id, vec![2, 0, 1]);
        assert_eq!(set.mass, vec![3.0, 1.0, 2.0]);
        let removed = set.remove_sorted(&[1]);
        assert_eq!(removed, 1.0);
        assert_eq!(set.id, vec![2, 1]);
        assert!(set.validate().is_ok());
    }

    #[test]
    fn validate_rejects_non_finite() {
        let mut set = sample();
        set.velocity[1].x = f64::NAN;
        assert!(set.validate().is_err());
        let mut set = sample();
        set.id[1] = 0;
        assert!(set.validate().is_err());
    }
}
