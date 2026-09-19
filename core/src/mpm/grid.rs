//! Background grid with quadratic B-spline kernels.
//!
//! The grid is a dense node array padded by one node on each side of the
//! domain `[0, L]^3` so that every particle inside the domain has its full
//! 3x3x3 quadratic stencil available. Nodes at or beyond the domain faces are
//! wall nodes. The list of nodes touched during a transfer is recorded so grid
//! updates only visit active nodes; storage is dense (an explicit gap relative
//! to sparse active-block storage, see `docs/research.md`).

use nalgebra::Vector3;

/// Grid layout: `n` cells per axis, `n + 3` nodes per axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridLayout {
    /// Cell size in m.
    pub spacing: f64,
    /// Cells per axis.
    pub cells: [usize; 3],
    /// Nodes per axis (`cells + 3`).
    pub nodes: [usize; 3],
}

impl GridLayout {
    /// Build a layout for a domain that is an integer multiple of `spacing`.
    ///
    /// Parameters:
    /// - `domain` (`[f64; 3]`): domain extents in m, positive.
    /// - `spacing` (`f64`): cell size in m, positive.
    /// - `max_nodes` (`usize`): allocation cap on total node count.
    ///
    /// Returns: `Result<GridLayout, String>`.
    ///
    /// # Errors
    ///
    /// if extents are not multiples of the spacing (relative tolerance
    /// 1e-6), if any axis has fewer than two cells, or the node cap is exceeded.
    pub fn new(domain: [f64; 3], spacing: f64, max_nodes: usize) -> Result<Self, String> {
        if !(spacing.is_finite() && spacing > 0.0) {
            return Err(format!(
                "`grid_spacing_m` must be positive and finite, got '{spacing}'"
            ));
        }
        let mut cells = [0usize; 3];
        for axis in 0..3 {
            let extent = domain[axis];
            if !(extent.is_finite() && extent > 0.0) {
                return Err(format!(
                    "`domain_m[{axis}]` must be positive and finite, got '{extent}'"
                ));
            }
            let ratio = extent / spacing;
            let rounded = ratio.round();
            let cap =
                u32::try_from(max_nodes.saturating_sub(3)).map_or(f64::from(u32::MAX), f64::from);
            if !ratio.is_finite() || rounded > cap {
                return Err("grid axis exceeds the node allocation cap".into());
            }
            if (ratio - rounded).abs() > 1e-6 * rounded.max(1.0) {
                return Err(format!(
                    "`domain_m[{axis}]`='{extent}' is not an integer multiple of `grid_spacing_m`='{spacing}'"
                ));
            }
            if rounded < 2.0 {
                return Err(format!("`domain_m[{axis}]` must span at least two cells"));
            }
            // Bounded above by the node cap check below; `rounded` is a small positive integer.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let count = rounded as usize;
            cells[axis] = count;
        }
        let nodes = [cells[0] + 3, cells[1] + 3, cells[2] + 3];
        let total = nodes[0]
            .checked_mul(nodes[1])
            .and_then(|v| v.checked_mul(nodes[2]))
            .ok_or_else(|| "grid node count overflows".to_string())?;
        if total > max_nodes {
            return Err(format!(
                "grid would allocate '{total}' nodes, exceeding the cap of '{max_nodes}'; coarsen `grid_spacing_m` or shrink `domain_m`"
            ));
        }
        Ok(Self {
            spacing,
            cells,
            nodes,
        })
    }

    /// Total node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes[0] * self.nodes[1] * self.nodes[2]
    }

    /// Domain extents in m.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // cell counts are small integers.
    pub fn extents(&self) -> Vector3<f64> {
        Vector3::new(
            self.cells[0] as f64 * self.spacing,
            self.cells[1] as f64 * self.spacing,
            self.cells[2] as f64 * self.spacing,
        )
    }

    /// Flat index of a node from padded integer coordinates.
    #[must_use]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.nodes[1] + j) * self.nodes[2] + k
    }

    /// Physical position of a padded node.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // node indices are small integers.
    pub fn node_position(&self, i: usize, j: usize, k: usize) -> Vector3<f64> {
        Vector3::new(
            (i as f64 - 1.0) * self.spacing,
            (j as f64 - 1.0) * self.spacing,
            (k as f64 - 1.0) * self.spacing,
        )
    }

    /// Padded integer coordinates of a flat node index.
    #[must_use]
    pub fn coords(&self, index: usize) -> (usize, usize, usize) {
        let k = index % self.nodes[2];
        let rest = index / self.nodes[2];
        let j = rest % self.nodes[1];
        let i = rest / self.nodes[1];
        (i, j, k)
    }

    /// Whether a padded node coordinate along an axis is a wall node, and the
    /// inward normal sign (+1 at the low face, -1 at the high face).
    #[must_use]
    pub fn wall_side(&self, axis: usize, coord: usize) -> Option<f64> {
        if coord <= 1 {
            Some(1.0)
        } else if coord > self.cells[axis] {
            Some(-1.0)
        } else {
            None
        }
    }

    /// Quadratic B-spline stencil for a particle position.
    ///
    /// Returns: `None` if the particle lies outside the domain padding, else the
    /// base padded node coordinates and per-axis weights `w[axis][offset]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // node counts and offsets are small integers.
    pub fn stencil(&self, position: &Vector3<f64>) -> Option<Stencil> {
        let mut base = [0usize; 3];
        let mut weights = [[0.0f64; 3]; 3];
        let mut offsets = [Vector3::zeros(); 3];
        for axis in 0..3 {
            let scaled = position[axis] / self.spacing;
            let base_f = (scaled - 0.5).floor();
            // Padded index is base + 1; must keep the full stencil inside the array.
            let padded = base_f + 1.0;
            if padded.is_nan() || padded < 0.0 || padded + 2.0 >= self.nodes[axis] as f64 {
                return None;
            }
            let fx = scaled - base_f;
            weights[axis] = [
                0.5 * (1.5 - fx).powi(2),
                0.75 - (fx - 1.0).powi(2),
                0.5 * (fx - 0.5).powi(2),
            ];
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            // checked >= 0 above.
            let padded_index = padded as usize;
            base[axis] = padded_index;
            for (offset, slot) in offsets[axis].iter_mut().enumerate() {
                *slot = (fx - offset as f64) * -self.spacing;
            }
        }
        Some(Stencil {
            base,
            weights,
            offsets,
        })
    }
}

/// Quadratic B-spline stencil of one particle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stencil {
    /// Padded base node coordinates.
    pub base: [usize; 3],
    /// Weights per axis and offset.
    pub weights: [[f64; 3]; 3],
    /// Node-minus-particle distance per axis and offset in m
    /// (`offsets[axis][o] = x_node - x_particle` along `axis`).
    pub offsets: [Vector3<f64>; 3],
}

impl Stencil {
    /// Scalar weight of the stencil entry.
    #[must_use]
    pub fn weight(&self, a: usize, b: usize, c: usize) -> f64 {
        self.weights[0][a] * self.weights[1][b] * self.weights[2][c]
    }

    /// Node-minus-particle vector of the stencil entry.
    #[must_use]
    pub fn distance(&self, a: usize, b: usize, c: usize) -> Vector3<f64> {
        Vector3::new(self.offsets[0][a], self.offsets[1][b], self.offsets[2][c])
    }
}

/// Node storage for one transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// Layout.
    pub layout: GridLayout,
    /// Node mass in kg.
    pub mass: Vec<f64>,
    /// Node momentum in kg m/s (becomes velocity after the grid update).
    pub momentum: Vec<Vector3<f64>>,
    /// Flat indices of nodes touched in the last transfer, in first-touch order.
    pub active: Vec<usize>,
}

impl Grid {
    /// Allocate zeroed storage for a layout.
    #[must_use]
    pub fn new(layout: GridLayout) -> Self {
        let n = layout.node_count();
        Self {
            layout,
            mass: vec![0.0; n],
            momentum: vec![Vector3::zeros(); n],
            active: Vec::new(),
        }
    }

    /// Zero the active nodes and clear the active list.
    pub fn clear(&mut self) {
        for &index in &self.active {
            self.mass[index] = 0.0;
            self.momentum[index] = Vector3::zeros();
        }
        self.active.clear();
    }

    /// Accumulate mass and momentum at a node, registering it as active.
    pub fn deposit(&mut self, index: usize, mass: f64, momentum: Vector3<f64>) {
        if self.mass[index] == 0.0 {
            self.active.push(index);
        }
        self.mass[index] += mass;
        self.momentum[index] += momentum;
    }

    /// Total momentum over active nodes (fixed order).
    #[must_use]
    pub fn total_momentum(&self) -> Vector3<f64> {
        self.active
            .iter()
            .fold(Vector3::zeros(), |acc, &i| acc + self.momentum[i])
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact values by construction.
mod tests {
    use super::GridLayout;
    use nalgebra::Vector3;

    #[test]
    fn layout_rejects_non_multiple_domain() {
        assert!(GridLayout::new([0.011, 0.02, 0.02], 0.002, 1_000_000).is_err());
        assert!(GridLayout::new([0.02, 0.02, 0.02], 0.002, 10).is_err());
        assert!(GridLayout::new([0.002, 0.02, 0.02], 0.002, 1_000_000).is_err());
        let layout =
            GridLayout::new([0.02, 0.04, 0.06], 0.002, 1_000_000).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(layout.cells, [10, 20, 30]);
        assert_eq!(layout.nodes, [13, 23, 33]);
    }

    #[test]
    fn stencil_partitions_unity_and_reproduces_position() {
        let layout =
            GridLayout::new([0.02, 0.02, 0.02], 0.002, 1_000_000).unwrap_or_else(|e| panic!("{e}"));
        for position in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0031, 0.0177, 0.02),
            Vector3::new(0.0195, 0.0004, 0.011),
        ] {
            let stencil = layout
                .stencil(&position)
                .unwrap_or_else(|| panic!("stencil missing for {position:?}"));
            let mut total = 0.0;
            let mut centroid = Vector3::zeros();
            for a in 0..3 {
                for b in 0..3 {
                    for c in 0..3 {
                        let w = stencil.weight(a, b, c);
                        total += w;
                        let node = layout.node_position(
                            stencil.base[0] + a,
                            stencil.base[1] + b,
                            stencil.base[2] + c,
                        );
                        centroid += node * w;
                        let d = stencil.distance(a, b, c);
                        assert!((node - position - d).norm() < 1e-12);
                    }
                }
            }
            assert!((total - 1.0).abs() < 1e-12);
            assert!((centroid - position).norm() < 1e-12);
        }
        assert!(layout.stencil(&Vector3::new(-0.003, 0.0, 0.0)).is_none());
        assert!(layout.stencil(&Vector3::new(0.0, 0.0, 0.0231)).is_none());
    }

    #[test]
    fn wall_sides_match_faces() {
        let layout =
            GridLayout::new([0.02, 0.02, 0.02], 0.002, 1_000_000).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(layout.wall_side(0, 0), Some(1.0));
        assert_eq!(layout.wall_side(0, 1), Some(1.0));
        assert_eq!(layout.wall_side(0, 2), None);
        assert_eq!(layout.wall_side(0, 11), Some(-1.0));
        assert_eq!(layout.wall_side(0, 12), Some(-1.0));
    }
}
