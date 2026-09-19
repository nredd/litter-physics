//! Coarse finite-volume field for fines, deposits and free water in one bed.
//!
//! Cells hold wood fines, deposit solids and mobile water in kilograms. Water
//! drains downward at a fixed rate limited by the void capacity of the cell
//! below, pools in the bottom layer, spreads laterally when a cell overflows,
//! and leaves through slot cells into the drawer. Fines settle downward and sift
//! through open slots, and spread laterally so piles resting on plate bars reach
//! neighbouring openings. Deposit solids do not move (no paste rheology in the
//! household baseline). All transfers are explicit mass moves, so the field is
//! conservative by construction; sinks return the mass they removed.
//!
//! References: `docs/household.md`.

use serde::{Deserialize, Serialize};

use crate::dem::{self, BoxGeometry, Vec3};

/// Water density used to convert void volume into water capacity, kg/m^3.
pub const WATER_DENSITY_KG_M3: f64 = 1000.0;
/// First-order downward drainage rate for mobile water, 1/s.
pub const WATER_DRAIN_RATE_S: f64 = 5.0;
/// First-order settling rate for fines through the bed, 1/s.
pub const FINES_SETTLE_RATE_S: f64 = 0.5;
/// First-order rate at which bottom-layer fines pass open slots, 1/s.
pub const FINES_SIFT_RATE_S: f64 = 0.5;
/// First-order rate at which fines spread into lateral neighbours, 1/s.
pub const FINES_SPREAD_RATE_S: f64 = 0.5;
/// Upper bound on pellet volume fraction per cell (random close packing).
pub const MAX_OCCUPANCY: f64 = 0.64;
/// Fraction of void volume the bed retains as water before draining.
pub const WATER_RETENTION_FRACTION: f64 = 0.05;

/// Mass moved out of the field into the drawer during one update.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Outflow {
    /// Wood fines passed through slots, kg.
    pub wood_kg: f64,
    /// Water passed through slots, kg.
    pub water_kg: f64,
}

/// One box's coarse field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    /// Cell counts along `x`, `y`, `z`.
    pub dims: [usize; 3],
    /// Cell spacing in metres.
    pub spacing_m: f64,
    /// Wood fines per cell, kg.
    pub fines_kg: Vec<f64>,
    /// Deposit solids per cell, kg.
    pub waste_kg: Vec<f64>,
    /// Mobile water per cell, kg.
    pub water_kg: Vec<f64>,
    /// Open slot fraction under each bottom-layer column, `dims[0] * dims[1]`.
    pub slot_open: Vec<f64>,
    /// Pellet volume fraction per cell, refreshed each step (not persisted).
    #[serde(skip)]
    pub occupancy: Vec<f64>,
}

impl Field {
    /// Create an empty field for a box.
    #[must_use]
    pub fn new(dims: [usize; 3], geometry: &BoxGeometry) -> Self {
        let cells = dims[0] * dims[1] * dims[2];
        let spacing = super::request::FIELD_SPACING_M;
        let mut slot_open = Vec::with_capacity(dims[0] * dims[1]);
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                slot_open.push(column_open_fraction(geometry, ix, iy, spacing));
            }
        }
        Self {
            dims,
            spacing_m: spacing,
            fines_kg: vec![0.0; cells],
            waste_kg: vec![0.0; cells],
            water_kg: vec![0.0; cells],
            slot_open,
            occupancy: vec![0.0; cells],
        }
    }

    /// Total cell count.
    #[must_use]
    pub fn cells(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Verify array lengths and finiteness after deserialisation.
    ///
    /// # Errors
    ///
    /// Returns a description of the first inconsistency.
    pub fn check(&mut self, expected_dims: [usize; 3]) -> Result<(), String> {
        if self.dims != expected_dims {
            return Err(format!(
                "field dims {:?} do not match the request geometry {expected_dims:?}",
                self.dims
            ));
        }
        let cells = self.cells();
        for (name, values, len) in [
            ("fines_kg", &self.fines_kg, cells),
            ("waste_kg", &self.waste_kg, cells),
            ("water_kg", &self.water_kg, cells),
            ("slot_open", &self.slot_open, self.dims[0] * self.dims[1]),
        ] {
            if values.len() != len {
                return Err(format!(
                    "field `{name}` has {} entries, expected {len}",
                    values.len()
                ));
            }
            if let Some(bad) = values.iter().find(|v| !v.is_finite() || **v < 0.0) {
                return Err(format!(
                    "field `{name}` holds a non-finite or negative value '{bad}'"
                ));
            }
        }
        if !(self.spacing_m.is_finite() && self.spacing_m > 0.0) {
            return Err(format!("field spacing '{}' is invalid", self.spacing_m));
        }
        self.occupancy = vec![0.0; cells];
        Ok(())
    }

    /// Flat index of a cell.
    #[must_use]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (iz * self.dims[1] + iy) * self.dims[0] + ix
    }

    /// Cell containing a local position, clamped into the grid.
    #[must_use]
    pub fn cell_of(&self, position: Vec3) -> [usize; 3] {
        let clamp = |v: f64, n: usize| {
            let i = dem::floor_to_i64(v / self.spacing_m).max(0);
            usize::try_from(i).unwrap_or(0).min(n.saturating_sub(1))
        };
        [
            clamp(position.x, self.dims[0]),
            clamp(position.y, self.dims[1]),
            clamp(position.z, self.dims[2]),
        ]
    }

    /// Local coordinates of a cell centre.
    #[must_use]
    pub fn centre(&self, ix: usize, iy: usize, iz: usize) -> Vec3 {
        let h = self.spacing_m;
        Vec3::new(
            (dem::count_to_f64(ix) + 0.5) * h,
            (dem::count_to_f64(iy) + 0.5) * h,
            (dem::count_to_f64(iz) + 0.5) * h,
        )
    }

    /// Highest occupied cell in a column, or the bottom cell when empty.
    #[must_use]
    pub fn surface_layer(&self, ix: usize, iy: usize) -> usize {
        (0..self.dims[2])
            .rev()
            .find(|&iz| self.occupancy[self.index(ix, iy, iz)] > 0.0)
            .unwrap_or(0)
    }

    /// Columns whose centre lies within `radius` of a local `xy` position.
    #[must_use]
    pub fn columns_within(&self, position: Vec3, radius: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for iy in 0..self.dims[1] {
            for ix in 0..self.dims[0] {
                let c = self.centre(ix, iy, 0);
                if (c.x - position.x).hypot(c.y - position.y) <= radius {
                    out.push((ix, iy));
                }
            }
        }
        if out.is_empty() {
            let [ix, iy, _] = self.cell_of(position);
            out.push((ix, iy));
        }
        out
    }

    /// Deposit solids and water onto the surface cells around a position.
    pub fn deposit(&mut self, position: Vec3, radius: f64, waste_kg: f64, water_kg: f64) {
        let columns = self.columns_within(position, radius);
        let share = 1.0 / dem::count_to_f64(columns.len());
        for (ix, iy) in columns {
            let iz = self.surface_layer(ix, iy);
            let i = self.index(ix, iy, iz);
            self.waste_kg[i] += waste_kg * share;
            self.water_kg[i] += water_kg * share;
        }
    }

    /// Add fines to the surface cell of the column under a position.
    pub fn add_fines(&mut self, position: Vec3, fines_kg: f64) {
        let [ix, iy, _] = self.cell_of(position);
        let iz = self.surface_layer(ix, iy);
        let i = self.index(ix, iy, iz);
        self.fines_kg[i] += fines_kg;
    }

    /// Add fines and water to a specific cell (used for pellet breakup).
    pub fn add_to_cell(&mut self, cell: [usize; 3], fines_kg: f64, water_kg: f64) {
        let i = self.index(cell[0], cell[1], cell[2]);
        self.fines_kg[i] += fines_kg;
        self.water_kg[i] += water_kg;
    }

    /// Take up to `amount` of water from a cell, returning what was taken.
    pub fn take_water(&mut self, cell: [usize; 3], amount: f64) -> f64 {
        let i = self.index(cell[0], cell[1], cell[2]);
        let taken = amount.min(self.water_kg[i]).max(0.0);
        self.water_kg[i] -= taken;
        taken
    }

    /// Remove all material from cells whose centre lies within `radius` of a
    /// position. Returns (fines, waste, water) removed.
    pub fn remove_within(&mut self, position: Vec3, radius: f64) -> (f64, f64, f64) {
        let mut out = (0.0, 0.0, 0.0);
        for iz in 0..self.dims[2] {
            for iy in 0..self.dims[1] {
                for ix in 0..self.dims[0] {
                    if (self.centre(ix, iy, iz) - position).norm() <= radius {
                        let i = self.index(ix, iy, iz);
                        out.0 += std::mem::take(&mut self.fines_kg[i]);
                        out.1 += std::mem::take(&mut self.waste_kg[i]);
                        out.2 += std::mem::take(&mut self.water_kg[i]);
                    }
                }
            }
        }
        out
    }

    /// Remove everything. Returns (fines, waste, water) removed.
    pub fn clear(&mut self) -> (f64, f64, f64) {
        let sum = |v: &mut Vec<f64>| {
            let s: f64 = v.iter().sum();
            for x in v.iter_mut() {
                *x = 0.0;
            }
            s
        };
        (
            sum(&mut self.fines_kg),
            sum(&mut self.waste_kg),
            sum(&mut self.water_kg),
        )
    }

    /// Species totals (fines, waste, water).
    #[must_use]
    pub fn totals(&self) -> (f64, f64, f64) {
        (
            self.fines_kg.iter().sum(),
            self.waste_kg.iter().sum(),
            self.water_kg.iter().sum(),
        )
    }

    /// Water capacity of a cell given current occupancy, kg.
    fn capacity(&self, i: usize) -> f64 {
        let void = (1.0 - self.occupancy[i]).max(0.0) * self.spacing_m.powi(3);
        void * WATER_DENSITY_KG_M3
    }

    /// Evaporate first-order from every cell; returns the water removed.
    pub fn evaporate(&mut self, rate_s: f64, dt: f64) -> f64 {
        let fraction = 1.0 - (-rate_s * dt).exp();
        let mut removed = 0.0;
        for w in &mut self.water_kg {
            let e = *w * fraction;
            *w -= e;
            removed += e;
        }
        removed
    }

    /// Advance drainage, pooling, lateral overflow and slot outflow by `dt`.
    pub fn transport(&mut self, dt: f64) -> Outflow {
        let mut out = Outflow::default();
        let drain = (1.0 - (-WATER_DRAIN_RATE_S * dt).exp()).min(1.0);
        let settle = (1.0 - (-FINES_SETTLE_RATE_S * dt).exp()).min(1.0);
        let sift = (1.0 - (-FINES_SIFT_RATE_S * dt).exp()).min(1.0);
        for iz in 0..self.dims[2] {
            for iy in 0..self.dims[1] {
                for ix in 0..self.dims[0] {
                    let i = self.index(ix, iy, iz);
                    let column_open = self.slot_open[iy * self.dims[0] + ix];
                    if iz == 0 {
                        let open = column_open * (1.0 - self.occupancy[i]).max(0.0);
                        let water = self.water_kg[i] * drain * open;
                        self.water_kg[i] -= water;
                        out.water_kg += water;
                        let wood = self.fines_kg[i] * sift * open;
                        self.fines_kg[i] -= wood;
                        out.wood_kg += wood;
                    } else {
                        let below = self.index(ix, iy, iz - 1);
                        let retained = WATER_RETENTION_FRACTION * self.capacity(i);
                        let mobile = (self.water_kg[i] - retained).max(0.0);
                        let room = (self.capacity(below) - self.water_kg[below]).max(0.0);
                        let water = (mobile * drain).min(room);
                        self.water_kg[i] -= water;
                        self.water_kg[below] += water;
                        let space = (1.0 - self.occupancy[below]).max(0.0);
                        let wood = self.fines_kg[i] * settle * space;
                        self.fines_kg[i] -= wood;
                        self.fines_kg[below] += wood;
                    }
                }
            }
        }
        self.spread_overflow();
        self.spread_fines(1.0 - (-FINES_SPREAD_RATE_S * dt).exp());
        out
    }

    /// Indices of in-plane neighbours of a cell.
    fn lateral_neighbours(&self, ix: usize, iy: usize, iz: usize) -> Vec<usize> {
        let candidates = [
            (ix.checked_sub(1), Some(iy)),
            (Some(ix + 1).filter(|x| *x < self.dims[0]), Some(iy)),
            (Some(ix), iy.checked_sub(1)),
            (Some(ix), Some(iy + 1).filter(|y| *y < self.dims[1])),
        ];
        candidates
            .iter()
            .filter_map(|(x, y)| Some(self.index((*x)?, (*y)?, iz)))
            .collect()
    }

    /// Move a fraction of each cell's fines into lateral neighbours with void
    /// space, using the pre-update distribution so the move is symmetric.
    fn spread_fines(&mut self, fraction: f64) {
        let before = self.fines_kg.clone();
        for iz in 0..self.dims[2] {
            for iy in 0..self.dims[1] {
                for ix in 0..self.dims[0] {
                    let i = self.index(ix, iy, iz);
                    if before[i] <= 0.0 {
                        continue;
                    }
                    let targets = self.lateral_neighbours(ix, iy, iz);
                    if targets.is_empty() {
                        continue;
                    }
                    let share = before[i] * fraction / dem::count_to_f64(targets.len());
                    for j in targets {
                        let moved = share * (1.0 - self.occupancy[j]).max(0.0);
                        self.fines_kg[j] += moved;
                        self.fines_kg[i] -= moved;
                    }
                }
            }
        }
    }

    /// Move water above capacity into lateral neighbours with room.
    fn spread_overflow(&mut self) {
        for iz in 0..self.dims[2] {
            for iy in 0..self.dims[1] {
                for ix in 0..self.dims[0] {
                    let i = self.index(ix, iy, iz);
                    let excess = self.water_kg[i] - self.capacity(i);
                    if excess <= 0.0 {
                        continue;
                    }
                    let targets = self.lateral_neighbours(ix, iy, iz);
                    if targets.is_empty() {
                        continue;
                    }
                    let share = excess / dem::count_to_f64(targets.len());
                    for j in targets {
                        let room = (self.capacity(j) - self.water_kg[j]).max(0.0);
                        let moved = share.min(room);
                        self.water_kg[j] += moved;
                        self.water_kg[i] -= moved;
                    }
                }
            }
        }
    }
}

/// Open slot fraction under a column footprint by regular sampling.
fn column_open_fraction(geometry: &BoxGeometry, ix: usize, iy: usize, spacing: f64) -> f64 {
    const SAMPLES: u32 = 8;
    let mut open = 0u32;
    for i in 0..SAMPLES {
        for j in 0..SAMPLES {
            let x = (dem::count_to_f64(ix) + (f64::from(i) + 0.5) / f64::from(SAMPLES)) * spacing;
            let y = (dem::count_to_f64(iy) + (f64::from(j) + 0.5) / f64::from(SAMPLES)) * spacing;
            if geometry.slot_under(x, y).is_some() {
                open += 1;
            }
        }
    }
    f64::from(open) / f64::from(SAMPLES * SAMPLES)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> BoxGeometry {
        BoxGeometry {
            size_m: Vec3::new(0.04, 0.04, 0.03),
            slot_width_m: 0.005,
            slot_length_m: 0.015,
            slot_pitch_m: 0.02,
            drawer_depth_m: 0.03,
        }
    }

    #[test]
    fn deposit_drains_conservatively_to_drawer() {
        let geometry = geometry();
        let mut field = Field::new([8, 8, 6], &geometry);
        field.deposit(Vec3::new(0.02, 0.02, 0.03), 0.01, 0.001, 0.01);
        let (_, waste, water) = field.totals();
        assert!((waste - 0.001).abs() < 1e-15);
        assert!((water - 0.01).abs() < 1e-15);
        let mut drained = 0.0;
        for _ in 0..2000 {
            drained += field.transport(0.01).water_kg;
        }
        let (_, _, remaining) = field.totals();
        assert!((remaining + drained - 0.01).abs() < 1e-12, "water lost");
        assert!(drained > 0.5 * 0.01, "drained only {drained}");
    }

    #[test]
    fn fines_sift_through_open_columns_only() {
        let geometry = geometry();
        let mut field = Field::new([8, 8, 6], &geometry);
        field.add_fines(Vec3::new(0.0125, 0.01, 0.02), 0.001);
        let mut sifted = 0.0;
        for _ in 0..1000 {
            sifted += field.transport(0.01).wood_kg;
        }
        let (fines, _, _) = field.totals();
        assert!((fines + sifted - 0.001).abs() < 1e-12);
        assert!(sifted > 0.0);
        let solid = field.slot_open.iter().filter(|o| **o == 0.0).count();
        assert!(solid > 0, "expected solid columns");
    }

    #[test]
    fn overflow_spreads_laterally_without_loss() {
        let mut field = Field::new([3, 3, 1], &geometry());
        field.occupancy = vec![0.0; 9];
        field.slot_open = vec![0.0; 9];
        let i = field.index(1, 1, 0);
        field.water_kg[i] = 1.0;
        field.transport(0.1);
        let (_, _, total) = field.totals();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(field.water_kg[i] < 1.0);
    }

    #[test]
    fn check_rejects_inconsistent_state() {
        let mut field = Field::new([2, 2, 2], &geometry());
        assert!(field.check([2, 2, 3]).is_err());
        field.water_kg[0] = -1.0;
        assert!(field.check([2, 2, 2]).is_err());
        field.water_kg[0] = 0.0;
        field.fines_kg.pop();
        assert!(field.check([2, 2, 2]).is_err());
    }
}
