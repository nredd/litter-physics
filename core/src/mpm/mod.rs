//! CPU MLS-MPM research kernels.
//!
//! The module implements a deterministic moving-least-squares material point
//! method (APIC transfers, quadratic B-spline kernels) with:
//!
//! - an elastoviscoplastic Herschel-Bulkley paste using Hencky elasticity and a
//!   local implicit radial-return update ([`constitutive`]),
//! - a weakly compressible Newtonian liquid tracked through its volume ratio,
//! - Coulomb-friction domain walls with logged impulses ([`solver`]),
//! - impulse-conserving two-way coupling to a rigid oriented multisphere pellet
//!   ([`rigid`]); the current split can create kinetic energy for light bodies,
//! - continuum CFL / viscous / contact-stability timestep control with step
//!   rejection on failed constitutive solves ([`solver`]).
//!
//! Everything here operates on plain SI values. Parsing, validation of the wire
//! request, checkpointing and output construction live in `crate::research`.
//!
//! References:
//! - Hu et al. 2018, MLS-MPM, <https://doi.org/10.1145/3197517.3201293>
//! - Yue et al. 2015, Herschel-Bulkley MPM, <https://doi.org/10.1145/2751541>
//! - Jiang et al. 2015, APIC, <https://doi.org/10.1145/2766996>
//! - Klar et al. 2016, principal-frame plastic return, <https://doi.org/10.1145/2897824.2925906>

pub mod audit;
#[cfg(test)]
mod audit_tests;
pub mod constitutive;
pub mod fixtures;
pub mod grid;
pub mod particles;
pub mod rigid;
pub mod rng;
pub mod solver;
