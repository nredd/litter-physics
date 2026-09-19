//! Constitutive models for the research MPM kernels.
//!
//! Two materials are implemented:
//!
//! - `Paste`: elastoviscoplastic Herschel-Bulkley. Elasticity is Hencky
//!   (logarithmic strain, `tau = 2 mu eps + lambda tr(eps) I` in Kirchhoff
//!   stress), plasticity is J2 with a Herschel-Bulkley overstress law
//!   `sigma_eq = sigma_y + K * eps_dot_p^n`. The plastic update is the local
//!   implicit radial return of Yue et al. 2015 written in the principal frame of
//!   the trial elastic deformation gradient: the scalar equation
//!   `sigma - sigma_tr + 3 mu dt ((sigma - sigma_y) / K)^(1/n) = 0` is solved
//!   with a bracketed Newton iteration, which is monotone on
//!   `[sigma_y, sigma_tr]` and therefore always converges.
//! - `Liquid`: weakly compressible Newtonian liquid tracked by its volume ratio
//!   `J` with the linear equation of state `p = K_bulk (1/J - 1)` (artificial
//!   bulk modulus, so the artificial wave speed is `sqrt(K_bulk / rho)`) and a
//!   deviatoric viscous Kirchhoff stress `2 mu_visc J dev(D)`.
//!
//! References:
//! - Yue et al. 2015, <https://doi.org/10.1145/2751541> (Herschel-Bulkley MPM)
//! - Klar et al. 2016, <https://doi.org/10.1145/2897824.2925906> (principal frame return)
//! - Tampubolon et al. 2017, <https://doi.org/10.1145/3072959.3073651> (weakly compressible liquid)

use std::fmt;

use nalgebra::{Matrix3, Vector3};

/// Relative tolerance of the scalar Herschel-Bulkley return solve.
const RETURN_RELATIVE_TOLERANCE: f64 = 1e-12;
/// Maximum iterations of the scalar Herschel-Bulkley return solve.
const RETURN_MAX_ITERATIONS: usize = 200;
/// Jacobi sweep cap for the symmetric 3x3 eigensolver.
const JACOBI_MAX_SWEEPS: usize = 64;
/// Relative off-diagonal norm at which the Jacobi iteration stops.
const JACOBI_TOLERANCE: f64 = 1e-30;
/// Smallest admissible singular value of an elastic deformation gradient.
const MIN_STRETCH: f64 = 1e-6;

/// Constitutive failure. Every variant maps to a rejected step in the solver.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstitutiveError {
    /// The trial deformation gradient is inverted or degenerate.
    InvertedDeformation {
        /// Determinant of the trial deformation gradient.
        determinant: f64,
    },
    /// The eigen-decomposition of the trial left Cauchy-Green tensor did not converge.
    EigenFailed,
    /// A stretch is below the internal `MIN_STRETCH` threshold or non-finite.
    DegenerateStretch {
        /// Offending singular value.
        stretch: f64,
    },
    /// The scalar radial-return solve did not converge.
    ReturnNotConverged {
        /// Trial equivalent stress in Pa.
        trial_stress: f64,
        /// Residual at the last iterate in Pa.
        residual: f64,
    },
    /// A non-finite quantity was produced.
    NonFinite {
        /// Name of the offending quantity.
        what: &'static str,
    },
}

impl fmt::Display for ConstitutiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvertedDeformation { determinant } => {
                write!(
                    f,
                    "trial deformation gradient inverted: det='{determinant}'"
                )
            }
            Self::EigenFailed => {
                write!(
                    f,
                    "eigen-decomposition of trial Cauchy-Green tensor did not converge"
                )
            }
            Self::DegenerateStretch { stretch } => {
                write!(f, "degenerate elastic stretch: '{stretch}'")
            }
            Self::ReturnNotConverged {
                trial_stress,
                residual,
            } => write!(
                f,
                "Herschel-Bulkley return did not converge: trial='{trial_stress}' residual='{residual}'"
            ),
            Self::NonFinite { what } => write!(f, "non-finite constitutive quantity: `{what}`"),
        }
    }
}

impl std::error::Error for ConstitutiveError {}

/// Material kind resolved from the wire request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialKind {
    /// Herschel-Bulkley elastoviscoplastic paste.
    Paste,
    /// Weakly compressible Newtonian liquid.
    Liquid,
}

/// Fully resolved material parameters in SI units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Material model.
    pub kind: MaterialKind,
    /// Reference density in kg/m^3.
    pub density: f64,
    /// Shear modulus `mu` in Pa (paste only, zero for liquid).
    pub shear_modulus: f64,
    /// First Lame parameter `lambda` in Pa (paste only, zero for liquid).
    pub lame_lambda: f64,
    /// Bulk modulus in Pa. For paste this is `lambda + 2 mu / 3`; for liquid it
    /// is the artificial bulk modulus of the equation of state.
    pub bulk_modulus: f64,
    /// Von Mises equivalent yield stress `sigma_y` in Pa (paste only).
    pub yield_stress: f64,
    /// Herschel-Bulkley consistency `K` in Pa s^n in the von Mises form
    /// `sigma_eq = sigma_y + K eps_dot_p^n` (paste), or dynamic viscosity in
    /// Pa s (liquid, where `flow_index == 1`).
    pub consistency: f64,
    /// Herschel-Bulkley flow index `n` (dimensionless).
    pub flow_index: f64,
}

impl Material {
    /// Build a paste from Young's modulus and Poisson's ratio.
    ///
    /// Parameters:
    /// - `density` (`f64`): reference density in kg/m^3, positive.
    /// - `young_modulus` (`f64`): Young's modulus in Pa, positive.
    /// - `poisson_ratio` (`f64`): Poisson's ratio in `[0, 0.5)`.
    /// - `yield_stress` (`f64`): yield stress in Pa, non-negative.
    /// - `consistency` (`f64`): Herschel-Bulkley consistency in Pa s^n, non-negative.
    /// - `flow_index` (`f64`): flow index, positive.
    ///
    /// Returns: `Material` with derived Lame parameters. Callers validate ranges.
    #[must_use]
    pub fn paste(
        density: f64,
        young_modulus: f64,
        poisson_ratio: f64,
        yield_stress: f64,
        consistency: f64,
        flow_index: f64,
    ) -> Self {
        let shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio));
        let lame_lambda =
            young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        Self {
            kind: MaterialKind::Paste,
            density,
            shear_modulus,
            lame_lambda,
            bulk_modulus: lame_lambda + 2.0 * shear_modulus / 3.0,
            yield_stress,
            consistency,
            flow_index,
        }
    }

    /// Build a paste from shear-rheometer Herschel-Bulkley parameters
    /// (`tau = tau_y + K_s gamma_dot^n` in simple shear), converting to the
    /// von Mises form used internally: `sigma_y = sqrt(3) tau_y` and
    /// `K = 3^((1 + n) / 2) K_s`, from `sigma_eq = sqrt(3) tau` and
    /// `eps_dot_p = gamma_dot / sqrt(3)`.
    ///
    /// Parameters: as [`Material::paste`], with `shear_yield_stress` in Pa and
    /// `shear_consistency` in Pa s^n measured in shear.
    ///
    /// Returns: `Material` in paste mode.
    #[must_use]
    pub fn paste_from_shear_rheology(
        density: f64,
        young_modulus: f64,
        poisson_ratio: f64,
        shear_yield_stress: f64,
        shear_consistency: f64,
        flow_index: f64,
    ) -> Self {
        let root3 = 3f64.sqrt();
        Self::paste(
            density,
            young_modulus,
            poisson_ratio,
            root3 * shear_yield_stress,
            3f64.powf(f64::midpoint(1.0, flow_index)) * shear_consistency,
            flow_index,
        )
    }

    /// Build a weakly compressible liquid whose artificial bulk modulus is
    /// derived from the wire `young_modulus_pa` / `poisson_ratio` pair as
    /// `E / (3 (1 - 2 nu))`, with `consistency` acting as dynamic viscosity.
    ///
    /// Parameters:
    /// - `density` (`f64`): reference density in kg/m^3.
    /// - `young_modulus` (`f64`): modulus used to derive the artificial bulk modulus.
    /// - `poisson_ratio` (`f64`): Poisson's ratio in `[0, 0.5)`.
    /// - `viscosity` (`f64`): dynamic viscosity in Pa s, non-negative.
    ///
    /// Returns: `Material` in liquid mode.
    #[must_use]
    pub fn liquid(density: f64, young_modulus: f64, poisson_ratio: f64, viscosity: f64) -> Self {
        let bulk_modulus = young_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio));
        Self {
            kind: MaterialKind::Liquid,
            density,
            shear_modulus: 0.0,
            lame_lambda: 0.0,
            bulk_modulus,
            yield_stress: 0.0,
            consistency: viscosity,
            flow_index: 1.0,
        }
    }

    /// Longitudinal wave speed in m/s used for the CFL restriction.
    #[must_use]
    pub fn wave_speed(&self) -> f64 {
        let modulus = match self.kind {
            MaterialKind::Paste => self.lame_lambda + 2.0 * self.shear_modulus,
            MaterialKind::Liquid => self.bulk_modulus,
        };
        (modulus / self.density).sqrt()
    }

    /// Kinematic viscosity-like diffusivity in m^2/s used for the viscous
    /// stability restriction. Zero for a paste (its viscous response is implicit).
    #[must_use]
    pub fn diffusivity(&self) -> f64 {
        match self.kind {
            MaterialKind::Paste => 0.0,
            MaterialKind::Liquid => self.consistency / self.density,
        }
    }
}

/// Result of the scalar Herschel-Bulkley radial return.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReturnSolution {
    /// Equivalent (von Mises) stress after the return in Pa.
    pub equivalent_stress: f64,
    /// Equivalent plastic strain increment (dimensionless).
    pub plastic_increment: f64,
    /// Newton/bisection iterations used.
    pub iterations: usize,
}

/// Solve the implicit Herschel-Bulkley radial return.
///
/// The discrete flow rule `sigma = sigma_tr - 3 mu dt eps_dot_p` combined with the
/// Herschel-Bulkley law `eps_dot_p = ((sigma - sigma_y) / K)^(1/n)` gives a scalar
/// residual `f(sigma) = sigma - sigma_tr + 3 mu dt ((sigma - sigma_y) / K)^(1/n)`
/// which is strictly increasing on `[sigma_y, sigma_tr]` with `f(sigma_y) < 0` and
/// `f(sigma_tr) > 0`, so a bracketed Newton iteration converges unconditionally.
/// A zero consistency degenerates to perfect plasticity (`sigma = sigma_y`).
///
/// Parameters:
/// - `trial_stress` (`f64`): trial equivalent stress in Pa, must exceed `yield_stress`.
/// - `yield_stress` (`f64`): yield stress in Pa, non-negative.
/// - `consistency` (`f64`): Herschel-Bulkley consistency in Pa s^n, non-negative.
/// - `flow_index` (`f64`): flow index `n`, positive.
/// - `shear_modulus` (`f64`): shear modulus in Pa, positive.
/// - `dt` (`f64`): time step in s, positive.
///
/// Returns: `Result<ReturnSolution, ConstitutiveError>`.
///
/// # Errors
///
/// `ReturnNotConverged` if the iteration cap is hit, `NonFinite` on
/// non-finite inputs.
pub fn solve_herschel_bulkley_return(
    trial_stress: f64,
    yield_stress: f64,
    consistency: f64,
    flow_index: f64,
    shear_modulus: f64,
    dt: f64,
) -> Result<ReturnSolution, ConstitutiveError> {
    let inputs = [
        trial_stress,
        yield_stress,
        consistency,
        flow_index,
        shear_modulus,
        dt,
    ];
    if inputs.iter().any(|value| !value.is_finite()) {
        return Err(ConstitutiveError::NonFinite {
            what: "return inputs",
        });
    }
    let viscous_scale = 3.0 * shear_modulus * dt;
    if trial_stress <= yield_stress {
        return Ok(ReturnSolution {
            equivalent_stress: trial_stress,
            plastic_increment: 0.0,
            iterations: 0,
        });
    }
    if consistency <= 0.0 {
        return Ok(ReturnSolution {
            equivalent_stress: yield_stress,
            plastic_increment: (trial_stress - yield_stress) / (3.0 * shear_modulus),
            iterations: 0,
        });
    }
    let exponent = 1.0 / flow_index;
    let residual = |sigma: f64| -> f64 {
        let over = ((sigma - yield_stress) / consistency).max(0.0);
        sigma - trial_stress + viscous_scale * over.powf(exponent)
    };
    let mut lower = yield_stress;
    let mut upper = trial_stress;
    let mut sigma = f64::midpoint(lower, upper);
    let tolerance = RETURN_RELATIVE_TOLERANCE * trial_stress.max(1.0);
    let mut last_residual = residual(sigma);
    for iteration in 1..=RETURN_MAX_ITERATIONS {
        if last_residual.abs() <= tolerance || (upper - lower) <= tolerance {
            return Ok(ReturnSolution {
                equivalent_stress: sigma,
                plastic_increment: (trial_stress - sigma) / (3.0 * shear_modulus),
                iterations: iteration,
            });
        }
        if last_residual > 0.0 {
            upper = sigma;
        } else {
            lower = sigma;
        }
        let over = ((sigma - yield_stress) / consistency).max(0.0);
        let derivative = if over > 0.0 {
            1.0 + viscous_scale * exponent * over.powf(exponent - 1.0) / consistency
        } else {
            f64::INFINITY
        };
        let newton = sigma - last_residual / derivative;
        sigma = if newton.is_finite() && newton > lower && newton < upper {
            newton
        } else {
            f64::midpoint(lower, upper)
        };
        last_residual = residual(sigma);
        if !last_residual.is_finite() {
            return Err(ConstitutiveError::NonFinite {
                what: "return residual",
            });
        }
    }
    Err(ConstitutiveError::ReturnNotConverged {
        trial_stress,
        residual: last_residual,
    })
}

/// Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors as columns, or `None`
/// if the sweep cap is hit or the input is non-finite. Jacobi is used instead of
/// a closed-form solver because it stays accurate for the nearly degenerate
/// spectra (near-identity stretches) that dominate MPM elastic states.
///
/// References: Golub & Van Loan, Matrix Computations, section 8.5.
#[must_use]
pub fn symmetric_eigen3(matrix: &Matrix3<f64>) -> Option<(Vector3<f64>, Matrix3<f64>)> {
    if !matrix.iter().all(|eigenvectors| eigenvectors.is_finite()) {
        return None;
    }
    let mut diagonalized = 0.5 * (matrix + matrix.transpose());
    let mut eigenvectors = Matrix3::identity();
    let scale = diagonalized.norm_squared().max(f64::MIN_POSITIVE);
    for _ in 0..JACOBI_MAX_SWEEPS {
        let off = diagonalized[(0, 1)].powi(2)
            + diagonalized[(0, 2)].powi(2)
            + diagonalized[(1, 2)].powi(2);
        if off <= JACOBI_TOLERANCE * scale {
            return Some((
                Vector3::new(
                    diagonalized[(0, 0)],
                    diagonalized[(1, 1)],
                    diagonalized[(2, 2)],
                ),
                eigenvectors,
            ));
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = diagonalized[(p, q)];
            if apq == 0.0 {
                continue;
            }
            let theta = (diagonalized[(q, q)] - diagonalized[(p, p)]) / (2.0 * apq);
            let tangent = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let cosine = 1.0 / (tangent * tangent + 1.0).sqrt();
            let sine = tangent * cosine;
            // diagonalized <- J^T diagonalized J with J the rotation in the (p, q) plane.
            for k in 0..3 {
                let akp = diagonalized[(k, p)];
                let akq = diagonalized[(k, q)];
                diagonalized[(k, p)] = cosine * akp - sine * akq;
                diagonalized[(k, q)] = sine * akp + cosine * akq;
            }
            for k in 0..3 {
                let apk = diagonalized[(p, k)];
                let aqk = diagonalized[(q, k)];
                diagonalized[(p, k)] = cosine * apk - sine * aqk;
                diagonalized[(q, k)] = sine * apk + cosine * aqk;
            }
            for k in 0..3 {
                let vkp = eigenvectors[(k, p)];
                let vkq = eigenvectors[(k, q)];
                eigenvectors[(k, p)] = cosine * vkp - sine * vkq;
                eigenvectors[(k, q)] = sine * vkp + cosine * vkq;
            }
        }
    }
    None
}

/// Outcome of a paste constitutive update.
#[derive(Debug, Clone, PartialEq)]
pub struct PasteUpdate {
    /// Updated elastic deformation gradient.
    pub deformation: Matrix3<f64>,
    /// Kirchhoff stress `tau = P F^T` for the updated state.
    pub kirchhoff: Matrix3<f64>,
    /// Equivalent plastic strain increment.
    pub plastic_increment: f64,
    /// Plastic dissipation per unit reference volume in J/m^3.
    pub dissipation_density: f64,
    /// Equivalent stress after the update in Pa.
    pub equivalent_stress: f64,
}

/// Perform the Hencky-elastic Herschel-Bulkley update for one particle.
///
/// Parameters:
/// - `trial` (`&Matrix3<f64>`): trial elastic deformation gradient `(I + dt C) F`.
/// - `material` (`&Material`): paste parameters.
/// - `dt` (`f64`): time step in s.
///
/// Returns: `Result<PasteUpdate, ConstitutiveError>` with the returned deformation
/// gradient (volumetric part untouched, deviatoric Hencky strain scaled by the
/// radial return) and the Kirchhoff stress.
///
/// # Errors
///
/// any [`ConstitutiveError`]; the solver rejects the step.
pub fn update_paste(
    trial: &Matrix3<f64>,
    material: &Material,
    dt: f64,
) -> Result<PasteUpdate, ConstitutiveError> {
    let determinant = trial.determinant();
    if !determinant.is_finite() {
        return Err(ConstitutiveError::NonFinite {
            what: "trial determinant",
        });
    }
    if determinant <= 0.0 {
        return Err(ConstitutiveError::InvertedDeformation { determinant });
    }
    // Left Cauchy-Green tensor b = F F^T = U diag(sigma^2) U^T; V is never needed
    // because F_new = U diag(sigma_new / sigma) U^T F.
    let left_cauchy_green = trial * trial.transpose();
    let (eigenvalues, u) =
        symmetric_eigen3(&left_cauchy_green).ok_or(ConstitutiveError::EigenFailed)?;
    let mut stretches = Vector3::zeros();
    let mut hencky = Vector3::zeros();
    for axis in 0..3 {
        let stretch = eigenvalues[axis].max(0.0).sqrt();
        if !stretch.is_finite() || stretch < MIN_STRETCH {
            return Err(ConstitutiveError::DegenerateStretch { stretch });
        }
        stretches[axis] = stretch;
        hencky[axis] = stretch.ln();
    }
    let volumetric = hencky.sum();
    let mean = volumetric / 3.0;
    let deviatoric = hencky - Vector3::repeat(mean);
    let mu = material.shear_modulus;
    // Deviatoric Kirchhoff stress s = 2 mu e; von Mises sigma_eq = sqrt(3/2) |s|.
    let deviatoric_norm = deviatoric.norm();
    let trial_equivalent = (1.5f64).sqrt() * 2.0 * mu * deviatoric_norm;
    let (scale, plastic_increment, equivalent_stress) =
        if trial_equivalent > material.yield_stress && deviatoric_norm > 0.0 {
            let solution = solve_herschel_bulkley_return(
                trial_equivalent,
                material.yield_stress,
                material.consistency,
                material.flow_index,
                mu,
                dt,
            )?;
            (
                solution.equivalent_stress / trial_equivalent,
                solution.plastic_increment,
                solution.equivalent_stress,
            )
        } else {
            (1.0, 0.0, trial_equivalent)
        };
    let deviatoric_new = deviatoric * scale;
    let hencky_new = deviatoric_new + Vector3::repeat(mean);
    let stretch_ratio = hencky_new.map(f64::exp).component_div(&stretches);
    let deformation = u * Matrix3::from_diagonal(&stretch_ratio) * u.transpose() * trial;
    let principal_stress =
        hencky_new * (2.0 * mu) + Vector3::repeat(material.lame_lambda * volumetric);
    let kirchhoff = u * Matrix3::from_diagonal(&principal_stress) * u.transpose();
    if !kirchhoff.iter().all(|value| value.is_finite()) {
        return Err(ConstitutiveError::NonFinite {
            what: "kirchhoff stress",
        });
    }
    // Dissipation density: sigma_eq * d(eps_p), evaluated with the returned stress.
    let dissipation_density = equivalent_stress * plastic_increment;
    Ok(PasteUpdate {
        deformation,
        kirchhoff,
        plastic_increment,
        dissipation_density,
        equivalent_stress,
    })
}

/// Outcome of a liquid constitutive update.
#[derive(Debug, Clone, PartialEq)]
pub struct LiquidUpdate {
    /// Updated volume ratio `J`.
    pub volume_ratio: f64,
    /// Kirchhoff stress `-p J I + 2 mu_visc J dev(D)`.
    pub kirchhoff: Matrix3<f64>,
    /// Pressure in Pa.
    pub pressure: f64,
}

/// Perform the weakly compressible liquid update for one particle.
///
/// Parameters:
/// - `volume_ratio` (`f64`): current `J`.
/// - `velocity_gradient` (`&Matrix3<f64>`): APIC affine matrix `C` (approximates grad v).
/// - `material` (`&Material`): liquid parameters.
/// - `dt` (`f64`): time step in s.
///
/// Returns: `Result<LiquidUpdate, ConstitutiveError>`.
///
/// # Errors
///
/// `InvertedDeformation` if `J` collapses, `NonFinite` on NaNs.
pub fn update_liquid(
    volume_ratio: f64,
    velocity_gradient: &Matrix3<f64>,
    material: &Material,
    dt: f64,
) -> Result<LiquidUpdate, ConstitutiveError> {
    let new_ratio = volume_ratio * (1.0 + dt * velocity_gradient.trace());
    if !new_ratio.is_finite() {
        return Err(ConstitutiveError::NonFinite {
            what: "volume ratio",
        });
    }
    if new_ratio <= MIN_STRETCH {
        return Err(ConstitutiveError::InvertedDeformation {
            determinant: new_ratio,
        });
    }
    let pressure = material.bulk_modulus * (1.0 / new_ratio - 1.0);
    let strain_rate = 0.5 * (velocity_gradient + velocity_gradient.transpose());
    let deviatoric_rate = strain_rate - Matrix3::identity() * (strain_rate.trace() / 3.0);
    let kirchhoff = Matrix3::identity() * (-pressure * new_ratio)
        + deviatoric_rate * (2.0 * material.consistency * new_ratio);
    if !kirchhoff.iter().all(|value| value.is_finite()) {
        return Err(ConstitutiveError::NonFinite {
            what: "liquid stress",
        });
    }
    Ok(LiquidUpdate {
        volume_ratio: new_ratio,
        kirchhoff,
        pressure,
    })
}

/// Liquid pressure from the equation of state, exposed for observables.
#[must_use]
pub fn liquid_pressure(material: &Material, volume_ratio: f64) -> f64 {
    material.bulk_modulus * (1.0 / volume_ratio - 1.0)
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact values by construction.
mod tests {
    use super::{
        ConstitutiveError, Material, solve_herschel_bulkley_return, symmetric_eigen3,
        update_liquid, update_paste,
    };
    use nalgebra::Matrix3;

    fn paste() -> Material {
        Material::paste(1000.0, 1.0e5, 0.3, 20.0, 5.0, 0.6)
    }

    #[test]
    fn return_is_elastic_below_yield() {
        let solution = solve_herschel_bulkley_return(10.0, 20.0, 5.0, 0.6, 1.0e4, 1e-3);
        assert_eq!(solution.map(|s| s.plastic_increment), Ok(0.0));
    }

    #[test]
    fn return_solves_scalar_residual_to_tolerance() {
        for flow_index in [0.3, 0.6, 1.0, 1.5] {
            let (trial, yield_stress, consistency, mu, dt) = (500.0, 20.0, 5.0, 1.0e4, 1e-3);
            let solution =
                solve_herschel_bulkley_return(trial, yield_stress, consistency, flow_index, mu, dt)
                    .unwrap_or_else(|e| panic!("solve failed: {e}"));
            let sigma = solution.equivalent_stress;
            let rate = ((sigma - yield_stress) / consistency).powf(1.0 / flow_index);
            let residual = sigma - trial + 3.0 * mu * dt * rate;
            assert!(
                residual.abs() < 1e-8 * trial,
                "n={flow_index} residual={residual}"
            );
            assert!(sigma > yield_stress && sigma < trial);
            assert!((solution.plastic_increment - rate * dt).abs() < 1e-6 * rate * dt);
        }
    }

    #[test]
    fn zero_consistency_is_perfect_plasticity() {
        let solution = solve_herschel_bulkley_return(500.0, 20.0, 0.0, 1.0, 1.0e4, 1e-3);
        assert_eq!(solution.map(|s| s.equivalent_stress), Ok(20.0));
    }

    #[test]
    fn non_finite_inputs_are_rejected() {
        let result = solve_herschel_bulkley_return(f64::NAN, 20.0, 5.0, 1.0, 1.0e4, 1e-3);
        assert!(matches!(result, Err(ConstitutiveError::NonFinite { .. })));
    }

    #[test]
    fn jacobi_eigensolver_reconstructs_matrix() {
        let m = Matrix3::new(2.0, -1.0, 0.3, -1.0, 3.0, 0.5, 0.3, 0.5, 1.5);
        let (values, vectors) = symmetric_eigen3(&m).unwrap_or_else(|| unreachable!());
        let rebuilt = vectors * Matrix3::from_diagonal(&values) * vectors.transpose();
        assert!((rebuilt - m).norm() < 1e-12);
        assert!((vectors * vectors.transpose() - Matrix3::identity()).norm() < 1e-12);
        let near_identity = Matrix3::identity() * 1.0
            + Matrix3::new(0.0, 1e-9, 0.0, 1e-9, 0.0, 0.0, 0.0, 0.0, 2e-9);
        let (values, vectors) = symmetric_eigen3(&near_identity).unwrap_or_else(|| unreachable!());
        let rebuilt = vectors * Matrix3::from_diagonal(&values) * vectors.transpose();
        assert!((rebuilt - near_identity).norm() < 1e-15);
        assert!(symmetric_eigen3(&(Matrix3::identity() * f64::NAN)).is_none());
    }

    #[test]
    fn inverted_deformation_is_rejected() {
        let inverted = Matrix3::from_diagonal(&nalgebra::Vector3::new(-1.0, 1.0, 1.0));
        let result = update_paste(&inverted, &paste(), 1e-3);
        assert!(matches!(
            result,
            Err(ConstitutiveError::InvertedDeformation { .. })
        ));
    }

    #[test]
    fn hydrostatic_compression_stays_elastic_and_matches_hencky() {
        let material = paste();
        let stretch: f64 = 0.99;
        let trial = Matrix3::identity() * stretch;
        let update = update_paste(&trial, &material, 1e-3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(update.plastic_increment, 0.0);
        let expected = (2.0 * material.shear_modulus + 3.0 * material.lame_lambda) * stretch.ln();
        assert!((update.kirchhoff[(0, 0)] - expected).abs() < 1e-9 * expected.abs());
        assert!((update.deformation - trial).norm() < 1e-12);
    }

    /// Drive a single material point in simple shear and verify that the steady
    /// state shear stress reproduces the Herschel-Bulkley law exactly. At steady
    /// state the implicit update is exact independent of `dt`; the elastic strain
    /// is tiny because `mu` is large relative to the flow stress.
    #[test]
    fn steady_simple_shear_reproduces_herschel_bulkley_law() {
        let material = Material::paste_from_shear_rheology(1000.0, 1.0e6, 0.3, 20.0, 5.0, 0.6);
        let shear_rate = 2.0;
        let dt = 1e-4;
        let mut deformation = Matrix3::identity();
        let mut velocity_gradient = Matrix3::zeros();
        velocity_gradient[(0, 2)] = shear_rate;
        let mut last = None;
        for _ in 0..20_000 {
            let trial = (Matrix3::identity() + velocity_gradient * dt) * deformation;
            let update = update_paste(&trial, &material, dt).unwrap_or_else(|e| panic!("{e}"));
            deformation = update.deformation;
            last = Some(update);
        }
        let update = last.unwrap_or_else(|| unreachable!());
        // Shear-rheometer law: tau = tau_y + K_s gamma_dot^n.
        let expected_shear = 20.0 + 5.0 * shear_rate.powf(0.6);
        let expected_equivalent = 3f64.sqrt() * expected_shear;
        let plastic_rate = shear_rate / 3f64.sqrt();
        let relative = (update.equivalent_stress - expected_equivalent).abs() / expected_equivalent;
        assert!(
            relative < 1e-6,
            "sigma_eq={} expected={expected_equivalent}",
            update.equivalent_stress
        );
        let shear = update.kirchhoff[(0, 2)];
        assert!(
            (shear - expected_shear).abs() / expected_shear < 1e-6,
            "tau_xz={shear}"
        );
        // Plastic rate consistent with the increment.
        assert!((update.plastic_increment / dt - plastic_rate).abs() / plastic_rate < 1e-6);
        // Elastic strain remains small: F stays close to a rotation times identity.
        assert!((deformation.determinant() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn liquid_pressure_follows_linear_eos() {
        let material = Material::liquid(1000.0, 1.0e5, 0.2, 1e-3);
        let gradient = Matrix3::identity() * -1.0; // volumetric contraction at rate 3/s
        let update =
            update_liquid(1.0, &gradient, &material, 1e-3).unwrap_or_else(|e| panic!("{e}"));
        let expected_j = 1.0 - 3e-3;
        assert!((update.volume_ratio - expected_j).abs() < 1e-15);
        let expected_p = material.bulk_modulus * (1.0 / expected_j - 1.0);
        assert!((update.pressure - expected_p).abs() < 1e-9);
        assert!((update.kirchhoff[(0, 0)] + expected_p * expected_j).abs() < 1e-9);
        assert!(update.kirchhoff[(0, 1)].abs() < 1e-15);
    }

    #[test]
    fn liquid_shear_stress_is_newtonian() {
        let material = Material::liquid(1000.0, 1.0e5, 0.2, 0.5);
        let mut gradient = Matrix3::zeros();
        gradient[(0, 2)] = 4.0;
        let update =
            update_liquid(1.0, &gradient, &material, 1e-4).unwrap_or_else(|e| panic!("{e}"));
        // tau_xz = mu * (dv_x/dz + dv_z/dx) * J = 0.5 * 4 * 1.
        assert!((update.kirchhoff[(0, 2)] - 2.0).abs() < 1e-12);
        assert!(update.pressure.abs() < 1e-12);
    }

    #[test]
    fn liquid_collapse_is_rejected() {
        let material = Material::liquid(1000.0, 1.0e5, 0.2, 0.0);
        let gradient = Matrix3::identity() * -1.0e6;
        let result = update_liquid(1.0, &gradient, &material, 1e-3);
        assert!(matches!(
            result,
            Err(ConstitutiveError::InvertedDeformation { .. })
        ));
    }
}
