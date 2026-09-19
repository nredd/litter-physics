//! Check the manuscript's worked examples against the actual constitutive kernel.
//!
//! References: `docs/formal/sections/constitutive.tex`, equations for the scalar
//! HB return, its isolated elastic-energy decrease, and the liquid volume update.

use _core::mpm::constitutive::{
    ConstitutiveError, Material, liquid_pressure, update_liquid, update_paste,
};
use nalgebra::{Matrix3, Vector3};

#[test]
fn manuscript_paste_return_and_energy_gap_match_native_kernel() -> Result<(), ConstitutiveError> {
    let material = Material::paste_from_shear_rheology(1000.0, 1000.0, 0.2, 30.0, 10.0, 0.6);
    let trial_stress = 80.0;
    let strain = trial_stress / (2.0 * 3_f64.sqrt() * material.shear_modulus);
    let trial = Matrix3::from_diagonal(&Vector3::new(strain.exp(), (-strain).exp(), 1.0));
    let update = update_paste(&trial, &material, 0.0014)?;
    assert!((update.equivalent_stress - 78.005_956_883_7).abs() < 1e-9);
    assert!((update.plastic_increment - 0.001_595_234_493).abs() < 1e-12);
    assert!((update.deformation.determinant() - 1.0).abs() < 1e-12);
    let elastic_drop =
        (trial_stress.powi(2) - update.equivalent_stress.powi(2)) / (6.0 * material.shear_modulus);
    let predicted_gap = 1.5 * material.shear_modulus * update.plastic_increment.powi(2);
    assert!((elastic_drop - update.dissipation_density - predicted_gap).abs() < 1e-12);
    Ok(())
}

#[test]
fn manuscript_water_eos_and_first_order_volume_update_match_kernel() -> Result<(), ConstitutiveError>
{
    let material = Material::liquid(1000.0, 1e5, 0.2, 1e-3);
    let ratio = (-material.density * 9.81 * 0.01 / material.bulk_modulus).exp();
    assert!((liquid_pressure(&material, ratio) - 98.186_663).abs() < 5e-7);
    let gradient = -Matrix3::identity();
    let update = update_liquid(1.0, &gradient, &material, 1e-3)?;
    assert!((update.volume_ratio - 0.997).abs() < 1e-14);
    let determinant_update = (Matrix3::identity() + gradient * 1e-3).determinant();
    assert!((determinant_update - update.volume_ratio - 2.999e-6).abs() < 1e-14);
    assert!(matches!(
        update_liquid(1.0, &gradient, &material, 0.4),
        Err(ConstitutiveError::InvertedDeformation { .. })
    ));
    Ok(())
}
