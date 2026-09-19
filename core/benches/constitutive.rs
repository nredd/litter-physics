//! Constitutive-kernel microbenchmark, not a full-visit performance claim.
//!
//! References: <https://doi.org/10.1145/2751541>.

use std::{hint::black_box, time::Instant};

use _core::mpm::constitutive::{Material, update_paste};
use nalgebra::Matrix3;

/// Measure repeated implicit paste updates with fixed, finite shear inputs.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let material = Material::paste_from_shear_rheology(1000.0, 10000.0, 0.2, 30.0, 10.0, 0.6);
    let trial = Matrix3::new(1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let calls = 10000_u32;
    let started = Instant::now();
    for _ in 0..calls {
        black_box(update_paste(
            black_box(&trial),
            black_box(&material),
            0.001,
        )?);
    }
    println!(
        "calls={calls} mean_update_s={:.9}",
        started.elapsed().as_secs_f64() / f64::from(calls)
    );
    Ok(())
}
