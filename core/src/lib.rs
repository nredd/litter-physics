//! Deterministic numerical kernels for litter-physics.
//!
//! References: <https://doi.org/10.1145/3197517.3201293>
use pyo3::prelude::*;

/// Return the wire protocol version understood by this build.
#[pyfunction]
#[must_use]
pub fn protocol_version() -> u32 {
    1
}

/// Register the Python extension's public functions.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(protocol_version, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::protocol_version;

    #[test]
    fn protocol_is_versioned() {
        assert_eq!(protocol_version(), 1);
    }
}
