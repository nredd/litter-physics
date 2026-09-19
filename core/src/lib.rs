//! Deterministic numerical kernels for litter-physics.
//!
//! References: <https://doi.org/10.1145/3197517.3201293>
use pyo3::prelude::*;

pub mod dem;
pub mod household;
pub mod mpm;
pub mod research;

/// Return the wire protocol version understood by this build.
#[pyfunction]
#[must_use]
pub fn protocol_version() -> u32 {
    1
}

/// Execute a validated JSON request without exposing per-particle FFI calls.
///
/// # Errors
///
/// Returns `ValueError` for invalid protocols, unsupported physics or numerical failures.
#[pyfunction(signature = (request_json, resume_json=None))]
pub fn run_json(request_json: &str, resume_json: Option<&str>) -> PyResult<String> {
    const MAX_JSON_BYTES: usize = 256 * 1024 * 1024;
    if request_json.len() > MAX_JSON_BYTES || resume_json.is_some_and(|v| v.len() > MAX_JSON_BYTES)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "native JSON input exceeds 256 MiB",
        ));
    }
    let request: serde_json::Value = serde_json::from_str(request_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid request JSON: {e}"))
    })?;
    let resume: Option<serde_json::Value> = resume_json
        .map(serde_json::from_str)
        .transpose()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid checkpoint JSON: {e}"))
        })?;
    let output = match request.get("mode").and_then(serde_json::Value::as_str) {
        Some("household") => household::run(&request, resume.as_ref()),
        Some("research") => research::run(&request, resume.as_ref()),
        _ => Err("mode must be household or research".into()),
    }
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    serde_json::to_string(&output)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Register the Python extension's public functions.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(protocol_version, module)?)?;
    module.add_function(wrap_pyfunction!(run_json, module)?)?;
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
