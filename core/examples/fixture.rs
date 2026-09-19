//! Run a native fixture from an existing frozen `request.json`.
//!
//! Usage: `cargo run --release --example fixture -- outputs/demo/request.json`
//! References: `docs/wire-contract.md`.

use std::{env, fs, io, path::PathBuf};

/// Execute an explicit JSON input and print only its scalar observables.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 1 {
        return Err(io::Error::other("provide one frozen request.json path").into());
    }
    let path = PathBuf::from(&args[0]).canonicalize()?;
    let input: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let output = match input["mode"].as_str() {
        Some("household") => _core::household::run(&input, None),
        Some("research") => _core::research::run(&input, None),
        _ => Err("unknown simulation mode".into()),
    }
    .map_err(io::Error::other)?;
    println!("{}", serde_json::to_string_pretty(&output["observables"])?);
    if output["status"] != "completed" {
        return Err(io::Error::other("fixture stopped at its wall budget").into());
    }
    Ok(())
}
