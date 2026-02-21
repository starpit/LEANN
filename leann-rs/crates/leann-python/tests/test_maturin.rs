//! Integration test: builds the Python wheel with `maturin develop` and runs pytest.
//!
//! Run with:
//!   cd crates/leann-python && cargo test --test test_maturin
//!
//! Requires: maturin and a Python interpreter available on PATH.
//! Pytest will be installed automatically if missing.

use std::path::PathBuf;
use std::process::Command;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn find_python() -> String {
    // Prefer python3, fall back to python
    for name in &["python3", "python"] {
        if Command::new(name)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return name.to_string();
        }
    }
    panic!("No python3 or python found on PATH");
}

fn ensure_pytest(python: &str) {
    let check = Command::new(python)
        .args(["-c", "import pytest"])
        .output()
        .expect("Failed to run python");

    if !check.status.success() {
        eprintln!("pytest not found, installing...");

        // Try uv first (used by this project), then fall back to pip
        let install = Command::new("uv")
            .args(["pip", "install", "pytest"])
            .output()
            .or_else(|_| {
                Command::new(python)
                    .args(["-m", "pip", "install", "--quiet", "pytest"])
                    .output()
            })
            .expect("Failed to install pytest (tried uv and pip)");

        assert!(
            install.status.success(),
            "Failed to install pytest:\n{}",
            String::from_utf8_lossy(&install.stderr),
        );
    }
}

#[test]
fn test_maturin_develop_and_pytest() {
    let crate_dir = crate_dir();
    let python = find_python();

    // Step 1: maturin develop (builds and installs into current Python env)
    let develop = Command::new("maturin")
        .arg("develop")
        .arg("--release")
        .current_dir(&crate_dir)
        .output()
        .expect("Failed to run `maturin develop`. Is maturin installed?");

    assert!(
        develop.status.success(),
        "maturin develop failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&develop.stdout),
        String::from_utf8_lossy(&develop.stderr),
    );

    // Step 2: ensure pytest is available
    ensure_pytest(&python);

    // Step 3: run pytest
    let pytest = Command::new(&python)
        .args(["-m", "pytest", "tests/test_bindings.py", "-v"])
        .current_dir(&crate_dir)
        .output()
        .expect("Failed to run pytest");

    // Print output for CI visibility
    let stdout = String::from_utf8_lossy(&pytest.stdout);
    let stderr = String::from_utf8_lossy(&pytest.stderr);
    println!("{stdout}");
    if !stderr.is_empty() {
        eprintln!("{stderr}");
    }

    assert!(
        pytest.status.success(),
        "pytest failed (exit code {:?}):\n{stdout}\n{stderr}",
        pytest.status.code(),
    );
}
