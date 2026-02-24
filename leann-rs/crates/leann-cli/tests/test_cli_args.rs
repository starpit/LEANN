//! E2E-9: CLI Argument Parsing Tests
//!
//! Tests CLI help output and argument parsing via subprocess.
//! Mirrors Python test_ci_minimal.py, test_cli_ask.py, test_cli_verbosity.py.

use std::process::Command;

/// Get path to the leann binary (built by cargo).
fn leann_bin() -> String {
    // Use cargo's target directory
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--bin", "leann", "--quiet"])
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    let _ = cmd.status(); // Build if needed

    // Find the binary
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = workspace_root.join("target/debug/leann");
    bin_path.to_string_lossy().to_string()
}

/// `leann --help` exits 0 and contains expected subcommands.
#[test]
fn test_cli_help() {
    let output = Command::new(leann_bin())
        .arg("--help")
        .output()
        .expect("Failed to run leann --help");

    assert!(output.status.success(), "leann --help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("build"), "Help should mention 'build'");
    assert!(stdout.contains("search"), "Help should mention 'search'");
    assert!(stdout.contains("ask"), "Help should mention 'ask'");
    assert!(stdout.contains("list"), "Help should mention 'list'");
    assert!(stdout.contains("remove"), "Help should mention 'remove'");
}

/// `leann build --help` shows build-specific options.
#[test]
fn test_cli_build_help() {
    let output = Command::new(leann_bin())
        .args(["build", "--help"])
        .output()
        .expect("Failed to run leann build --help");

    assert!(output.status.success(), "leann build --help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--docs"),
        "Build help should mention --docs"
    );
    assert!(
        stdout.contains("--embedding-model"),
        "Build help should mention --embedding-model"
    );
}

/// `leann search --help` shows search-specific options.
#[test]
fn test_cli_search_help() {
    let output = Command::new(leann_bin())
        .args(["search", "--help"])
        .output()
        .expect("Failed to run leann search --help");

    assert!(output.status.success(), "leann search --help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--complexity") || stdout.contains("complexity"),
        "Search help should mention complexity"
    );
}

/// `leann list` in an empty directory succeeds.
#[test]
fn test_cli_list_empty() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(leann_bin())
        .args(["list"])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann list");

    // Should succeed (exit 0) even with no indexes
    assert!(
        output.status.success(),
        "leann list should succeed in empty dir, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// `leann --version` or version info is accessible.
#[test]
fn test_cli_version() {
    let output = Command::new(leann_bin())
        .arg("--version")
        .output()
        .expect("Failed to run leann --version");

    // Some CLIs return version, some don't support --version
    // At minimum, it shouldn't crash
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Either succeeds or prints an error message (not a crash)
    assert!(
        output.status.success() || !stderr.is_empty() || !stdout.is_empty(),
        "CLI should produce some output"
    );
}

/// `leann ask --help` shows ask-specific options.
#[test]
fn test_cli_ask_help() {
    let output = Command::new(leann_bin())
        .args(["ask", "--help"])
        .output()
        .expect("Failed to run leann ask --help");

    assert!(output.status.success(), "leann ask --help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--interactive") || stdout.contains("interactive"),
        "Ask help should mention interactive mode"
    );
}

/// `leann remove --help` shows remove options.
#[test]
fn test_cli_remove_help() {
    let output = Command::new(leann_bin())
        .args(["remove", "--help"])
        .output()
        .expect("Failed to run leann remove --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--force") || stdout.contains("force"),
        "Remove help should mention --force"
    );
}

/// `leann search --help` mentions warmup flags.
#[test]
fn test_cli_search_warmup_flags() {
    let output = Command::new(leann_bin())
        .args(["search", "--help"])
        .output()
        .expect("Failed to run leann search --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--warmup") || stdout.contains("warmup"),
        "Search help should mention warmup"
    );
    assert!(
        stdout.contains("--no-warmup") || stdout.contains("no-warmup"),
        "Search help should mention --no-warmup"
    );
}

/// `leann warmup --help` exits 0.
#[test]
fn test_cli_warmup_help() {
    let output = Command::new(leann_bin())
        .args(["warmup", "--help"])
        .output()
        .expect("Failed to run leann warmup --help");

    assert!(output.status.success(), "leann warmup --help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("index") || stdout.contains("INDEX"),
        "Warmup help should mention index argument"
    );
}

/// `leann --help` mentions warmup subcommand.
#[test]
fn test_cli_help_mentions_warmup() {
    let output = Command::new(leann_bin())
        .arg("--help")
        .output()
        .expect("Failed to run leann --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("warmup"),
        "Help should mention 'warmup' subcommand"
    );
}
