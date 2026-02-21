//! E2E-9: CLI List & Remove Integration Tests
//!
//! Tests `leann list` and `leann remove` commands via subprocess.
//! Creates fake index directory structures (no embedding server needed).

use std::fs;
use std::process::Command;

/// Get path to the leann binary (built by cargo).
fn leann_bin() -> String {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--bin", "leann", "--quiet"])
        .current_dir(env!("CARGO_MANIFEST_DIR"));
    let _ = cmd.status();

    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = workspace_root.join("target/debug/leann");
    bin_path.to_string_lossy().to_string()
}

/// Create a minimal fake index directory that `leann list` will recognize.
/// Convention: .leann/indexes/<name>/documents.leann.meta.json
fn create_fake_index(root: &std::path::Path, name: &str) {
    let index_dir = root.join(".leann").join("indexes").join(name);
    fs::create_dir_all(&index_dir).unwrap();

    let meta = serde_json::json!({
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": "test-model",
        "embedding_mode": "test",
        "dimensions": 64,
        "total_passages": 10,
        "passage_sources": [{"source_type": "jsonl", "path": "documents.leann.passages.jsonl"}],
    });
    fs::write(
        index_dir.join("documents.leann.meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();
}

/// `leann list` shows a created index.
#[test]
fn test_cli_list_with_index() {
    let dir = tempfile::tempdir().unwrap();
    create_fake_index(dir.path(), "my-test-docs");

    let output = Command::new(leann_bin())
        .args(["list"])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann list");

    assert!(
        output.status.success(),
        "leann list should succeed, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("my-test-docs"),
        "Output should contain 'my-test-docs': {}",
        stdout
    );
}

/// `leann list` shows multiple indexes.
#[test]
fn test_cli_list_multiple_indexes() {
    let dir = tempfile::tempdir().unwrap();
    create_fake_index(dir.path(), "docs-a");
    create_fake_index(dir.path(), "docs-b");

    let output = Command::new(leann_bin())
        .args(["list"])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann list");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("docs-a"), "Should list docs-a: {}", stdout);
    assert!(stdout.contains("docs-b"), "Should list docs-b: {}", stdout);
}

/// `leann remove --force <name>` deletes an index directory.
#[test]
fn test_cli_remove_force() {
    let dir = tempfile::tempdir().unwrap();
    create_fake_index(dir.path(), "to-delete");

    let index_dir = dir.path().join(".leann").join("indexes").join("to-delete");
    assert!(index_dir.exists(), "Index dir should exist before remove");

    let output = Command::new(leann_bin())
        .args(["remove", "to-delete", "--force"])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann remove");

    assert!(
        output.status.success(),
        "leann remove --force should succeed, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !index_dir.exists(),
        "Index dir should be deleted after remove --force"
    );
}

/// `leann remove <nonexistent> --force` handles gracefully.
#[test]
fn test_cli_remove_nonexistent() {
    let dir = tempfile::tempdir().unwrap();

    let output = Command::new(leann_bin())
        .args(["remove", "does-not-exist", "--force"])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann remove");

    // Should succeed (just prints "not found"), not crash
    assert!(
        output.status.success(),
        "leann remove nonexistent should not crash, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("not found") || stdout.contains("Not found"),
        "Should indicate index not found: {}",
        stdout
    );
}

/// `leann list` shows "Get started" when empty, then shows index after creating one.
#[test]
fn test_cli_list_then_remove_lifecycle() {
    let dir = tempfile::tempdir().unwrap();

    // Initially empty
    let output = Command::new(leann_bin())
        .args(["list"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Get started") || stdout.contains("leann build"),
        "Empty list should show getting started message: {}",
        stdout
    );

    // Create index
    create_fake_index(dir.path(), "lifecycle-test");

    // Should now appear
    let output = Command::new(leann_bin())
        .args(["list"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("lifecycle-test"),
        "Should list the index: {}",
        stdout
    );

    // Remove it
    let output = Command::new(leann_bin())
        .args(["remove", "lifecycle-test", "--force"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Should be empty again
    let output = Command::new(leann_bin())
        .args(["list"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Get started") || stdout.contains("leann build"),
        "After remove, list should be empty again: {}",
        stdout
    );
}
