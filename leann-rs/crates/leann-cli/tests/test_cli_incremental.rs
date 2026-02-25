//! CLI Incremental Build Tests
//!
//! Tests the incremental build detection logic: sources manifest tracking,
//! "up to date" detection, new-file detection, and `--force` bypass.
//! No embedding provider needed — these paths exit before reaching embedding.

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

/// Create a minimal fake index (meta.json only, no sources manifest).
fn create_fake_index(root: &std::path::Path, name: &str) {
    let index_dir = root.join(".leann").join("indexes").join(name);
    fs::create_dir_all(&index_dir).unwrap();

    let meta = serde_json::json!({
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": "test-model",
        "embedding_mode": "sentence-transformers",
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

/// Create a sources manifest file for an existing fake index.
/// `files` maps canonical file paths to mtime values.
fn create_manifest(
    root: &std::path::Path,
    index_name: &str,
    files: &std::collections::HashMap<String, f64>,
) {
    let index_dir = root.join(".leann").join("indexes").join(index_name);
    let manifest = serde_json::json!({
        "sources": files,
    });
    fs::write(
        index_dir.join("documents.leann.sources.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

/// Create a text file with some content and return its canonical path.
fn create_text_file(dir: &std::path::Path, name: &str, content: &str) -> String {
    let path = dir.join(name);
    fs::write(&path, content).unwrap();
    fs::canonicalize(&path)
        .unwrap()
        .to_string_lossy()
        .to_string()
}

/// Get mtime of a file as f64 seconds since epoch.
fn get_mtime(path: &str) -> f64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .map(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64()
        })
        .unwrap_or(0.0)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

/// Existing index without manifest → "built before incremental support".
/// This path exits immediately, no document loading needed.
#[test]
fn test_build_pre_incremental_index_without_force() {
    let dir = tempfile::tempdir().unwrap();
    create_fake_index(dir.path(), "old-idx");

    // Create a docs directory so the CLI has something to point at
    let docs_dir = dir.path().join("docs");
    fs::create_dir_all(&docs_dir).unwrap();
    fs::write(docs_dir.join("hello.txt"), "hello world").unwrap();

    let output = Command::new(leann_bin())
        .args(["build", "old-idx", "--docs", docs_dir.to_str().unwrap()])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann build");

    assert!(
        output.status.success(),
        "Should exit 0, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("built before incremental support"),
        "Should mention 'built before incremental support': {}",
        stdout
    );
    assert!(
        stdout.contains("--force"),
        "Should suggest --force: {}",
        stdout
    );
}

/// Existing index with manifest, all files match → "up to date".
#[test]
fn test_build_up_to_date() {
    let dir = tempfile::tempdir().unwrap();
    create_fake_index(dir.path(), "uptodate");

    // Create docs directory with a text file
    let docs_dir = dir.path().join("docs");
    fs::create_dir_all(&docs_dir).unwrap();
    let canonical = create_text_file(&docs_dir, "notes.txt", "some notes here");
    let mtime = get_mtime(&canonical);

    // Create manifest with matching entry
    let mut sources = std::collections::HashMap::new();
    sources.insert(canonical, mtime);
    create_manifest(dir.path(), "uptodate", &sources);

    let output = Command::new(leann_bin())
        .args(["build", "uptodate", "--docs", docs_dir.to_str().unwrap()])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann build");

    assert!(
        output.status.success(),
        "Should exit 0, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("up to date"),
        "Should say 'up to date': {}",
        stdout
    );
}

/// Existing index with manifest, new file added → detects new file.
/// The command will proceed past detection to the embedding step which
/// will fail, but we verify the detection message appears in stdout.
#[test]
fn test_build_detects_new_files() {
    let dir = tempfile::tempdir().unwrap();
    create_fake_index(dir.path(), "partial");

    let docs_dir = dir.path().join("docs");
    fs::create_dir_all(&docs_dir).unwrap();

    // File that's already in the manifest
    let old_path = create_text_file(&docs_dir, "old.txt", "old content");
    let old_mtime = get_mtime(&old_path);

    // New file NOT in the manifest
    create_text_file(&docs_dir, "new.txt", "brand new content");

    let mut sources = std::collections::HashMap::new();
    sources.insert(old_path, old_mtime);
    create_manifest(dir.path(), "partial", &sources);

    let output = Command::new(leann_bin())
        .args(["build", "partial", "--docs", docs_dir.to_str().unwrap()])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann build");

    // The command may fail at the embedding step, but the detection message
    // should appear in stdout before that.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Updating index") && stdout.contains("1 new file(s)"),
        "Should detect 1 new file: {}",
        stdout
    );
    assert!(
        stdout.contains("full rebuild"),
        "Should mention full rebuild: {}",
        stdout
    );
}

/// `--force` on existing index with manifest skips incremental check.
/// It should NOT print "up to date" or "built before incremental support".
#[test]
fn test_build_force_bypasses_manifest() {
    let dir = tempfile::tempdir().unwrap();
    create_fake_index(dir.path(), "forced");

    let docs_dir = dir.path().join("docs");
    fs::create_dir_all(&docs_dir).unwrap();
    let canonical = create_text_file(&docs_dir, "file.txt", "some content");
    let mtime = get_mtime(&canonical);

    // Create manifest with matching entry (would be "up to date" without --force)
    let mut sources = std::collections::HashMap::new();
    sources.insert(canonical, mtime);
    create_manifest(dir.path(), "forced", &sources);

    let output = Command::new(leann_bin())
        .args([
            "build",
            "forced",
            "--force",
            "--docs",
            docs_dir.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann build");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should NOT be short-circuited by incremental logic
    assert!(
        !stdout.contains("up to date"),
        "Should NOT say 'up to date' with --force: {}",
        stdout
    );
    assert!(
        !stdout.contains("built before incremental support"),
        "Should NOT mention 'built before incremental support' with --force: {}",
        stdout
    );
    // Should proceed to building (shows chunk/build messages)
    assert!(
        stdout.contains("Loaded") || stdout.contains("Building index"),
        "Should proceed to build phase: {}",
        stdout
    );
}

/// New index (doesn't exist yet) proceeds to build regardless of --force.
/// No manifest check should happen.
#[test]
fn test_build_new_index_no_manifest_check() {
    let dir = tempfile::tempdir().unwrap();

    let docs_dir = dir.path().join("docs");
    fs::create_dir_all(&docs_dir).unwrap();
    create_text_file(&docs_dir, "readme.txt", "hello world");

    let output = Command::new(leann_bin())
        .args(["build", "brand-new", "--docs", docs_dir.to_str().unwrap()])
        .current_dir(dir.path())
        .output()
        .expect("Failed to run leann build");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should NOT hit any incremental path
    assert!(
        !stdout.contains("up to date"),
        "New index should not say 'up to date': {}",
        stdout
    );
    assert!(
        !stdout.contains("built before incremental support"),
        "New index should not mention pre-incremental: {}",
        stdout
    );
    // Should proceed to loading
    assert!(
        stdout.contains("Loaded"),
        "Should load documents: {}",
        stdout
    );
}

/// Build help text mentions incremental semantics for --force.
#[test]
fn test_build_help_force_text() {
    let output = Command::new(leann_bin())
        .args(["build", "--help"])
        .output()
        .expect("Failed to run leann build --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("incremental"),
        "Build --force help should mention 'incremental': {}",
        stdout
    );
}
