//! E2E-8: Index File Format Validation
//!
//! Tests the on-disk format of LEANN indexes: meta.json schema, distance metrics.

mod common;

use common::build_test_index;
use leann_core::index::{IndexMeta, IndexPaths};

/// .meta.json has all required schema fields.
#[test]
fn test_meta_json_schema() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    // Parse as raw JSON to check field presence
    let content = std::fs::read_to_string(paths.meta_path()).unwrap();
    let raw: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert!(raw.get("version").is_some(), "Missing 'version'");
    assert!(raw.get("backend_name").is_some(), "Missing 'backend_name'");
    assert!(
        raw.get("embedding_model").is_some(),
        "Missing 'embedding_model'"
    );
    assert!(raw.get("dimensions").is_some(), "Missing 'dimensions'");
    assert!(
        raw.get("passage_sources").is_some(),
        "Missing 'passage_sources'"
    );
    assert!(
        raw.get("embedding_mode").is_some(),
        "Missing 'embedding_mode'"
    );

    // Also verify it parses into IndexMeta correctly
    let meta: IndexMeta = serde_json::from_str(&content).unwrap();
    assert!(meta.dimensions > 0);
    assert!(!meta.passage_sources.is_empty());
}

/// Distance metric from backend_kwargs.
#[test]
fn test_meta_distance_metric() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let meta = IndexMeta::load(&paths.meta_path()).unwrap();

    // The builder sets L2 distance metric
    let metric = meta.distance_metric();
    assert_eq!(metric, leann_core::index::DistanceMetric::L2);
}

/// Requires recompute flag is correctly set.
#[test]
fn test_meta_requires_recompute() {
    let dir = tempfile::tempdir().unwrap();

    // Build with recompute=true
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);
    let meta = IndexMeta::load(&paths.meta_path()).unwrap();
    assert!(meta.requires_recompute(), "Should require recompute");

    // Build with recompute=false
    let dir2 = tempfile::tempdir().unwrap();
    let index_path2 = build_test_index(10, dir2.path(), false, false).unwrap();
    let paths2 = IndexPaths::new(&index_path2);
    let meta2 = IndexMeta::load(&paths2.meta_path()).unwrap();
    assert!(!meta2.requires_recompute(), "Should not require recompute");
}
