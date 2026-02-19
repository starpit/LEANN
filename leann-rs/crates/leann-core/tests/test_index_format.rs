//! E2E-8: Index File Format Validation
//!
//! Tests the on-disk format of LEANN indexes: JSONL passages, offset maps, meta.json schema.
//! Mirrors parts of Python test_diskann_partition.py (file format validation).

mod common;

use common::build_test_index;
use leann_core::index::{IndexMeta, IndexPaths};
use leann_core::passages::{load_id_map, write_id_map, write_passages, Passage, PassageManager};
use std::collections::HashMap;
use std::io::{BufRead, BufReader};

/// Each line of .passages.jsonl is valid JSON with required fields.
#[test]
fn test_passages_jsonl_format() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(15, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let file = std::fs::File::open(paths.passages_path()).unwrap();
    let reader = BufReader::new(file);
    let mut count = 0;

    for line in reader.lines() {
        let line = line.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("Invalid JSON on line {}: {}", count + 1, e));

        // Required fields
        assert!(
            parsed.get("id").is_some(),
            "Missing 'id' field on line {}",
            count + 1
        );
        assert!(
            parsed.get("text").is_some(),
            "Missing 'text' field on line {}",
            count + 1
        );

        let text = parsed["text"].as_str().unwrap();
        assert!(!text.is_empty(), "Empty text on line {}", count + 1);

        count += 1;
    }

    assert_eq!(count, 15, "Expected 15 passages, got {}", count);
}

/// ID map roundtrip: write, read back, verify exact match.
#[test]
fn test_id_map_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.ids.txt");

    let ids: Vec<String> = (0..50).map(|i| format!("doc_{}", i)).collect();
    write_id_map(&ids, &path).unwrap();

    let loaded = load_id_map(&path).unwrap();
    assert_eq!(loaded.len(), ids.len());
    for (a, b) in ids.iter().zip(loaded.iter()) {
        assert_eq!(a, b);
    }
}

/// Passage offset map enables correct random access.
#[test]
fn test_passages_offset_random_access() {
    let dir = tempfile::tempdir().unwrap();
    let passages_path = dir.path().join("test.passages.jsonl");
    let offset_path = dir.path().join("test.passages.idx");

    let passages: Vec<Passage> = (0..20)
        .map(|i| Passage {
            id: format!("p_{}", i),
            text: format!(
                "Passage number {} with some content about topic {}",
                i,
                i % 3
            ),
            metadata: {
                let mut m = HashMap::new();
                m.insert("index".to_string(), serde_json::json!(i));
                m
            },
        })
        .collect();

    let offset_map = write_passages(&passages, &passages_path, &offset_path).unwrap();
    assert_eq!(offset_map.len(), 20);

    // Load and verify random access
    let sources = vec![leann_core::index::PassageSource {
        source_type: "jsonl".to_string(),
        path: passages_path.to_string_lossy().to_string(),
        index_path: offset_path.to_string_lossy().to_string(),
        path_relative: None,
        index_path_relative: None,
    }];

    let manager = PassageManager::load(&sources, None).unwrap();

    // Access in random order
    for i in [15, 3, 0, 19, 7, 12] {
        let p = manager.get_passage(&format!("p_{}", i)).unwrap();
        assert!(
            p.text.contains(&format!("Passage number {}", i)),
            "Wrong passage for p_{}: '{}'",
            i,
            p.text
        );
    }
}

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

/// Passage sources in meta.json correctly reference existing files.
#[test]
fn test_passage_sources_reference_valid_files() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let meta = IndexMeta::load(&paths.meta_path()).unwrap();

    for source in &meta.passage_sources {
        assert_eq!(source.source_type, "jsonl");
        // The relative paths should resolve to existing files
        assert!(
            !source.path.is_empty(),
            "Passage source path should not be empty"
        );
    }

    // Loading passages should succeed
    let manager = PassageManager::load(&meta.passage_sources, Some(&paths.meta_path())).unwrap();
    assert_eq!(manager.len(), 10);
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
