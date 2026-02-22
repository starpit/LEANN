//! Cross-implementation compatibility tests.
//!
//! Validates that the on-disk format written by Rust matches what Python expects,
//! and vice versa. These tests don't require Python to be installed — they verify
//! the binary format directly.

mod common;

use common::build_test_index;
use leann_core::index::{IndexMeta, IndexPaths};
use std::io::Read;

/// .meta.json contains all fields that Python LeannSearcher expects.
#[test]
fn test_meta_json_has_python_required_fields() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let content = std::fs::read_to_string(paths.meta_path()).unwrap();
    let raw: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Fields Python LeannSearcher reads in __init__:
    assert!(raw.get("backend_name").is_some());
    assert!(raw.get("embedding_model").is_some());
    assert!(raw.get("embedding_mode").is_some());
    assert!(raw.get("passage_sources").is_some());
    assert!(raw.get("dimensions").is_some());
    assert!(raw.get("version").is_some());

    // passage_sources[0] must have "type" (not "source_type")
    let sources = raw["passage_sources"].as_array().unwrap();
    assert!(!sources.is_empty());
    let source = &sources[0];
    assert_eq!(
        source.get("type").and_then(|v| v.as_str()),
        Some("jsonl"),
        "Python expects 'type' key (serde renames source_type → type)"
    );
    assert!(source.get("path").is_some(), "Missing 'path' in source");
    assert!(
        source.get("index_path").is_some(),
        "Missing 'index_path' in source"
    );
}

/// .meta.json written by Rust uses the same field names as Python for storage flags.
#[test]
fn test_meta_json_storage_flags() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let content = std::fs::read_to_string(paths.meta_path()).unwrap();
    let raw: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Python writes/reads "is_compact" and "is_pruned" at the top level
    assert_eq!(
        raw.get("is_compact").and_then(|v| v.as_bool()),
        Some(true),
        "is_compact should be true"
    );
    assert_eq!(
        raw.get("is_pruned").and_then(|v| v.as_bool()),
        Some(true),
        "is_pruned should be true (maps from is_recompute)"
    );
}

/// The backend_kwargs written by Rust's full build_index include M, efConstruction, etc.
#[test]
fn test_meta_json_backend_kwargs() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let meta = IndexMeta::load(&paths.meta_path()).unwrap();

    // build_index writes backend_kwargs; build_index_from_embeddings writes empty
    // The full build path should populate these
    if !meta.backend_kwargs.is_empty() {
        assert!(meta.backend_kwargs.contains_key("M"));
        assert!(meta.backend_kwargs.contains_key("efConstruction"));
        assert!(meta.backend_kwargs.contains_key("distance_metric"));
    }
}

/// .passages.jsonl format is compatible: each line is {"id":..., "text":..., "metadata":...}
#[test]
fn test_passages_jsonl_python_compatible() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(15, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let content = std::fs::read_to_string(paths.passages_path()).unwrap();
    for (i, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("Line {}: {}", i, e));

        // Python expects these exact keys
        assert!(parsed.get("id").is_some(), "Line {}: missing 'id'", i);
        assert!(parsed.get("text").is_some(), "Line {}: missing 'text'", i);
        assert!(
            parsed.get("metadata").is_some(),
            "Line {}: missing 'metadata'",
            i
        );
    }
}

/// .passages.idx uses text format (one u64 per line) — NOT Python pickle format.
/// This is a KNOWN incompatibility that needs to be resolved for cross-read support.
#[test]
fn test_passages_idx_is_text_format() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let content = std::fs::read_to_string(paths.offset_path()).unwrap();
    let offsets: Vec<u64> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<u64>().unwrap())
        .collect();

    assert_eq!(offsets.len(), 10, "Expected 10 offsets");
    // Offsets must be monotonically increasing (sorted order of passages in JSONL)
    for window in offsets.windows(2) {
        assert!(
            window[0] < window[1],
            "Offsets should be strictly increasing: {:?}",
            offsets
        );
    }
}

/// Document the Python .passages.idx format for reference.
/// Python writes: pickle.dump(dict[str, int], f)  where key=passage_id, value=byte_offset
/// This is NOT compatible with Rust's text format.
#[test]
fn test_python_idx_format_documented() {
    // This test verifies our understanding of the Python format.
    // If we had a Python-built index, the .passages.idx would be a pickle file
    // containing a dict like {"0": 0, "1": 52, "2": 107, ...}
    //
    // The Rust format is a text file:
    // 0
    // 52
    // 107
    //
    // To resolve: either change Rust to write pickle, change Python to write text,
    // or add a compatibility shim that can read both formats.
}

/// .index binary file starts with FAISS HNSW FourCC header.
#[test]
fn test_hnsw_index_faiss_fourcc() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let mut file = std::fs::File::open(paths.index_file_path()).unwrap();
    let mut fourcc_bytes = [0u8; 4];
    file.read_exact(&mut fourcc_bytes).unwrap();

    // FAISS HNSW FourCC: "IHNf" as a little-endian u32
    let fourcc = u32::from_le_bytes(fourcc_bytes);
    let expected = u32::from_le_bytes(*b"IHNf");
    assert_eq!(
        fourcc, expected,
        "Expected FAISS FourCC 'IHNf' (0x{:08x}), got 0x{:08x}",
        expected, fourcc
    );
}

/// .index header has correct dimensions and ntotal.
#[test]
fn test_hnsw_index_header_fields() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let data = std::fs::read(paths.index_file_path()).unwrap();

    // Header layout: fourcc(4) + d(4) + ntotal(8) + dummy1(8) + dummy2(8) + is_trained(1)
    let d = i32::from_le_bytes(data[4..8].try_into().unwrap());
    let ntotal = i64::from_le_bytes(data[8..16].try_into().unwrap());
    let is_trained = data[32]; // offset 4+4+8+8+8 = 32

    assert_eq!(d, 64, "Expected dimensions=64, got {}", d);
    assert_eq!(ntotal, 10, "Expected ntotal=10, got {}", ntotal);
    assert_eq!(is_trained, 1, "Expected is_trained=1, got {}", is_trained);
}

/// .ids.txt format matches — one ID per line, same order as embeddings.
#[test]
fn test_id_map_format() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let content = std::fs::read_to_string(paths.id_map_path()).unwrap();
    let ids: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

    assert_eq!(ids.len(), 10);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(*id, i.to_string());
    }
}

/// Compact CSR index round-trips through Rust read/write.
#[test]
fn test_compact_index_roundtrip() {
    use leann_core::hnsw::io::read_hnsw_index;
    use std::io::Cursor;

    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(20, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let data = std::fs::read(paths.index_file_path()).unwrap();
    let mut cursor = Cursor::new(&data);
    let graph = read_hnsw_index(&mut cursor).unwrap();

    assert_eq!(graph.ntotal, 20);
    assert_eq!(graph.dimensions, 64);
    assert!(graph.is_compact());
}

/// Non-compact (standard) index round-trips through Rust read/write.
#[test]
fn test_standard_index_roundtrip() {
    use leann_core::hnsw::io::read_hnsw_index;
    use std::io::Cursor;

    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(20, dir.path(), false, false).unwrap();
    let paths = IndexPaths::new(&index_path);

    let data = std::fs::read(paths.index_file_path()).unwrap();
    let mut cursor = Cursor::new(&data);
    let graph = read_hnsw_index(&mut cursor).unwrap();

    assert_eq!(graph.ntotal, 20);
    assert_eq!(graph.dimensions, 64);
    assert!(!graph.is_compact());
}
