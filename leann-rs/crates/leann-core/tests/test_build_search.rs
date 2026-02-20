//! E2E-1: Core Build & Search Pipeline
//!
//! Tests the full pipeline: LeannBuilder → build_index → files on disk → read back → search.
//! Mirrors Python test_basic.py::test_backend_basic and test_basic.py::test_large_index.

mod common;

use common::{FakeEmbeddingProvider, build_test_index};
use leann_core::LeannBuilder;
use leann_core::embedding::EmbeddingProvider;
use leann_core::hnsw::build::build_hnsw;
use leann_core::hnsw::graph::HnswConfig;
use leann_core::hnsw::io::{read_hnsw_index, write_hnsw_compact, write_hnsw_standard};
use leann_core::hnsw::search::{SearchParams, search_hnsw};
use leann_core::index::{DistanceMetric, IndexMeta, IndexPaths};
use ndarray::Array2;
use std::collections::HashMap;
use std::io::Cursor;

/// Build index from 100 docs, read graph back, search with stored vectors, verify results.
/// (Python: test_backend_basic with 100 docs)
#[test]
fn test_build_and_search_100_docs() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(100, dir.path(), false, false).unwrap();
    let paths = IndexPaths::new(&index_path);

    // Read graph back
    let index_data = std::fs::read(paths.index_file_path()).unwrap();
    let graph = read_hnsw_index(&mut Cursor::new(index_data)).unwrap();

    // Extract stored vectors
    let flat_vectors = match &graph.vector_storage {
        leann_core::hnsw::graph::VectorStorage::Raw { data, .. } => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            floats
        }
        _ => panic!("Expected stored vectors (non-recompute build)"),
    };

    // Search for a query similar to document 0 ("topic_0")
    let provider = FakeEmbeddingProvider::new(64);
    let query_emb = provider
        .compute_embeddings(&["This is document 0 about topic_0".to_string()])
        .unwrap();
    let query: Vec<f32> = query_emb.row(0).to_vec();

    let params = SearchParams {
        ef_search: 128, // High ef for better recall
        ..Default::default()
    };

    let (labels, distances) = search_hnsw(&graph, &query, 5, &flat_vectors, &params);
    assert_eq!(labels.len(), 5, "Expected 5 results");
    assert_eq!(distances.len(), 5);

    // Distances should be sorted ascending (L2)
    for i in 1..distances.len() {
        assert!(
            distances[i] >= distances[i - 1] - 1e-6,
            "Distances not sorted: {} < {}",
            distances[i],
            distances[i - 1]
        );
    }

    // Top distances should be small (near 0) since query matches indexed docs
    assert!(
        distances[0] < 0.1,
        "Top result distance should be small, got {}",
        distances[0]
    );
}

/// Same test with 1000 docs — verifies scaling doesn't break.
/// (Python: test_large_index)
#[test]
fn test_build_and_search_1000_docs() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(1000, dir.path(), false, false).unwrap();
    let paths = IndexPaths::new(&index_path);

    let index_data = std::fs::read(paths.index_file_path()).unwrap();
    let graph = read_hnsw_index(&mut Cursor::new(index_data)).unwrap();
    assert_eq!(graph.ntotal, 1000);

    let flat_vectors = match &graph.vector_storage {
        leann_core::hnsw::graph::VectorStorage::Raw { data, .. } => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<f32>>(),
        _ => panic!("Expected stored vectors"),
    };

    let provider = FakeEmbeddingProvider::new(64);
    let query_emb = provider
        .compute_embeddings(&["document about topic_0".to_string()])
        .unwrap();
    let query: Vec<f32> = query_emb.row(0).to_vec();

    let params = SearchParams {
        ef_search: 64,
        ..Default::default()
    };
    let (labels, distances) = search_hnsw(&graph, &query, 10, &flat_vectors, &params);
    assert_eq!(labels.len(), 10, "Expected 10 results from 1000 docs");
}

/// Build index, verify expected files are created and non-empty.
#[test]
fn test_build_creates_expected_files() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(20, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let meta_path = paths.meta_path();
    let passages_path = paths.passages_path();
    let offset_path = paths.offset_path();
    let index_file = paths.index_file_path();
    let id_map = paths.id_map_path();

    assert!(meta_path.exists(), ".meta.json missing");
    assert!(passages_path.exists(), ".passages.jsonl missing");
    assert!(offset_path.exists(), ".passages.idx missing");
    assert!(index_file.exists(), ".index missing");
    assert!(id_map.exists(), ".ids.txt missing");

    // All files should be non-empty
    assert!(std::fs::metadata(&meta_path).unwrap().len() > 0);
    assert!(std::fs::metadata(&passages_path).unwrap().len() > 0);
    assert!(std::fs::metadata(&offset_path).unwrap().len() > 0);
    assert!(std::fs::metadata(&index_file).unwrap().len() > 0);
    assert!(std::fs::metadata(&id_map).unwrap().len() > 0);
}

/// Parse .meta.json and verify key fields.
#[test]
fn test_meta_json_content() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let meta = IndexMeta::load(&paths.meta_path()).unwrap();
    assert_eq!(meta.backend_name, "hnsw");
    assert_eq!(meta.embedding_model, "fake-test-model");
    assert_eq!(meta.dimensions, 64);
    assert_eq!(meta.version, "1.0");
    assert_eq!(meta.embedding_mode, "test");
    assert_eq!(meta.total_passages, Some(10));
    assert_eq!(meta.is_compact, Some(true));
    assert_eq!(meta.is_pruned, Some(true));
    assert!(!meta.passage_sources.is_empty());
    assert_eq!(meta.passage_sources[0].source_type, "jsonl");
}

/// Build with compact CSR and verify the graph reads back as compact.
#[test]
fn test_build_with_compact_csr() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let paths = IndexPaths::new(&index_path);

    let index_data = std::fs::read(paths.index_file_path()).unwrap();
    let graph = read_hnsw_index(&mut Cursor::new(index_data)).unwrap();
    assert!(graph.is_compact(), "Expected compact CSR format");
    assert!(graph.is_pruned(), "Expected pruned vector storage");
}

/// Build with non-compact format and verify stored vectors.
#[test]
fn test_build_standard_with_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(20, dir.path(), false, false).unwrap();
    let paths = IndexPaths::new(&index_path);

    let index_data = std::fs::read(paths.index_file_path()).unwrap();
    let graph = read_hnsw_index(&mut Cursor::new(index_data)).unwrap();
    assert!(!graph.is_compact(), "Expected standard format");
    assert!(!graph.is_pruned(), "Expected stored vectors");
}

/// Build with different distance metrics — all should succeed.
#[test]
fn test_build_with_distance_metrics() {
    let provider = FakeEmbeddingProvider::new(32);

    for metric in [
        DistanceMetric::L2,
        DistanceMetric::Cosine,
        DistanceMetric::Mips,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut builder = LeannBuilder::new("test-model", Some(32), "test");
        builder = builder
            .with_m(8)
            .with_ef_construction(20)
            .with_compact(false)
            .with_recompute(false)
            .with_distance_metric(metric);

        for i in 0..20 {
            let mut meta = HashMap::new();
            meta.insert("id".to_string(), serde_json::json!(i.to_string()));
            builder.add_text(&format!("Document {} about things", i), meta);
        }

        let index_path = dir.path().join("test_index");
        builder.build_index(&index_path, &provider).unwrap();

        // Read back and search
        let paths = IndexPaths::new(&index_path);
        let index_data = std::fs::read(paths.index_file_path()).unwrap();
        let graph = read_hnsw_index(&mut Cursor::new(index_data)).unwrap();

        let flat_vectors = match &graph.vector_storage {
            leann_core::hnsw::graph::VectorStorage::Raw { data, .. } => data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<f32>>(),
            _ => panic!("Expected stored vectors for {:?}", metric),
        };

        let query_emb = provider
            .compute_embeddings(&["Document 0 about things".to_string()])
            .unwrap();
        let query: Vec<f32> = query_emb.row(0).to_vec();

        let params = SearchParams {
            ef_search: 16,
            ..Default::default()
        };
        let (labels, distances) = search_hnsw(&graph, &query, 3, &flat_vectors, &params);
        assert_eq!(labels.len(), 3, "Expected 3 results for {:?}", metric);
    }
}

/// top_k bounds: top_k=1 returns 1, top_k > n_docs returns all docs.
#[test]
fn test_search_top_k_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), false, false).unwrap();
    let paths = IndexPaths::new(&index_path);

    let index_data = std::fs::read(paths.index_file_path()).unwrap();
    let graph = read_hnsw_index(&mut Cursor::new(index_data)).unwrap();

    let flat_vectors = match &graph.vector_storage {
        leann_core::hnsw::graph::VectorStorage::Raw { data, .. } => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<f32>>(),
        _ => panic!("Expected stored vectors"),
    };

    let provider = FakeEmbeddingProvider::new(64);
    let query_emb = provider
        .compute_embeddings(&["test query".to_string()])
        .unwrap();
    let query: Vec<f32> = query_emb.row(0).to_vec();

    let params = SearchParams {
        ef_search: 32,
        ..Default::default()
    };

    // top_k = 1
    let (labels, _) = search_hnsw(&graph, &query, 1, &flat_vectors, &params);
    assert_eq!(labels.len(), 1);

    // top_k = 100 (> n_docs=10) — should return at most 10
    let (labels, _) = search_hnsw(&graph, &query, 100, &flat_vectors, &params);
    assert!(
        labels.len() <= 10,
        "Got {} results for 10 docs",
        labels.len()
    );
}

/// Build from pre-computed embeddings via build_index_from_embeddings.
#[test]
fn test_build_from_precomputed_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let n = 50;
    let dims = 32;

    // Create random-ish embeddings
    let mut data = Vec::with_capacity(n * dims);
    for i in 0..n {
        for d in 0..dims {
            data.push(((i * dims + d) as f32) / (n * dims) as f32);
        }
    }
    let embeddings = Array2::from_shape_vec((n, dims), data).unwrap();
    let ids: Vec<String> = (0..n).map(|i| format!("doc_{}", i)).collect();

    let mut builder = LeannBuilder::new("precomputed-model", Some(dims), "precomputed");
    builder = builder
        .with_m(8)
        .with_ef_construction(20)
        .with_compact(false)
        .with_recompute(false);

    let index_path = dir.path().join("precomputed_index");
    builder
        .build_index_from_embeddings(&index_path, &ids, &embeddings)
        .unwrap();

    // Verify files
    let paths = IndexPaths::new(&index_path);
    assert!(paths.meta_path().exists());
    assert!(paths.passages_path().exists());
    assert!(paths.index_file_path().exists());

    let meta = IndexMeta::load(&paths.meta_path()).unwrap();
    assert_eq!(meta.dimensions, dims);
    assert_eq!(meta.total_passages, Some(n));
    assert_eq!(meta.built_from_precomputed_embeddings, Some(true));
}

/// HNSW index binary roundtrip: write compact, read back, verify structure.
#[test]
fn test_hnsw_index_compact_roundtrip() {
    let provider = FakeEmbeddingProvider::new(16);
    let data = provider
        .compute_embeddings(&(0..20).map(|i| format!("doc {}", i)).collect::<Vec<_>>())
        .unwrap();

    let config = HnswConfig {
        m: 8,
        ef_construction: 20,
        distance_metric: DistanceMetric::L2,
        is_compact: false,
        is_recompute: true,
        ..Default::default()
    };

    let graph = build_hnsw(&data, &config).unwrap();
    let compact = leann_core::hnsw::csr::convert_to_csr(&graph).unwrap();

    // Write
    let mut buf = Vec::new();
    write_hnsw_compact(&mut buf, &compact).unwrap();

    // Read back
    let graph2 = read_hnsw_index(&mut Cursor::new(&buf)).unwrap();
    assert!(graph2.is_compact());
    assert_eq!(graph2.ntotal, 20);
    assert_eq!(graph2.dimensions, 16);
    assert_eq!(graph2.entry_point, graph.entry_point);
}

/// HNSW index binary roundtrip: standard format.
#[test]
fn test_hnsw_index_standard_roundtrip() {
    let provider = FakeEmbeddingProvider::new(16);
    let data = provider
        .compute_embeddings(&(0..15).map(|i| format!("doc {}", i)).collect::<Vec<_>>())
        .unwrap();

    let config = HnswConfig {
        m: 8,
        ef_construction: 20,
        distance_metric: DistanceMetric::L2,
        is_compact: false,
        is_recompute: false,
        ..Default::default()
    };

    let graph = build_hnsw(&data, &config).unwrap();

    let mut buf = Vec::new();
    write_hnsw_standard(&mut buf, &graph).unwrap();

    let graph2 = read_hnsw_index(&mut Cursor::new(&buf)).unwrap();
    assert!(!graph2.is_compact());
    assert_eq!(graph2.ntotal, 15);
}
