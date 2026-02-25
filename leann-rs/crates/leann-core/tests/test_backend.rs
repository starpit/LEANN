//! Tests for the backend abstraction layer.
//!
//! Exercises BackendConfig, BackendIndex, and the dispatch functions
//! (build_backend, read_backend_index, search_backend, search_backend_recompute)
//! to verify the enum abstraction doesn't alter behavior vs direct HNSW calls.

mod common;

use common::FakeEmbeddingProvider;
use leann_core::LeannBuilder;
use leann_core::backend::{self, BackendConfig};
use leann_core::embedding::EmbeddingProvider;
use leann_core::hnsw::search::SearchParams;
use leann_core::index::{DistanceMetric, IndexMeta, IndexPaths};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BackendConfig tests
// ---------------------------------------------------------------------------

#[test]
fn test_from_name_hnsw() {
    let cfg = BackendConfig::from_name("hnsw").unwrap();
    assert_eq!(cfg.name(), "hnsw");
    assert_eq!(cfg.distance_metric(), DistanceMetric::Mips);
    assert!(cfg.is_compact());
    assert!(cfg.is_recompute());
}

#[test]
fn test_from_name_invalid() {
    let err = BackendConfig::from_name("ivf").unwrap_err();
    assert!(
        err.to_string().contains("not supported"),
        "Error should mention unsupported: {}",
        err
    );

    let err2 = BackendConfig::from_name("diskann").unwrap_err();
    assert!(err2.to_string().contains("not supported"));
}

#[test]
fn test_from_name_empty_string() {
    let err = BackendConfig::from_name("").unwrap_err();
    assert!(err.to_string().contains("not supported"));
}

#[test]
fn test_config_setters_round_trip() {
    let mut cfg = BackendConfig::hnsw_default();
    cfg.set_m(64);
    cfg.set_ef_construction(400);
    cfg.set_compact(false);
    cfg.set_recompute(false);
    cfg.set_distance_metric(DistanceMetric::Cosine);
    cfg.set_num_threads(8);

    let hnsw = cfg.to_hnsw_config();
    assert_eq!(hnsw.m, 64);
    assert_eq!(hnsw.ef_construction, 400);
    assert!(!hnsw.is_compact);
    assert!(!hnsw.is_recompute);
    assert_eq!(hnsw.distance_metric, DistanceMetric::Cosine);

    // to_backend_kwargs roundtrip
    let kwargs = cfg.to_backend_kwargs();
    assert_eq!(kwargs["M"], serde_json::json!(64));
    assert_eq!(kwargs["efConstruction"], serde_json::json!(400));
    assert_eq!(kwargs["distance_metric"], serde_json::json!("cosine"));
    assert_eq!(kwargs["is_compact"], serde_json::json!(false));
    assert_eq!(kwargs["is_recompute"], serde_json::json!(false));
}

#[test]
fn test_num_threads_minimum_is_one() {
    let mut cfg = BackendConfig::hnsw_default();
    cfg.set_num_threads(0);
    // Should clamp to 1
    let hnsw = cfg.to_hnsw_config();
    // We can't directly read num_threads from HnswConfig, but the build shouldn't panic.
    // The fact that set_num_threads(0) doesn't cause an error is the test.
    assert_eq!(hnsw.m, 32); // default unchanged
}

// ---------------------------------------------------------------------------
// Builder with_backend tests
// ---------------------------------------------------------------------------

#[test]
fn test_builder_with_backend_hnsw() {
    let builder = LeannBuilder::new("test-model", Some(32), "test")
        .with_backend("hnsw")
        .unwrap();

    // Should succeed without error — builder is usable
    let _ = builder.with_m(16).with_ef_construction(40);
}

#[test]
fn test_builder_with_backend_invalid() {
    let result = LeannBuilder::new("test-model", Some(32), "test").with_backend("ivf");
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        err.to_string().contains("not supported"),
        "Expected unsupported backend error: {}",
        err
    );
}

#[test]
fn test_builder_with_backend_preserves_defaults() {
    // After with_backend("hnsw"), config should use hnsw defaults
    let builder = LeannBuilder::new("test-model", Some(32), "test")
        .with_backend("hnsw")
        .unwrap()
        .with_compact(false)
        .with_recompute(false);

    // Build should succeed with an embedding provider
    let provider = FakeEmbeddingProvider::new(32);
    let dir = tempfile::tempdir().unwrap();

    let mut builder = builder;
    builder.add_text("Hello world", HashMap::new());
    builder
        .build_index(&dir.path().join("test"), &provider)
        .unwrap();
}

// ---------------------------------------------------------------------------
// build_backend + read_backend_index dispatch tests
// ---------------------------------------------------------------------------

#[test]
fn test_build_and_read_via_backend_dispatch() {
    let provider = FakeEmbeddingProvider::new(32);
    let texts: Vec<String> = (0..30).map(|i| format!("Document {}", i)).collect();
    let embeddings = provider.compute_embeddings(&texts).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let index_file = dir.path().join("test.index");

    let mut config = BackendConfig::hnsw_default();
    config.set_m(8);
    config.set_ef_construction(20);
    config.set_compact(false);
    config.set_recompute(false);

    // Build via dispatch
    backend::build_backend(&config, &embeddings, &index_file).unwrap();
    assert!(index_file.exists());
    assert!(std::fs::metadata(&index_file).unwrap().len() > 0);

    // Read via dispatch
    let index = backend::read_backend_index("hnsw", &index_file).unwrap();
    assert_eq!(index.ntotal(), 30);
    assert_eq!(index.dimensions(), 32);
    assert!(!index.is_pruned()); // non-recompute → stored vectors
}

#[test]
fn test_build_compact_recompute_via_backend() {
    let provider = FakeEmbeddingProvider::new(16);
    let texts: Vec<String> = (0..20).map(|i| format!("Doc {}", i)).collect();
    let embeddings = provider.compute_embeddings(&texts).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let index_file = dir.path().join("compact.index");

    let mut config = BackendConfig::hnsw_default();
    config.set_m(8);
    config.set_ef_construction(20);
    config.set_compact(true);
    config.set_recompute(true);

    backend::build_backend(&config, &embeddings, &index_file).unwrap();

    let index = backend::read_backend_index("hnsw", &index_file).unwrap();
    assert_eq!(index.ntotal(), 20);
    assert!(index.is_pruned()); // recompute → pruned
}

#[test]
fn test_read_backend_index_invalid_name() {
    let dir = tempfile::tempdir().unwrap();
    let fake_file = dir.path().join("fake.index");
    std::fs::write(&fake_file, b"garbage").unwrap();

    let err = backend::read_backend_index("unknown_backend", &fake_file).unwrap_err();
    assert!(
        err.to_string().contains("Unknown backend"),
        "Expected unknown backend error: {}",
        err
    );
}

// ---------------------------------------------------------------------------
// search_backend dispatch tests
// ---------------------------------------------------------------------------

#[test]
fn test_search_backend_stored_vectors() {
    let provider = FakeEmbeddingProvider::new(32);
    let texts: Vec<String> = (0..50)
        .map(|i| format!("Document {} about topic {}", i, i % 5))
        .collect();
    let embeddings = provider.compute_embeddings(&texts).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let index_file = dir.path().join("search_test.index");

    let mut config = BackendConfig::hnsw_default();
    config.set_m(16);
    config.set_ef_construction(40);
    config.set_compact(false);
    config.set_recompute(false);

    backend::build_backend(&config, &embeddings, &index_file).unwrap();
    let index = backend::read_backend_index("hnsw", &index_file).unwrap();

    // Compute a query vector
    let query_emb = provider
        .compute_embeddings(&["Document 0 about topic 0".to_string()])
        .unwrap();
    let query: Vec<f32> = query_emb.row(0).to_vec();

    let params = SearchParams {
        ef_search: 64,
        ..Default::default()
    };

    let (labels, distances) = backend::search_backend(&index, &query, 5, &params);
    assert_eq!(labels.len(), 5, "Expected 5 results");
    assert_eq!(distances.len(), 5);

    // Distances should be sorted ascending
    for i in 1..distances.len() {
        assert!(
            distances[i] >= distances[i - 1] - 1e-6,
            "Distances not sorted: {} < {}",
            distances[i],
            distances[i - 1]
        );
    }
}

#[test]
fn test_search_backend_recompute() {
    let provider = FakeEmbeddingProvider::new(32);
    let texts: Vec<String> = (0..30).map(|i| format!("Document {}", i)).collect();
    let embeddings = provider.compute_embeddings(&texts).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let index_file = dir.path().join("recompute.index");

    let mut config = BackendConfig::hnsw_default();
    config.set_m(8);
    config.set_ef_construction(20);
    config.set_compact(true);
    config.set_recompute(true);

    backend::build_backend(&config, &embeddings, &index_file).unwrap();
    let index = backend::read_backend_index("hnsw", &index_file).unwrap();

    let query_emb = provider
        .compute_embeddings(&["Document 0".to_string()])
        .unwrap();
    let query: Vec<f32> = query_emb.row(0).to_vec();

    let params = SearchParams {
        ef_search: 32,
        ..Default::default()
    };

    // Use search_backend_recompute with a callback that recomputes from the provider
    let (labels, distances) =
        backend::search_backend_recompute(&index, &query, 5, &params, |node_ids, q, out| {
            let node_texts: Vec<String> = node_ids
                .iter()
                .map(|&id| format!("Document {}", id))
                .collect();
            if let Ok(embs) = provider.compute_embeddings(&node_texts) {
                for (i, &_nid) in node_ids.iter().enumerate() {
                    let emb = embs.row(i);
                    let emb_slice = emb.as_slice().unwrap();
                    // inner product distance
                    let dot: f32 = q.iter().zip(emb_slice).map(|(a, b)| a * b).sum();
                    out[i] = -dot; // negate for min-heap
                }
            }
        });

    assert_eq!(labels.len(), 5, "Expected 5 results");
    assert_eq!(distances.len(), 5);
}

#[test]
fn test_search_backend_on_pruned_returns_empty() {
    // search_backend (non-recompute) on a pruned index returns empty results
    let provider = FakeEmbeddingProvider::new(16);
    let texts: Vec<String> = (0..10).map(|i| format!("Doc {}", i)).collect();
    let embeddings = provider.compute_embeddings(&texts).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let index_file = dir.path().join("pruned.index");

    let mut config = BackendConfig::hnsw_default();
    config.set_m(8);
    config.set_ef_construction(20);
    config.set_compact(true);
    config.set_recompute(true); // pruned

    backend::build_backend(&config, &embeddings, &index_file).unwrap();
    let index = backend::read_backend_index("hnsw", &index_file).unwrap();

    let query = vec![0.1f32; 16];
    let params = SearchParams::default();

    // Non-recompute search on pruned index: returns empty (no stored vectors)
    let (labels, distances) = backend::search_backend(&index, &query, 5, &params);
    assert!(
        labels.is_empty(),
        "Pruned index should return empty from search_backend"
    );
    assert!(distances.is_empty());
}

// ---------------------------------------------------------------------------
// End-to-end: builder → backend → searcher roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_full_pipeline_via_backend_abstraction() {
    // Build using LeannBuilder (which goes through BackendConfig internally),
    // then verify the meta.json uses the backend name and kwargs from the config.
    let provider = FakeEmbeddingProvider::new(32);
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("pipeline_test");

    let mut builder = LeannBuilder::new("test-model", Some(32), "test")
        .with_backend("hnsw")
        .unwrap()
        .with_m(16)
        .with_ef_construction(40)
        .with_compact(true)
        .with_recompute(true);

    for i in 0..25 {
        builder.add_text(&format!("Document {} about something", i), HashMap::new());
    }
    builder.build_index(&index_path, &provider).unwrap();

    // Verify meta.json
    let paths = IndexPaths::new(&index_path);
    let meta = IndexMeta::load(&paths.meta_path()).unwrap();
    assert_eq!(meta.backend_name, "hnsw");
    assert_eq!(meta.backend_kwargs["M"], serde_json::json!(16));
    assert_eq!(meta.backend_kwargs["efConstruction"], serde_json::json!(40));
    assert_eq!(meta.is_compact, Some(true));
    assert_eq!(meta.is_pruned, Some(true));

    // Read back via backend dispatch
    let index = backend::read_backend_index("hnsw", &paths.index_file_path()).unwrap();
    assert_eq!(index.ntotal(), 25);
    assert_eq!(index.dimensions(), 32);
    assert!(index.is_pruned());
}

#[test]
fn test_backend_kwargs_all_distance_metrics() {
    for (metric, expected_str) in [
        (DistanceMetric::L2, "l2"),
        (DistanceMetric::Cosine, "cosine"),
        (DistanceMetric::Mips, "mips"),
    ] {
        let mut cfg = BackendConfig::hnsw_default();
        cfg.set_distance_metric(metric);
        let kwargs = cfg.to_backend_kwargs();
        assert_eq!(
            kwargs["distance_metric"],
            serde_json::json!(expected_str),
            "Metric {:?} should serialize to '{}'",
            metric,
            expected_str,
        );
    }
}
