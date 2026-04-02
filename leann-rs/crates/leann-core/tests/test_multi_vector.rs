//! E2E tests for multi-vector (ColBERT) index support.
//!
//! Tests the full pipeline: MultiVectorBuilder → build → files on disk → open → search.
//! Simulates the ColBERT workflow with synthetic token embeddings.

#![cfg(feature = "multi-vector")]

use leann_core::multi_vector::{MultiVectorBuilder, MultiVectorSearcher};
use ndarray::{Array2, array};
use std::collections::HashMap;

/// Simulate a ColBERT-style corpus: each "document" is a set of token embeddings.
/// Returns (docs, queries) where docs[i] = (doc_id, embeddings, metadata).
fn colbert_corpus(
    n_docs: usize,
    tokens_per_doc: usize,
    dim: usize,
) -> Vec<(u32, Array2<f32>, HashMap<String, serde_json::Value>)> {
    // Use a simple seeded PRNG for reproducibility.
    let mut rng_state: u64 = 42;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    let mut docs = Vec::new();
    for i in 0..n_docs {
        let mut emb = Array2::<f32>::zeros((tokens_per_doc, dim));
        for t in 0..tokens_per_doc {
            // Strong signal in doc-specific dimensions.
            let base_dim = (i * 3) % dim;
            emb[[t, base_dim]] = 3.0;
            emb[[t, (base_dim + 1) % dim]] = 2.0;
            // Small random noise in all dims.
            for d in 0..dim {
                emb[[t, d]] += next_f32() * 0.1;
            }
        }
        let mut meta = HashMap::new();
        meta.insert(
            "filepath".to_string(),
            serde_json::json!(format!("page_{}.png", i)),
        );
        meta.insert("doc_num".to_string(), serde_json::json!(i));
        docs.push((i as u32, emb, meta));
    }
    docs
}

// ── Build & Open ────────────────────────────────────────────────────────

#[test]
fn test_e2e_build_open_search() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("colbert_e2e");

    let docs = colbert_corpus(20, 8, 32);

    let mut builder = MultiVectorBuilder::new(32);
    for (doc_id, emb, meta) in &docs {
        builder.insert(*doc_id, emb.clone(), meta.clone());
    }
    builder.build(&index_path).unwrap();

    let searcher = MultiVectorSearcher::open(&index_path).unwrap();
    assert_eq!(searcher.num_docs(), 20);
    assert_eq!(searcher.num_tokens(), 20 * 8);

    // Query matching doc 5's pattern.
    let (_, doc5_emb, _) = &docs[5];
    let query = doc5_emb.slice(ndarray::s![0..2, ..]).to_owned();
    let results = searcher.search(&query, 5).unwrap();

    assert!(!results.is_empty());
    // Doc 5 should be in top results (exact search is more reliable).
    let exact = searcher.search_exact(&query, 5, 20).unwrap();
    assert_eq!(exact[0].doc_id, 5);
}

#[test]
fn test_e2e_exact_search_matches_doc() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("colbert_exact");

    let docs = colbert_corpus(10, 4, 16);

    let mut builder = MultiVectorBuilder::new(16);
    for (doc_id, emb, meta) in &docs {
        builder.insert(*doc_id, emb.clone(), meta.clone());
    }
    builder.build(&index_path).unwrap();

    let searcher = MultiVectorSearcher::open(&index_path).unwrap();

    // Use doc 3's own tokens as query — should rank doc 3 first.
    let (_, doc3_emb, _) = &docs[3];
    let results = searcher.search_exact(doc3_emb, 3, 40).unwrap();

    assert_eq!(results[0].doc_id, 3);
}

// ── Metadata round-trip ─────────────────────────────────────────────────

#[test]
fn test_e2e_metadata_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("colbert_meta");

    let mut builder = MultiVectorBuilder::new(4);
    for i in 0..3u32 {
        let emb = array![[1.0, 0.0, 0.0, 0.0]];
        let mut meta = HashMap::new();
        meta.insert("page".to_string(), serde_json::json!(i));
        meta.insert(
            "source".to_string(),
            serde_json::json!(format!("doc_{}.pdf", i)),
        );
        builder.insert(i, emb, meta);
    }
    builder.build(&index_path).unwrap();

    let searcher = MultiVectorSearcher::open(&index_path).unwrap();
    let query = array![[1.0, 0.0, 0.0, 0.0]];
    let results = searcher.search(&query, 3).unwrap();

    // All results should have metadata.
    for r in &results {
        assert!(r.metadata.contains_key("page"));
        assert!(r.metadata.contains_key("source"));
        let page = r.metadata["page"].as_u64().unwrap() as u32;
        assert_eq!(page, r.doc_id);
    }
}

// ── Scale test ──────────────────────────────────────────────────────────

#[test]
fn test_e2e_100_docs_128dim() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("colbert_scale");

    let dim = 128;
    let n_docs = 100;
    let tokens_per_doc = 16;

    let docs = colbert_corpus(n_docs, tokens_per_doc, dim);

    let mut builder = MultiVectorBuilder::new(dim);
    builder.set_m(16).set_ef_construction(100);
    for (doc_id, emb, meta) in &docs {
        builder.insert(*doc_id, emb.clone(), meta.clone());
    }
    builder.build(&index_path).unwrap();

    let searcher = MultiVectorSearcher::open(&index_path).unwrap();
    assert_eq!(searcher.num_docs(), n_docs);
    assert_eq!(searcher.num_tokens(), n_docs * tokens_per_doc);

    // Query with doc 42's tokens.
    let (_, doc42_emb, _) = &docs[42];
    let query = doc42_emb.slice(ndarray::s![0..4, ..]).to_owned();

    let exact = searcher.search_exact(&query, 10, 100).unwrap();
    assert_eq!(exact[0].doc_id, 42);

    // Approximate search should also find doc 42 in results.
    let approx = searcher.search(&query, 10).unwrap();
    assert!(approx.iter().any(|r| r.doc_id == 42));
}

// ── Sidecar file integrity ──────────────────────────────────────────────

#[test]
fn test_e2e_sidecar_files_valid() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("colbert_sidecar");

    let mut builder = MultiVectorBuilder::new(8);
    builder.insert(0, Array2::ones((5, 8)), HashMap::new());
    builder.insert(1, Array2::ones((3, 8)), HashMap::new());
    builder.build(&index_path).unwrap();

    // Verify all sidecar files exist and have valid content.
    let index_file = dir.path().join("colbert_sidecar.index");
    let labels_file = dir.path().join("colbert_sidecar.labels.json");
    let emb_file = dir.path().join("colbert_sidecar.emb.npy");

    assert!(index_file.exists());
    assert!(labels_file.exists());
    assert!(emb_file.exists());

    // Labels should have 8 entries (5 + 3 tokens).
    let labels: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&labels_file).unwrap()).unwrap();
    assert_eq!(labels.len(), 8);

    // NPY file should start with numpy magic.
    let npy_data = std::fs::read(&emb_file).unwrap();
    assert_eq!(&npy_data[0..6], b"\x93NUMPY");
    // Data section: 8 vectors × 8 dims × 4 bytes = 256 bytes (plus header).
    assert!(npy_data.len() > 256);
}

// ── Reopen persistence ──────────────────────────────────────────────────

#[test]
fn test_e2e_reopen_gives_same_results() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = dir.path().join("colbert_reopen");

    // Use distinct scores to avoid ties.
    let mut builder = MultiVectorBuilder::new(4);
    builder.insert(0, array![[1.0, 0.0, 0.0, 0.0]], HashMap::new());
    builder.insert(1, array![[0.0, 0.7, 0.3, 0.0]], HashMap::new());
    builder.insert(2, array![[0.0, 0.0, 1.0, 0.0]], HashMap::new());
    builder.build(&index_path).unwrap();

    let query = array![[0.0, 0.0, 1.0, 0.0]];

    // Search with first open.
    let searcher1 = MultiVectorSearcher::open(&index_path).unwrap();
    let results1 = searcher1.search_exact(&query, 3, 10).unwrap();

    // Drop and reopen.
    drop(searcher1);
    let searcher2 = MultiVectorSearcher::open(&index_path).unwrap();
    let results2 = searcher2.search_exact(&query, 3, 10).unwrap();

    assert_eq!(results1.len(), results2.len());
    // Scores should be identical across reopens.
    let mut scores1: Vec<f32> = results1.iter().map(|r| r.score).collect();
    let mut scores2: Vec<f32> = results2.iter().map(|r| r.score).collect();
    scores1.sort_by(|a, b| b.partial_cmp(a).unwrap());
    scores2.sort_by(|a, b| b.partial_cmp(a).unwrap());
    for (s1, s2) in scores1.iter().zip(scores2.iter()) {
        assert!((s1 - s2).abs() < 1e-6);
    }
    // Top result (unique score) should be the same.
    assert_eq!(results1[0].doc_id, results2[0].doc_id);
}
