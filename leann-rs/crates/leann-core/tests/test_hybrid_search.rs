//! E2E-3: Hybrid Search (Vector + BM25)
//!
//! Tests BM25 keyword search, grep search, and the hybrid search path
//! via LeannSearcher. Vector search through LeannSearcher requires a ZMQ
//! embedding server, so we test that at the lower HNSW level instead.
//!
//! Mirrors Python test_hybrid_search.py.

mod common;

use common::{build_test_index, diverse_documents, FakeEmbeddingProvider};
use leann_core::index::IndexPaths;
use leann_core::searcher::{LeannSearcher, SearchConfig};
use leann_core::LeannBuilder;
use std::collections::HashMap;

/// Helper: build an index from diverse documents and return a searcher.
fn build_diverse_index() -> (tempfile::TempDir, LeannSearcher) {
    let dir = tempfile::tempdir().unwrap();
    let provider = FakeEmbeddingProvider::new(64);
    let docs = diverse_documents();

    let mut builder = LeannBuilder::new("fake-model", Some(64), "test");
    builder = builder
        .with_m(16)
        .with_ef_construction(40)
        .with_compact(true)
        .with_recompute(true)
        .with_distance_metric(leann_core::index::DistanceMetric::L2);

    for (text, meta) in &docs {
        builder.add_text(text, meta.clone());
    }

    let index_path = dir.path().join("diverse_index");
    builder.build_index(&index_path, &provider).unwrap();

    let searcher = LeannSearcher::open(&index_path).unwrap();
    (dir, searcher)
}

/// Pure BM25 search (gemma=0.0) — should match keyword queries.
/// (Python: test_pure_keyword_search)
#[test]
fn test_pure_bm25_search() {
    let (_dir, searcher) = build_diverse_index();

    let config = SearchConfig {
        gemma: 0.0,
        ..Default::default()
    };

    let results = searcher
        .search_with_params("python programming machine learning", 5, &config)
        .unwrap();

    assert!(!results.is_empty(), "BM25 should return results");

    // Top results should have positive scores (matching keywords)
    assert!(
        results[0].score > 0.0,
        "Top BM25 result should have positive score for matching query"
    );

    // Doc 1 ("Python programming...") should rank in top 3
    let top_ids: Vec<&str> = results.iter().take(3).map(|r| r.id.as_str()).collect();
    assert!(
        top_ids.contains(&"1"),
        "Doc 1 (Python/ML) should be in top 3 for 'python programming': got {:?}",
        top_ids
    );
}

/// BM25 search for cooking-related query.
#[test]
fn test_bm25_search_cooking() {
    let (_dir, searcher) = build_diverse_index();

    let config = SearchConfig {
        gemma: 0.0,
        ..Default::default()
    };

    let results = searcher
        .search_with_params("cooking pasta ingredients", 3, &config)
        .unwrap();

    assert!(!results.is_empty());
    // Doc 4 ("Cooking Italian pasta...") should rank highest
    let top_ids: Vec<&str> = results.iter().take(2).map(|r| r.id.as_str()).collect();
    assert!(
        top_ids.contains(&"4"),
        "Doc 4 (cooking) should be in top 2 for 'cooking pasta': got {:?}",
        top_ids
    );
}

/// BM25 scores should be descending.
#[test]
fn test_bm25_scores_descending() {
    let (_dir, searcher) = build_diverse_index();

    let config = SearchConfig {
        gemma: 0.0,
        ..Default::default()
    };

    let results = searcher
        .search_with_params("database query performance", 5, &config)
        .unwrap();

    for i in 1..results.len() {
        assert!(
            results[i].score <= results[i - 1].score + 1e-6,
            "BM25 scores not descending: {} > {}",
            results[i].score,
            results[i - 1].score
        );
    }
}

/// Grep search should match exact substrings case-insensitively.
#[test]
fn test_grep_search() {
    let (_dir, searcher) = build_diverse_index();

    let config = SearchConfig {
        use_grep: true,
        ..Default::default()
    };

    let results = searcher.search_with_params("pyramids", 5, &config).unwrap();

    assert!(!results.is_empty(), "Grep should find 'pyramids'");
    assert!(
        results[0].text.to_lowercase().contains("pyramids"),
        "Grep result should contain 'pyramids'"
    );
}

/// Grep search with no match returns empty results.
#[test]
fn test_grep_search_no_match() {
    let (_dir, searcher) = build_diverse_index();

    let config = SearchConfig {
        use_grep: true,
        ..Default::default()
    };

    let results = searcher
        .search_with_params("xyznonexistent123", 5, &config)
        .unwrap();
    assert!(results.is_empty(), "Grep should return empty for no match");
}

/// BM25 search for nonexistent terms returns results with zero scores.
#[test]
fn test_bm25_nonexistent_terms() {
    let (_dir, searcher) = build_diverse_index();

    let config = SearchConfig {
        gemma: 0.0,
        ..Default::default()
    };

    let results = searcher
        .search_with_params("xyzqrs9999", 3, &config)
        .unwrap();

    // All scores should be 0 (no matching terms)
    for r in &results {
        assert!(
            r.score.abs() < 1e-6,
            "Expected zero score for nonexistent terms, got {}",
            r.score
        );
    }
}

/// BM25 on a larger index (100 docs) with topic-based queries.
#[test]
fn test_bm25_with_sample_documents() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(100, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let config = SearchConfig {
        gemma: 0.0,
        ..Default::default()
    };

    let results = searcher.search_with_params("topic_0", 10, &config).unwrap();

    assert!(!results.is_empty(), "Should find docs about topic_0");

    // Top results should have positive scores (matching "topic_0")
    assert!(
        results[0].score > 0.0,
        "Top result should have positive BM25 score"
    );

    // Documents about topic_0 are 0, 5, 10, 15, ... (every 5th)
    // Their IDs should rank highly
    let top_ids: Vec<&str> = results.iter().take(5).map(|r| r.id.as_str()).collect();
    let topic_0_ids: Vec<String> = (0..100).step_by(5).map(|i| i.to_string()).collect();
    let matching = top_ids
        .iter()
        .filter(|id| topic_0_ids.contains(&id.to_string()))
        .count();
    assert!(
        matching > 0,
        "At least one topic_0 doc should be in top 5: got {:?}",
        top_ids
    );
}
