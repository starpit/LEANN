//! E2E-4: Metadata Filtering Integration Tests
//!
//! Tests all 13 metadata filter operators at the integration level by building
//! real indexes with diverse metadata and searching with BM25 (gemma=0.0) to
//! avoid ZMQ dependency. Complements the 8 unit tests in metadata_filter.rs.

mod common;

use common::build_test_index;
use leann_core::searcher::{LeannSearcher, SearchConfig};
use std::collections::HashMap;

/// Helper to create filters in the expected format.
fn filter(field: &str, op: &str, value: serde_json::Value) -> HashMap<String, HashMap<String, serde_json::Value>> {
    let mut filters = HashMap::new();
    let mut spec = HashMap::new();
    spec.insert(op.to_string(), value);
    filters.insert(field.to_string(), spec);
    filters
}

/// Helper to do a BM25 search with metadata filters.
fn bm25_search_filtered(
    searcher: &LeannSearcher,
    query: &str,
    filters: HashMap<String, HashMap<String, serde_json::Value>>,
    top_k: usize,
) -> Vec<leann_core::SearchResult> {
    let config = SearchConfig {
        gemma: 0.0,
        metadata_filters: Some(filters),
        ..Default::default()
    };
    searcher.search_with_params(query, top_k, &config).unwrap()
}

/// Helper to do an unfiltered BM25 search.
fn bm25_search(
    searcher: &LeannSearcher,
    query: &str,
    top_k: usize,
) -> Vec<leann_core::SearchResult> {
    let config = SearchConfig {
        gemma: 0.0,
        ..Default::default()
    };
    searcher.search_with_params(query, top_k, &config).unwrap()
}

// --- Equality Operators ---

/// == filter: only matching docs returned.
#[test]
fn test_filter_equals() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("topic", "==", serde_json::json!("topic_0")),
        50,
    );

    for r in &results {
        let topic = r.metadata.get("topic").and_then(|v| v.as_str()).unwrap();
        assert_eq!(topic, "topic_0", "All results should have topic_0");
    }
    assert!(!results.is_empty(), "Should have at least one result");
}

/// != filter: matching topic excluded.
#[test]
fn test_filter_not_equals() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("topic", "!=", serde_json::json!("topic_0")),
        50,
    );

    for r in &results {
        let topic = r.metadata.get("topic").and_then(|v| v.as_str()).unwrap();
        assert_ne!(topic, "topic_0", "topic_0 should be excluded");
    }
}

// --- Comparison Operators ---

/// < filter: only docs with doc_num < threshold.
#[test]
fn test_filter_less_than() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("doc_num", "<", serde_json::json!(10)),
        50,
    );

    for r in &results {
        let num = r.metadata.get("doc_num").and_then(|v| v.as_i64()).unwrap();
        assert!(num < 10, "doc_num should be < 10, got {}", num);
    }
    assert!(!results.is_empty());
}

/// <= filter.
#[test]
fn test_filter_less_than_or_equal() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("doc_num", "<=", serde_json::json!(9)),
        50,
    );

    for r in &results {
        let num = r.metadata.get("doc_num").and_then(|v| v.as_i64()).unwrap();
        assert!(num <= 9, "doc_num should be <= 9, got {}", num);
    }
    assert!(!results.is_empty());
}

/// > filter.
#[test]
fn test_filter_greater_than() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("doc_num", ">", serde_json::json!(40)),
        50,
    );

    for r in &results {
        let num = r.metadata.get("doc_num").and_then(|v| v.as_i64()).unwrap();
        assert!(num > 40, "doc_num should be > 40, got {}", num);
    }
    assert!(!results.is_empty());
}

/// >= filter.
#[test]
fn test_filter_greater_equal() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("doc_num", ">=", serde_json::json!(45)),
        50,
    );

    for r in &results {
        let num = r.metadata.get("doc_num").and_then(|v| v.as_i64()).unwrap();
        assert!(num >= 45, "doc_num should be >= 45, got {}", num);
    }
    assert!(!results.is_empty());
}

// --- Membership Operators ---

/// in filter: only listed topics returned.
#[test]
fn test_filter_in() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("topic", "in", serde_json::json!(["topic_0", "topic_1"])),
        50,
    );

    for r in &results {
        let topic = r.metadata.get("topic").and_then(|v| v.as_str()).unwrap();
        assert!(
            topic == "topic_0" || topic == "topic_1",
            "Topic should be topic_0 or topic_1, got {}",
            topic
        );
    }
    assert!(!results.is_empty());
}

/// not_in filter: listed topics excluded.
#[test]
fn test_filter_not_in() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("topic", "not_in", serde_json::json!(["topic_0"])),
        50,
    );

    for r in &results {
        let topic = r.metadata.get("topic").and_then(|v| v.as_str()).unwrap();
        assert_ne!(topic, "topic_0", "topic_0 should be excluded");
    }
}

// --- String Operators ---

/// contains filter: substring match.
#[test]
fn test_filter_contains() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("topic", "contains", serde_json::json!("_0")),
        50,
    );

    for r in &results {
        let topic = r.metadata.get("topic").and_then(|v| v.as_str()).unwrap();
        assert!(
            topic.contains("_0"),
            "Topic should contain '_0', got {}",
            topic
        );
    }
    assert!(!results.is_empty());
}

/// starts_with filter.
#[test]
fn test_filter_starts_with() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("topic", "starts_with", serde_json::json!("topic_")),
        50,
    );

    for r in &results {
        let topic = r.metadata.get("topic").and_then(|v| v.as_str()).unwrap();
        assert!(
            topic.starts_with("topic_"),
            "Topic should start with 'topic_', got {}",
            topic
        );
    }
    // All docs have topic_N metadata, so all should pass
    let unfiltered = bm25_search(&searcher, "document", 50);
    assert_eq!(results.len(), unfiltered.len());
}

/// ends_with filter.
#[test]
fn test_filter_ends_with() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("topic", "ends_with", serde_json::json!("_0")),
        50,
    );

    for r in &results {
        let topic = r.metadata.get("topic").and_then(|v| v.as_str()).unwrap();
        assert!(
            topic.ends_with("_0"),
            "Topic should end with '_0', got {}",
            topic
        );
    }
    assert!(!results.is_empty());
}

// --- Compound Filters ---

/// Range query: two operators on the same field (AND logic).
#[test]
fn test_filter_range() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let mut filters: HashMap<String, HashMap<String, serde_json::Value>> = HashMap::new();
    let mut spec = HashMap::new();
    spec.insert(">=".to_string(), serde_json::json!(10));
    spec.insert("<".to_string(), serde_json::json!(20));
    filters.insert("doc_num".to_string(), spec);

    let config = SearchConfig {
        gemma: 0.0,
        metadata_filters: Some(filters),
        ..Default::default()
    };

    let results = searcher.search_with_params("document", 50, &config).unwrap();

    for r in &results {
        let num = r.metadata.get("doc_num").and_then(|v| v.as_i64()).unwrap();
        assert!(
            num >= 10 && num < 20,
            "doc_num should be in [10, 20), got {}",
            num
        );
    }
    assert!(!results.is_empty());
}

/// Multiple filters on different fields: AND logic across fields.
#[test]
fn test_filter_compound_and() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(100, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let mut filters: HashMap<String, HashMap<String, serde_json::Value>> = HashMap::new();

    // topic == "topic_0"
    let mut topic_spec = HashMap::new();
    topic_spec.insert("==".to_string(), serde_json::json!("topic_0"));
    filters.insert("topic".to_string(), topic_spec);

    // doc_num < 30
    let mut num_spec = HashMap::new();
    num_spec.insert("<".to_string(), serde_json::json!(30));
    filters.insert("doc_num".to_string(), num_spec);

    let config = SearchConfig {
        gemma: 0.0,
        metadata_filters: Some(filters),
        ..Default::default()
    };

    let results = searcher.search_with_params("document", 100, &config).unwrap();

    for r in &results {
        let topic = r.metadata.get("topic").and_then(|v| v.as_str()).unwrap();
        let num = r.metadata.get("doc_num").and_then(|v| v.as_i64()).unwrap();
        assert_eq!(topic, "topic_0", "topic should be topic_0");
        assert!(num < 30, "doc_num should be < 30, got {}", num);
    }
    // topic_0 docs with doc_num < 30: 0, 5, 10, 15, 20, 25 = 6
    assert!(!results.is_empty());
}

// --- Edge Cases ---

/// Filter that matches nothing returns empty results.
#[test]
fn test_filter_no_matches() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let results = bm25_search_filtered(
        &searcher,
        "document",
        filter("doc_num", ">", serde_json::json!(99999)),
        50,
    );

    assert!(results.is_empty(), "Should have no results");
}

/// No filters returns all results unmodified.
#[test]
fn test_filter_none_passthrough() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let config_no_filter = SearchConfig {
        gemma: 0.0,
        metadata_filters: None,
        ..Default::default()
    };
    let config_empty_filter = SearchConfig {
        gemma: 0.0,
        metadata_filters: Some(HashMap::new()),
        ..Default::default()
    };

    let results_none = searcher
        .search_with_params("document", 50, &config_no_filter)
        .unwrap();
    let results_empty = searcher
        .search_with_params("document", 50, &config_empty_filter)
        .unwrap();

    assert_eq!(
        results_none.len(),
        results_empty.len(),
        "None and empty filters should return same count"
    );
}

/// Filtered results are a subset of unfiltered results.
#[test]
fn test_filter_reduces_results() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(100, dir.path(), true, true).unwrap();
    let searcher = LeannSearcher::open(&index_path).unwrap();

    let all_results = bm25_search(&searcher, "document", 100);
    let filtered = bm25_search_filtered(
        &searcher,
        "document",
        filter("doc_num", "<", serde_json::json!(20)),
        100,
    );

    assert!(
        filtered.len() < all_results.len(),
        "Filtered ({}) should be fewer than unfiltered ({})",
        filtered.len(),
        all_results.len()
    );
}
