//! E2E-4: Metadata Filtering
//!
//! Tests all 13 filter operators + compound AND logic applied to real SearchResults.
//! Mirrors Python test_metadata_filtering.py.

use leann_core::metadata_filter::{FilterSpec, MetadataFilterEngine, MetadataFilters};
use leann_core::search_result::SearchResult;
use serde_json::json;
use std::collections::HashMap;

fn sample_results() -> Vec<SearchResult> {
    vec![
        SearchResult::with_metadata("doc0".into(), 0.95, "A tale of two cities".into(), {
            let mut m = HashMap::new();
            m.insert("chapter".to_string(), json!(1));
            m.insert("genre".to_string(), json!("fiction"));
            m.insert("word_count".to_string(), json!(1500));
            m.insert("is_published".to_string(), json!(true));
            m.insert("tags".to_string(), json!(["classic", "drama"]));
            m.insert("topic".to_string(), json!("topic_0"));
            m.insert("doc_num".to_string(), json!(0));
            m
        }),
        SearchResult::with_metadata(
            "doc1".into(),
            0.85,
            "Data structures and algorithms".into(),
            {
                let mut m = HashMap::new();
                m.insert("chapter".to_string(), json!(5));
                m.insert("genre".to_string(), json!("science"));
                m.insert("word_count".to_string(), json!(3000));
                m.insert("is_published".to_string(), json!(true));
                m.insert("tags".to_string(), json!(["textbook"]));
                m.insert("topic".to_string(), json!("topic_1"));
                m.insert("doc_num".to_string(), json!(5));
                m
            },
        ),
        SearchResult::with_metadata("doc2".into(), 0.75, "Pride and Prejudice".into(), {
            let mut m = HashMap::new();
            m.insert("chapter".to_string(), json!(3));
            m.insert("genre".to_string(), json!("fiction"));
            m.insert("word_count".to_string(), json!(2000));
            m.insert("is_published".to_string(), json!(false));
            m.insert("tags".to_string(), json!(["classic", "romance"]));
            m.insert("topic".to_string(), json!("topic_0"));
            m.insert("doc_num".to_string(), json!(12));
            m
        }),
        SearchResult::with_metadata(
            "doc3".into(),
            0.65,
            "Introduction to machine learning".into(),
            {
                let mut m = HashMap::new();
                m.insert("chapter".to_string(), json!(10));
                m.insert("genre".to_string(), json!("science"));
                m.insert("word_count".to_string(), json!(5000));
                m.insert("is_published".to_string(), json!(true));
                m.insert("tags".to_string(), json!(["textbook", "ai"]));
                m.insert("topic".to_string(), json!("topic_2"));
                m.insert("doc_num".to_string(), json!(25));
                m
            },
        ),
    ]
}

/// Convert SearchResults to the dict format that MetadataFilterEngine expects.
fn results_to_dicts(results: &[SearchResult]) -> Vec<HashMap<String, serde_json::Value>> {
    results
        .iter()
        .map(|r| {
            let mut map = HashMap::new();
            map.insert("id".to_string(), json!(r.id));
            map.insert("score".to_string(), json!(r.score));
            map.insert("text".to_string(), json!(r.text));
            map.insert(
                "metadata".to_string(),
                serde_json::to_value(&r.metadata).unwrap(),
            );
            map
        })
        .collect()
}

fn make_filter(field: &str, op: &str, value: serde_json::Value) -> MetadataFilters {
    let mut filters = MetadataFilters::new();
    let mut spec = FilterSpec::new();
    spec.insert(op.to_string(), value);
    filters.insert(field.to_string(), spec);
    filters
}

#[test]
fn test_filter_equals() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("genre", "==", json!("fiction"));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2);
    for r in &filtered {
        let genre = r
            .get("metadata")
            .unwrap()
            .get("genre")
            .unwrap()
            .as_str()
            .unwrap();
        assert_eq!(genre, "fiction");
    }
}

#[test]
fn test_filter_not_equals() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("genre", "!=", json!("fiction"));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2);
    for r in &filtered {
        let genre = r
            .get("metadata")
            .unwrap()
            .get("genre")
            .unwrap()
            .as_str()
            .unwrap();
        assert_ne!(genre, "fiction");
    }
}

#[test]
fn test_filter_less_than() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("chapter", "<", json!(5));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2); // chapter 1 and 3
}

#[test]
fn test_filter_less_than_or_equal() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("chapter", "<=", json!(5));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 3); // chapter 1, 3, 5
}

#[test]
fn test_filter_greater_than() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("chapter", ">", json!(5));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 1); // chapter 10
}

#[test]
fn test_filter_greater_equal() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("doc_num", ">=", json!(12));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2); // doc_num 12 and 25
}

#[test]
fn test_filter_in() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("topic", "in", json!(["topic_0", "topic_1"]));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 3); // topic_0 x2, topic_1 x1
}

#[test]
fn test_filter_not_in() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("topic", "not_in", json!(["topic_0"]));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2); // topic_1 and topic_2
}

#[test]
fn test_filter_contains() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("topic", "contains", json!("_0"));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2);
}

#[test]
fn test_filter_starts_with() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("topic", "starts_with", json!("topic_"));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 4); // all have topic_ prefix
}

#[test]
fn test_filter_ends_with() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("topic", "ends_with", json!("_0"));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2); // topic_0 only
}

#[test]
fn test_filter_is_true() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("is_published", "is_true", json!(null));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 3); // doc0, doc1, doc3 are published
}

#[test]
fn test_filter_is_false() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("is_published", "is_false", json!(null));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 1); // only doc2
}

#[test]
fn test_filter_compound_and() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());

    let mut filters = MetadataFilters::new();
    // genre == "fiction" AND chapter <= 3
    let mut genre_spec = FilterSpec::new();
    genre_spec.insert("==".to_string(), json!("fiction"));
    filters.insert("genre".to_string(), genre_spec);

    let mut chapter_spec = FilterSpec::new();
    chapter_spec.insert("<=".to_string(), json!(3));
    filters.insert("chapter".to_string(), chapter_spec);

    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2); // doc0 (ch1, fiction) and doc2 (ch3, fiction)
}

#[test]
fn test_filter_range() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());

    let mut filters = MetadataFilters::new();
    let mut spec = FilterSpec::new();
    spec.insert(">=".to_string(), json!(2000));
    spec.insert("<".to_string(), json!(4000));
    filters.insert("word_count".to_string(), spec);

    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 2); // 2000 and 3000
}

#[test]
fn test_filter_no_matches() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("chapter", "==", json!(999));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert!(filtered.is_empty());
}

#[test]
fn test_filter_none_passthrough() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = MetadataFilters::new(); // empty
    let filtered = engine.apply_filters(&dicts, &filters);
    assert_eq!(filtered.len(), 4); // all returned
}

#[test]
fn test_filter_missing_field() {
    let engine = MetadataFilterEngine::new();
    let dicts = results_to_dicts(&sample_results());
    let filters = make_filter("nonexistent_field", "==", json!("value"));
    let filtered = engine.apply_filters(&dicts, &filters);
    assert!(filtered.is_empty(), "Missing field should exclude all");
}
