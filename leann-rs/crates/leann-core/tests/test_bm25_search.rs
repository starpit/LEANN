//! BM25 keyword search end-to-end tests.
//!
//! Tests BM25Scorer independently from the searcher, with larger corpora.

use leann_core::bm25::BM25Scorer;

fn large_corpus() -> Vec<(String, String)> {
    vec![
        ("0".into(), "Python is a versatile programming language used for web development, data science, and machine learning".into()),
        ("1".into(), "JavaScript runs in web browsers and is essential for front-end web development and user interfaces".into()),
        ("2".into(), "Rust provides memory safety without garbage collection, making it ideal for systems programming".into()),
        ("3".into(), "Machine learning algorithms can identify patterns in large datasets and make predictions".into()),
        ("4".into(), "Database systems store and retrieve data efficiently using indexing and query optimization".into()),
        ("5".into(), "Cloud computing offers scalable infrastructure and services like storage and computation".into()),
        ("6".into(), "Neural networks are a subset of machine learning inspired by the human brain".into()),
        ("7".into(), "The weather forecast uses atmospheric models to predict temperature and rainfall".into()),
        ("8".into(), "Cooking Italian cuisine requires fresh ingredients like olive oil, tomatoes, and basil".into()),
        ("9".into(), "Ancient Egyptian pyramids were engineering marvels built over four thousand years ago".into()),
        ("10".into(), "Vector databases enable similarity search across high-dimensional embedding spaces".into()),
        ("11".into(), "Deep learning has revolutionized computer vision, natural language processing, and speech recognition".into()),
        ("12".into(), "Graph databases model relationships between entities using nodes and edges".into()),
        ("13".into(), "Functional programming emphasizes immutability and pure functions without side effects".into()),
        ("14".into(), "The human genome contains approximately three billion base pairs of DNA".into()),
    ]
}

#[test]
fn test_bm25_programming_query() {
    let mut scorer = BM25Scorer::default();
    scorer.fit(&large_corpus());

    let results = scorer.search("programming language", 5);
    assert_eq!(results.len(), 5);

    // Top results should be about programming
    let top_ids: Vec<&str> = results.iter().take(3).map(|r| r.id.as_str()).collect();
    assert!(
        top_ids.contains(&"0") || top_ids.contains(&"2") || top_ids.contains(&"13"),
        "Top results for 'programming language' should include Python/Rust/Functional: got {:?}",
        top_ids
    );
}

#[test]
fn test_bm25_machine_learning_query() {
    let mut scorer = BM25Scorer::default();
    scorer.fit(&large_corpus());

    let results = scorer.search("machine learning neural networks", 5);
    assert_eq!(results.len(), 5);

    let top_ids: Vec<&str> = results.iter().take(3).map(|r| r.id.as_str()).collect();
    // Docs 3, 6, 11 are about ML/neural networks
    let has_ml = top_ids.contains(&"3") || top_ids.contains(&"6") || top_ids.contains(&"11");
    assert!(
        has_ml,
        "Top results for 'machine learning neural networks' should include ML docs: got {:?}",
        top_ids
    );
}

#[test]
fn test_bm25_database_query() {
    let mut scorer = BM25Scorer::default();
    scorer.fit(&large_corpus());

    let results = scorer.search("database indexing query", 3);
    assert!(!results.is_empty());

    // Doc 4 or 12 should rank highly
    let top_id = &results[0].id;
    assert!(
        top_id == "4" || top_id == "12" || top_id == "10",
        "Top result for 'database' should be doc 4, 10, or 12: got {}",
        top_id
    );
}

#[test]
fn test_bm25_scores_are_positive_for_matches() {
    let mut scorer = BM25Scorer::default();
    scorer.fit(&large_corpus());

    let results = scorer.search("Python programming", 5);
    // At least the top result should have positive score
    assert!(
        results[0].score > 0.0,
        "Top BM25 score should be positive for matching query"
    );
}

#[test]
fn test_bm25_no_match_zero_scores() {
    let mut scorer = BM25Scorer::default();
    scorer.fit(&large_corpus());

    let results = scorer.search("xyznonexistentterm123", 3);
    for r in &results {
        assert!(
            r.score.abs() < 1e-10,
            "Score should be ~0 for nonexistent term, got {}",
            r.score
        );
    }
}

#[test]
fn test_bm25_scores_descending() {
    let mut scorer = BM25Scorer::default();
    scorer.fit(&large_corpus());

    let results = scorer.search("web development JavaScript", 10);
    for i in 1..results.len() {
        assert!(
            results[i].score <= results[i - 1].score + 1e-10,
            "BM25 scores not descending at pos {}: {} > {}",
            i,
            results[i].score,
            results[i - 1].score
        );
    }
}

#[test]
fn test_bm25_top_k_larger_than_corpus() {
    let mut scorer = BM25Scorer::default();
    scorer.fit(&large_corpus());

    let results = scorer.search("Python", 100);
    assert_eq!(
        results.len(),
        15,
        "Should return all 15 docs when top_k > corpus size"
    );
}

#[test]
fn test_bm25_custom_k1_b() {
    let mut scorer = BM25Scorer::new(2.0, 0.5); // higher k1, lower b
    scorer.fit(&large_corpus());

    let results = scorer.search("machine learning", 3);
    assert!(!results.is_empty());
    // Should still find ML-related docs
    assert!(results[0].score > 0.0);
}
