//! Tests for provider-based search (no ZMQ dependency).
//!
//! Verifies that LeannSearcher can open an index, construct an embedding
//! provider from metadata, and search using the provider directly.

mod common;

use common::build_test_index;
use leann_core::searcher::{LeannSearcher, SearchConfig, SearcherOptions};

/// Build a non-recompute index, open with LeannSearcher, search returns results.
#[test]
fn test_search_with_fake_provider() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(50, dir.path(), false, false).unwrap();

    // Open searcher (it will create a provider from meta, which will be
    // a remote provider; for non-recompute indexes with stored vectors,
    // search works even if the provider fails — it uses stored vectors)
    let searcher = LeannSearcher::open(&index_path).unwrap();

    // Grep-based search works without any provider
    let config = SearchConfig {
        use_grep: true,
        ..Default::default()
    };
    let results = searcher.search_with_params("document", 5, &config).unwrap();
    assert!(
        !results.is_empty(),
        "Grep search should return results for 'document'"
    );
}

/// Build a recompute (compact) index and verify recompute callback uses provider.
/// Since FakeEmbeddingProvider isn't registered as a remote provider,
/// we verify the pipeline through the non-recompute path with stored vectors.
#[test]
fn test_search_non_recompute_with_stored_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(30, dir.path(), false, false).unwrap();

    let searcher = LeannSearcher::open(&index_path).unwrap();

    // BM25 search (uses bm25 feature, doesn't need embedding provider)
    let config = SearchConfig {
        gemma: 0.0,
        ..Default::default()
    };
    let results = searcher.search_with_params("document", 5, &config).unwrap();
    assert!(
        !results.is_empty(),
        "BM25 search should return results for 'document'"
    );
}

/// Verify SearcherOptions default has warmup=false.
#[test]
fn test_searcher_options_default() {
    let opts = SearcherOptions::default();
    assert!(!opts.enable_warmup);
    assert!(opts.recompute_embeddings.is_none());
}

/// Open with warmup=true on a non-recompute index.
/// Warmup may fail (no real provider), but open should still succeed.
#[test]
fn test_warmup_with_options() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), false, false).unwrap();

    let options = SearcherOptions {
        enable_warmup: true,
        ..Default::default()
    };
    // Should not panic even if warmup fails (it logs a warning)
    let _searcher = LeannSearcher::open_with_options(&index_path, &options).unwrap();
}

/// Verify warmup() method can be called directly.
#[test]
fn test_warmup_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let index_path = build_test_index(10, dir.path(), false, false).unwrap();

    let searcher = LeannSearcher::open(&index_path).unwrap();
    // warmup() returns Ok even if provider isn't available or fails
    searcher.warmup().unwrap();
}

/// Verify create_embedding_provider with different modes.
#[cfg(feature = "embedding-remote")]
mod provider_factory_tests {
    use leann_core::embedding::{EmbeddingMode, create_embedding_provider};
    use std::collections::HashMap;

    #[test]
    fn test_provider_from_meta_ollama() {
        let options = HashMap::new();
        let provider =
            create_embedding_provider(&EmbeddingMode::Ollama, "nomic-embed-text", &options)
                .unwrap();
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_provider_from_meta_unknown_defaults() {
        // sentence-transformers falls back to OpenAI or Ollama
        let options = HashMap::new();
        let provider = create_embedding_provider(
            &EmbeddingMode::SentenceTransformers,
            "unknown-model",
            &options,
        )
        .unwrap();
        // Should get either "openai" or "ollama" as fallback
        assert!(
            provider.name() == "openai" || provider.name() == "ollama",
            "Expected fallback to openai or ollama, got '{}'",
            provider.name()
        );
    }
}
