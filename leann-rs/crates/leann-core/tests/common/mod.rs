use anyhow::Result;
use leann_core::embedding::EmbeddingProvider;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Deterministic embedding provider for tests — no network, no model loading.
/// Maps each text to a unique vector using a simple hash-based approach.
pub struct FakeEmbeddingProvider {
    pub dims: usize,
}

impl FakeEmbeddingProvider {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    /// Create a deterministic vector for a given text.
    /// Texts with similar content produce somewhat similar vectors.
    fn text_to_vector(&self, text: &str) -> Vec<f32> {
        let bytes = text.as_bytes();
        let mut vec = vec![0.0f32; self.dims];
        for (i, &b) in bytes.iter().enumerate() {
            vec[i % self.dims] += b as f32 / 255.0;
        }
        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}

impl EmbeddingProvider for FakeEmbeddingProvider {
    fn compute_embeddings(&self, chunks: &[String]) -> Result<Array2<f32>> {
        let mut data = Vec::with_capacity(chunks.len() * self.dims);
        for chunk in chunks {
            data.extend(self.text_to_vector(chunk));
        }
        Ok(Array2::from_shape_vec((chunks.len(), self.dims), data)?)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn name(&self) -> &str {
        "fake-test-provider"
    }
}

/// Generate N synthetic documents with topics.
/// "This is document {i} about topic {i % 5}"
pub fn sample_documents(n: usize) -> Vec<(String, HashMap<String, serde_json::Value>)> {
    (0..n)
        .map(|i| {
            let topic = format!("topic_{}", i % 5);
            let text = format!("This is document {} about {}", i, topic);
            let mut meta = HashMap::new();
            meta.insert("id".to_string(), serde_json::json!(i.to_string()));
            meta.insert("doc_num".to_string(), serde_json::json!(i));
            meta.insert("topic".to_string(), serde_json::json!(topic));
            (text, meta)
        })
        .collect()
}

/// 10 diverse documents for hybrid search tests (matching Python test_hybrid_search.py).
pub fn diverse_documents() -> Vec<(String, HashMap<String, serde_json::Value>)> {
    let texts = [
        "The quick brown fox jumps over the lazy dog in the sunny meadow",
        "Python programming language is widely used for machine learning and data science",
        "The weather forecast predicts heavy rainfall and thunderstorms tomorrow",
        "Database indexing improves query performance significantly in large datasets",
        "Cooking Italian pasta requires fresh ingredients and proper timing",
        "The stock market experienced a significant downturn last quarter",
        "Neural networks and deep learning have revolutionized artificial intelligence",
        "The ancient Egyptian pyramids were built over four thousand years ago",
        "Cloud computing services offer scalable infrastructure for modern applications",
        "Marine biology studies the diverse ecosystems found in the world oceans",
    ];
    texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let mut meta = HashMap::new();
            meta.insert("id".to_string(), serde_json::json!(i.to_string()));
            meta.insert("doc_num".to_string(), serde_json::json!(i));
            (text.to_string(), meta)
        })
        .collect()
}

/// Build a test index from sample docs using FakeEmbeddingProvider.
/// Returns the index name path component (the dir/name portion).
pub fn build_test_index(
    n_docs: usize,
    dir: &Path,
    compact: bool,
    recompute: bool,
) -> Result<PathBuf> {
    let provider = FakeEmbeddingProvider::new(64);
    let docs = sample_documents(n_docs);

    let mut builder = leann_core::LeannBuilder::new("fake-test-model", Some(64), "test");
    builder = builder
        .with_m(16)
        .with_ef_construction(40)
        .with_compact(compact)
        .with_recompute(recompute)
        .with_distance_metric(leann_core::index::DistanceMetric::L2);

    for (text, meta) in &docs {
        builder.add_text(text, meta.clone());
    }

    let index_path = dir.join("test_index");
    builder.build_index(&index_path, &provider)?;
    Ok(index_path)
}
