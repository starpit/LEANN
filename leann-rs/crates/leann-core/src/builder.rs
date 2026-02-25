use anyhow::Result;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::backend::{self, BackendConfig};
use crate::embedding::{EmbeddingMode, EmbeddingProvider};
use crate::hnsw::simd::normalize_l2_inplace;
use crate::index::{DistanceMetric, IndexMeta, IndexPaths, PassageSource};
use crate::passages::{Passage, write_id_map, write_passages};

/// Detect whether a model produces normalized embeddings (L2 norm = 1),
/// in which case cosine distance should be used instead of MIPS.
///
/// Matches the Python `LeannBuilder` auto-detection logic in `api.py`.
pub fn is_normalized_embeddings_model(embedding_model: &str, embedding_mode: &str) -> bool {
    let model = embedding_model.to_lowercase();
    let mode = embedding_mode.to_lowercase();

    // Exact (mode, model) matches
    const KNOWN_MODELS: &[(&str, &str)] = &[
        ("openai", "text-embedding-ada-002"),
        ("openai", "text-embedding-3-small"),
        ("openai", "text-embedding-3-large"),
        ("voyage", "voyage-2"),
        ("voyage", "voyage-3"),
        ("voyage", "voyage-large-2"),
        ("voyage", "voyage-multilingual-2"),
        ("voyage", "voyage-code-2"),
        ("cohere", "embed-english-v3.0"),
        ("cohere", "embed-multilingual-v3.0"),
        ("cohere", "embed-english-light-v3.0"),
        ("cohere", "embed-multilingual-light-v3.0"),
    ];

    for &(known_mode, known_model) in KNOWN_MODELS {
        if (mode == known_mode && model == known_model)
            || (mode.contains(known_mode) && model.contains(known_model))
        {
            return true;
        }
    }

    // Pattern-based detection
    // OpenAI patterns
    if (mode.contains("openai") || model.contains("openai"))
        && ["text-embedding", "ada", "3-small", "3-large"]
            .iter()
            .any(|p| model.contains(p))
    {
        return true;
    }
    // Voyage patterns (all Voyage models produce normalized embeddings)
    if mode.contains("voyage") || model.contains("voyage") {
        return true;
    }
    // Cohere embed-* models
    if (mode.contains("cohere") || model.contains("cohere")) && model.contains("embed") {
        return true;
    }

    false
}

/// Builder for creating LEANN indexes.
pub struct LeannBuilder {
    embedding_model: String,
    dimensions: Option<usize>,
    embedding_mode: String,
    backend_config: BackendConfig,
    chunks: Vec<Passage>,
    embedding_options: HashMap<String, serde_json::Value>,
    /// Whether the distance metric was auto-detected (not explicitly set by the user).
    distance_metric_auto: bool,
}

impl LeannBuilder {
    pub fn new(embedding_model: &str, dimensions: Option<usize>, embedding_mode: &str) -> Self {
        let mut backend_config = BackendConfig::hnsw_default();
        let mut distance_metric_auto = false;

        if is_normalized_embeddings_model(embedding_model, embedding_mode) {
            info!(
                "Detected normalized embeddings model '{}' (mode '{}'). \
                 Auto-setting distance_metric=cosine for optimal performance.",
                embedding_model, embedding_mode
            );
            backend_config.set_distance_metric(DistanceMetric::Cosine);
            distance_metric_auto = true;
        }

        Self {
            embedding_model: embedding_model.to_string(),
            dimensions,
            embedding_mode: embedding_mode.to_string(),
            backend_config,
            chunks: Vec::new(),
            embedding_options: HashMap::new(),
            distance_metric_auto,
        }
    }

    /// Switch to a different backend (e.g. `"hnsw"`).
    pub fn with_backend(mut self, name: &str) -> Result<Self> {
        self.backend_config = BackendConfig::from_name(name)?;
        Ok(self)
    }

    /// Set M parameter (number of bi-directional links).
    pub fn with_m(mut self, m: usize) -> Self {
        self.backend_config.set_m(m);
        self
    }

    /// Set efConstruction parameter.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.backend_config.set_ef_construction(ef);
        self
    }

    /// Set the distance metric (overrides auto-detection).
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.backend_config.set_distance_metric(metric);
        self.distance_metric_auto = false;
        self
    }

    /// Set compact mode.
    pub fn with_compact(mut self, compact: bool) -> Self {
        self.backend_config.set_compact(compact);
        self
    }

    /// Set recompute mode.
    pub fn with_recompute(mut self, recompute: bool) -> Self {
        self.backend_config.set_recompute(recompute);
        self
    }

    /// Set the number of threads for index construction.
    pub fn with_num_threads(mut self, n: usize) -> Self {
        self.backend_config.set_num_threads(n);
        self
    }

    /// Set embedding options.
    pub fn with_embedding_options(mut self, options: HashMap<String, serde_json::Value>) -> Self {
        self.embedding_options = options;
        self
    }

    /// Add a text chunk with optional metadata.
    pub fn add_text(&mut self, text: &str, metadata: HashMap<String, serde_json::Value>) {
        let id = metadata
            .get("id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| self.chunks.len().to_string());

        self.chunks.push(Passage {
            id,
            text: text.to_string(),
            metadata,
        });
    }

    /// Build the index using the provided embedding provider.
    pub fn build_index(
        &mut self,
        index_path: &Path,
        provider: &dyn EmbeddingProvider,
    ) -> Result<()> {
        if self.chunks.is_empty() {
            anyhow::bail!("No chunks added");
        }

        // Filter out empty chunks
        self.chunks.retain(|c| !c.text.trim().is_empty());
        if self.chunks.is_empty() {
            anyhow::bail!("All provided chunks are empty or invalid");
        }

        // Detect dimensions if not set
        if self.dimensions.is_none() {
            let dummy = provider.compute_embeddings(&["dummy".to_string()])?;
            self.dimensions = Some(dummy.ncols());
        }
        let dimensions = self.dimensions.unwrap();

        let paths = IndexPaths::new(index_path);

        // Create directory
        std::fs::create_dir_all(&paths.base_dir)?;

        // Collect texts for embedding computation before entering scope
        let texts: Vec<String> = self.chunks.iter().map(|c| c.text.clone()).collect();
        let ids: Vec<String> = self.chunks.iter().map(|c| c.id.clone()).collect();

        // Run passage/ID writing (CPU/IO) and embedding computation (GPU) in parallel.
        // write_passages serializes 322K+ passages as JSONL which takes ~80s for large
        // indexes; overlapping with GPU embedding hides this latency entirely.
        let (write_result, embed_result) = std::thread::scope(|s| {
            let passages_path = paths.passages_path();
            let offset_path = paths.offset_path();
            let id_map_path = paths.id_map_path();
            let chunks = &self.chunks;
            let ids_ref = &ids;

            let writer = s.spawn(move || -> Result<()> {
                info!("Writing {} passages to disk", chunks.len());
                write_passages(chunks, &passages_path, &offset_path)?;
                write_id_map(ids_ref, &id_map_path)?;
                info!("Passage writing complete");
                Ok(())
            });

            info!("Computing embeddings for {} chunks", texts.len());
            let emb = provider.compute_embeddings(&texts);

            let wr = writer.join().expect("passage writer thread panicked");
            (wr, emb)
        });

        write_result?;
        let mut embeddings = embed_result?;

        // Normalize for cosine distance
        if self.backend_config.distance_metric() == DistanceMetric::Cosine {
            normalize_l2_inplace(&mut embeddings);
        }

        // Build and write index
        backend::build_backend(&self.backend_config, &embeddings, &paths.index_file_path())?;

        // Write metadata
        let meta = IndexMeta {
            version: "1.0".to_string(),
            backend_name: self.backend_config.name().to_string(),
            embedding_model: self.embedding_model.clone(),
            dimensions,
            backend_kwargs: self.backend_config.to_backend_kwargs(),
            embedding_mode: self.embedding_mode.clone(),
            passage_sources: vec![PassageSource {
                source_type: "jsonl".to_string(),
                path: paths
                    .passages_path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                index_path: paths
                    .offset_path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                path_relative: Some(
                    paths
                        .passages_path()
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string(),
                ),
                index_path_relative: Some(
                    paths
                        .offset_path()
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_string(),
                ),
            }],
            embedding_options: self.embedding_options.clone(),
            is_compact: Some(self.backend_config.is_compact()),
            is_pruned: Some(self.backend_config.is_recompute()),
            total_passages: Some(self.chunks.len()),
            built_from_precomputed_embeddings: None,
            embeddings_source: None,
        };

        meta.save(&paths.meta_path())?;
        info!("Index built successfully at {}", index_path.display());

        Ok(())
    }

    /// Build index from pre-computed embeddings.
    pub fn build_index_from_embeddings(
        &mut self,
        index_path: &Path,
        ids: &[String],
        embeddings: &Array2<f32>,
    ) -> Result<()> {
        let dimensions = embeddings.ncols();
        self.dimensions = Some(dimensions);

        if ids.len() != embeddings.nrows() {
            anyhow::bail!(
                "Mismatch: {} IDs vs {} embeddings",
                ids.len(),
                embeddings.nrows()
            );
        }

        // Ensure we have passages for all embeddings
        if self.chunks.is_empty() {
            for id in ids {
                self.add_text(&format!("Document {}", id), {
                    let mut m = HashMap::new();
                    m.insert("id".to_string(), serde_json::json!(id));
                    m.insert("from_embeddings".to_string(), serde_json::json!(true));
                    m
                });
            }
        }

        let paths = IndexPaths::new(index_path);
        std::fs::create_dir_all(&paths.base_dir)?;

        write_passages(&self.chunks, &paths.passages_path(), &paths.offset_path())?;
        write_id_map(ids, &paths.id_map_path())?;

        let mut emb = embeddings.to_owned();
        if self.backend_config.distance_metric() == DistanceMetric::Cosine {
            normalize_l2_inplace(&mut emb);
        }

        backend::build_backend(&self.backend_config, &emb, &paths.index_file_path())?;

        let meta = IndexMeta {
            version: "1.0".to_string(),
            backend_name: self.backend_config.name().to_string(),
            embedding_model: self.embedding_model.clone(),
            dimensions,
            backend_kwargs: HashMap::new(),
            embedding_mode: self.embedding_mode.clone(),
            passage_sources: vec![PassageSource {
                source_type: "jsonl".to_string(),
                path: paths
                    .passages_path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                index_path: paths
                    .offset_path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                path_relative: None,
                index_path_relative: None,
            }],
            embedding_options: HashMap::new(),
            is_compact: Some(self.backend_config.is_compact()),
            is_pruned: Some(self.backend_config.is_recompute()),
            total_passages: Some(self.chunks.len()),
            built_from_precomputed_embeddings: Some(true),
            embeddings_source: None,
        };

        meta.save(&paths.meta_path())?;
        Ok(())
    }

    /// Create an embedding provider based on the builder's `embedding_mode` and `embedding_model`.
    ///
    /// Dispatches to:
    /// - `"ollama"` → `OllamaEmbedding`
    /// - `"openai"` → `OpenAiEmbedding`
    /// - `"gemini"` → `GeminiEmbedding`
    /// - `"sentence-transformers"` (default) → OpenAI fallback, then Ollama
    #[cfg(feature = "embedding-remote")]
    pub fn create_embedding_provider(&self) -> Result<Box<dyn EmbeddingProvider>> {
        let mode = EmbeddingMode::from_str_lossy(&self.embedding_mode);
        crate::embedding::create_embedding_provider(
            &mode,
            &self.embedding_model,
            &self.embedding_options,
        )
    }
}
