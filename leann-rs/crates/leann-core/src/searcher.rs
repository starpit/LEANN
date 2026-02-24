use anyhow::Result;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use tracing::warn;

#[cfg(feature = "bm25")]
use crate::bm25::BM25Scorer;
use crate::embedding::EmbeddingProvider;
use crate::hnsw::graph::HnswGraph;
use crate::hnsw::io::read_hnsw_index;
use crate::hnsw::search::{PruningStrategy, SearchParams, search_hnsw, search_hnsw_recompute};
use crate::hnsw::simd::{inner_product_distance, l2_distance};
use crate::index::{DistanceMetric, IndexMeta, IndexPaths};
#[cfg(feature = "bm25")]
use crate::passages::Passage;
use crate::passages::{PassageManager, load_id_map};
use crate::search_result::SearchResult;

/// Options for opening a LEANN searcher with non-default behavior.
#[derive(Default)]
pub struct SearcherOptions {
    /// Override `recompute_embeddings` from meta.json. `None` = use meta default.
    pub recompute_embeddings: Option<bool>,
    /// If true, send a probe embedding request at construction to verify the provider.
    pub enable_warmup: bool,
}

/// High-level searcher for LEANN indexes.
#[allow(dead_code)]
pub struct LeannSearcher {
    meta: IndexMeta,
    passages: PassageManager,
    graph: HnswGraph,
    id_map: Vec<String>,
    distance_metric: DistanceMetric,
    recompute_embeddings: bool,
    provider: Option<Arc<dyn EmbeddingProvider>>,
    #[cfg(feature = "bm25")]
    bm25: Option<BM25Scorer>,
    meta_path: std::path::PathBuf,
}

impl LeannSearcher {
    /// Open an existing LEANN index for searching.
    pub fn open(index_path: &Path) -> Result<Self> {
        let index_path = if index_path.is_relative() {
            std::env::current_dir()?.join(index_path)
        } else {
            index_path.to_path_buf()
        };

        let paths = IndexPaths::new(&index_path);
        let meta_path = paths.meta_path();

        if !meta_path.exists() {
            anyhow::bail!("LEANN metadata file not found at {}", meta_path.display());
        }

        let meta = IndexMeta::load(&meta_path)?;
        let distance_metric = meta.distance_metric();
        let recompute = meta.requires_recompute();

        // Load passages
        let passages = PassageManager::load(&meta.passage_sources, Some(&meta_path))?;

        // Load HNSW graph
        let index_file = paths.index_file_path();
        if !index_file.exists() {
            anyhow::bail!("HNSW index file not found at {}", index_file.display());
        }
        let index_data = std::fs::read(&index_file)?;
        let mut cursor = Cursor::new(index_data);
        let graph = read_hnsw_index(&mut cursor)?;

        // Load ID map
        let id_map_path = paths.id_map_path();
        let id_map = if id_map_path.exists() {
            load_id_map(&id_map_path)?
        } else {
            Vec::new()
        };

        // Construct embedding provider from meta
        let provider = Self::create_provider_from_meta(&meta);

        Ok(Self {
            meta,
            passages,
            graph,
            id_map,
            distance_metric,
            recompute_embeddings: recompute,
            provider,
            #[cfg(feature = "bm25")]
            bm25: None,
            meta_path,
        })
    }

    /// Open an existing LEANN index with custom options.
    ///
    /// This allows overriding `recompute_embeddings` from meta.json and
    /// optionally warming up the embedding provider at construction time.
    pub fn open_with_options(index_path: &Path, options: &SearcherOptions) -> Result<Self> {
        let mut searcher = Self::open(index_path)?;

        // Override recompute_embeddings if explicitly specified
        if let Some(recompute) = options.recompute_embeddings {
            searcher.recompute_embeddings = recompute;
        }

        // Warmup: send a probe embedding request to verify the provider responds
        if options.enable_warmup {
            searcher.warmup()?;
        }

        Ok(searcher)
    }

    /// Send a probe embedding request to verify the provider is reachable.
    ///
    /// This is useful for detecting misconfiguration early (e.g. Ollama not running)
    /// rather than waiting until the first search call.
    pub fn warmup(&self) -> Result<()> {
        if let Some(ref provider) = self.provider {
            match provider.compute_embeddings(&["__LEANN_WARMUP__".to_string()]) {
                Ok(_) => {}
                Err(e) => {
                    warn!("Warmup embedding request failed (provider may not be running): {e}");
                }
            }
        }
        Ok(())
    }

    /// Construct an embedding provider from index metadata.
    #[cfg(feature = "embedding-remote")]
    fn create_provider_from_meta(meta: &IndexMeta) -> Option<Arc<dyn EmbeddingProvider>> {
        use crate::embedding::{EmbeddingMode, create_embedding_provider};

        let mode = EmbeddingMode::from_str_lossy(&meta.embedding_mode);
        match create_embedding_provider(&mode, &meta.embedding_model, &meta.embedding_options) {
            Ok(provider) => Some(Arc::from(provider)),
            Err(e) => {
                warn!("Could not create embedding provider from meta: {e}");
                None
            }
        }
    }

    #[cfg(not(feature = "embedding-remote"))]
    fn create_provider_from_meta(_meta: &IndexMeta) -> Option<Arc<dyn EmbeddingProvider>> {
        None
    }

    /// Search for nearest neighbors.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        self.search_with_params(query, top_k, &SearchConfig::default())
    }

    /// Search with full configuration.
    pub fn search_with_params(
        &self,
        query: &str,
        top_k: usize,
        config: &SearchConfig,
    ) -> Result<Vec<SearchResult>> {
        let top_k = top_k.min(self.passages.len());

        // Handle pure BM25 search
        #[cfg(feature = "bm25")]
        if config.gemma == 0.0 {
            let results = self.bm25_search(query, top_k)?;
            if let Some(ref filters) = config.metadata_filters {
                return Ok(self.passages.filter_search_results(&results, filters));
            }
            return Ok(results);
        }
        #[cfg(not(feature = "bm25"))]
        if config.gemma == 0.0 {
            anyhow::bail!("BM25 search requires the `bm25` feature");
        }

        // Handle grep search
        if config.use_grep {
            let results = self.grep_search(query, top_k)?;
            if let Some(ref filters) = config.metadata_filters {
                return Ok(self.passages.filter_search_results(&results, filters));
            }
            return Ok(results);
        }

        // Vector search requires an embedding provider
        let results = self.vector_search(query, top_k, config)?;
        Ok(results)
    }

    fn vector_search(
        &self,
        query: &str,
        top_k: usize,
        config: &SearchConfig,
    ) -> Result<Vec<SearchResult>> {
        let provider = self.provider.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "No embedding provider available. Ensure the index was built with a supported \
                 embedding mode (ollama, openai, gemini) and the `embedding-remote` feature is enabled."
            )
        })?;

        // Compute query embedding
        let query_embedding = provider.compute_embeddings(&[query.to_string()])?;
        let query_vec: Vec<f32> = query_embedding.row(0).to_vec();

        // Normalize for cosine
        let query_vec = if self.distance_metric == DistanceMetric::Cosine {
            let norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                query_vec.iter().map(|x| x / norm).collect()
            } else {
                query_vec
            }
        } else {
            query_vec
        };

        let pruning_strategy = config
            .pruning_strategy
            .as_deref()
            .map(|s| match s {
                "local" => PruningStrategy::Local,
                "proportional" => PruningStrategy::Proportional,
                _ => PruningStrategy::Global,
            })
            .unwrap_or(PruningStrategy::Global);

        let params = SearchParams {
            ef_search: config.complexity,
            beam_size: config.beam_width,
            prune_ratio: config.prune_ratio,
            recompute_embeddings: self.recompute_embeddings,
            batch_size: config.batch_size,
            pruning_strategy,
            ..Default::default()
        };

        // Search
        let (labels, distances) = if self.recompute_embeddings {
            // Recompute: look up passage texts, compute embeddings, compute distances locally
            let provider = Arc::clone(provider);
            let passages = &self.passages;
            let distance_metric = self.distance_metric;

            search_hnsw_recompute(
                &self.graph,
                &query_vec,
                top_k,
                &params,
                |node_ids, q, out| {
                    // Look up texts for each node ID
                    let mut texts = Vec::new();
                    let mut found_indices = Vec::new();

                    for (idx, &nid) in node_ids.iter().enumerate() {
                        if let Ok(passage) = passages.get_passage_by_index(nid)
                            && !passage.text.is_empty()
                        {
                            texts.push(passage.text);
                            found_indices.push(idx);
                        }
                    }

                    // Default to large distance for unfound passages
                    for d in out.iter_mut().take(node_ids.len()) {
                        *d = 1e9;
                    }

                    if texts.is_empty() {
                        return;
                    }

                    if let Ok(embeddings) = provider.compute_embeddings(&texts) {
                        for (i, &original_idx) in found_indices.iter().enumerate() {
                            let emb = embeddings.row(i);
                            let emb_slice = emb.as_slice().unwrap();
                            let dist = match distance_metric {
                                DistanceMetric::L2 => l2_distance(q, emb_slice),
                                _ => inner_product_distance(q, emb_slice),
                            };
                            out[original_idx] = dist;
                        }
                    }
                },
            )
        } else {
            // Non-recompute: use stored vectors
            match &self.graph.vector_storage {
                crate::hnsw::graph::VectorStorage::Raw { data, .. } => {
                    let flat_vectors: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    search_hnsw(&self.graph, &query_vec, top_k, &flat_vectors, &params)
                }
                _ => {
                    // No stored vectors — fall back to recompute via provider
                    let provider = Arc::clone(provider);
                    let passages = &self.passages;
                    let distance_metric = self.distance_metric;

                    search_hnsw_recompute(
                        &self.graph,
                        &query_vec,
                        top_k,
                        &params,
                        |node_ids, q, out| {
                            let mut texts = Vec::new();
                            let mut found_indices = Vec::new();

                            for (idx, &nid) in node_ids.iter().enumerate() {
                                if let Ok(passage) = passages.get_passage_by_index(nid)
                                    && !passage.text.is_empty()
                                {
                                    texts.push(passage.text);
                                    found_indices.push(idx);
                                }
                            }

                            for d in out.iter_mut().take(node_ids.len()) {
                                *d = 1e9;
                            }

                            if texts.is_empty() {
                                return;
                            }

                            if let Ok(embeddings) = provider.compute_embeddings(&texts) {
                                for (i, &original_idx) in found_indices.iter().enumerate() {
                                    let emb = embeddings.row(i);
                                    let emb_slice = emb.as_slice().unwrap();
                                    let dist = match distance_metric {
                                        DistanceMetric::L2 => l2_distance(q, emb_slice),
                                        _ => inner_product_distance(q, emb_slice),
                                    };
                                    out[original_idx] = dist;
                                }
                            }
                        },
                    )
                }
            }
        };

        // Map labels to passages and enrich results
        let mut results = Vec::new();
        for (label, dist) in labels.iter().zip(distances.iter()) {
            let string_id = self.map_label(*label);
            match self.passages.get_passage_by_index(*label) {
                Ok(passage) => {
                    results.push(SearchResult::with_metadata(
                        string_id,
                        *dist as f64,
                        passage.text,
                        passage.metadata,
                    ));
                }
                Err(e) => {
                    warn!("Passage not found for label {}: {}", label, e);
                }
            }
        }

        // Apply metadata filters
        if let Some(ref filters) = config.metadata_filters {
            let filtered = self.passages.filter_search_results(&results, filters);
            return Ok(filtered);
        }

        // Handle hybrid search
        #[cfg(feature = "bm25")]
        if config.gemma < 1.0 {
            let bm25_results = self.bm25_search(query, top_k)?;
            let bm25_weight = 1.0 - config.gemma;

            let mut hybrid_scores: HashMap<String, f64> = HashMap::new();

            for r in &results {
                if let Some(s) = hybrid_scores.get_mut(&r.id) {
                    *s += config.gemma * r.score;
                } else {
                    hybrid_scores.insert(r.id.clone(), config.gemma * r.score);
                }
            }
            for r in &bm25_results {
                if let Some(s) = hybrid_scores.get_mut(&r.id) {
                    *s += bm25_weight * r.score;
                } else {
                    hybrid_scores.insert(r.id.clone(), bm25_weight * r.score);
                }
            }

            let mut sorted: Vec<(String, f64)> = hybrid_scores.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sorted.truncate(top_k);

            // Build lookup for text/metadata to avoid O(k·n) linear scans
            let result_lookup: HashMap<&str, usize> = results
                .iter()
                .enumerate()
                .map(|(i, r)| (r.id.as_str(), i))
                .collect();

            let mut hybrid_results = Vec::new();
            for (id, score) in sorted {
                let (text, metadata) = match result_lookup.get(id.as_str()) {
                    Some(&idx) => (results[idx].text.clone(), results[idx].metadata.clone()),
                    None => (String::new(), HashMap::new()),
                };
                hybrid_results.push(SearchResult::with_metadata(id, score, text, metadata));
            }

            return Ok(hybrid_results);
        }

        Ok(results)
    }

    fn map_label(&self, label: usize) -> String {
        if !self.id_map.is_empty() && label < self.id_map.len() {
            self.id_map[label].clone()
        } else {
            label.to_string()
        }
    }

    #[cfg(feature = "bm25")]
    fn bm25_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let mut scorer = BM25Scorer::default();

        let mut documents = Vec::new();
        let mut passage_map: HashMap<String, Passage> = HashMap::new();
        for file_path in self.passages.passage_files() {
            let file = std::fs::File::open(file_path)?;
            let reader = std::io::BufReader::new(file);
            use std::io::BufRead;
            for line in reader.lines() {
                let line = line?;
                if let Ok(passage) = serde_json::from_str::<Passage>(&line) {
                    documents.push((passage.id.clone(), passage.text.clone()));
                    passage_map.insert(passage.id.clone(), passage);
                }
            }
        }

        scorer.fit(&documents);
        let mut results = scorer.search(query, top_k);

        // Enrich results with passage text and metadata
        for result in &mut results {
            if let Some(passage) = passage_map.get(&result.id) {
                result.text.clone_from(&passage.text);
                result.metadata.clone_from(&passage.metadata);
            }
        }

        Ok(results)
    }

    fn grep_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let pattern = regex::RegexBuilder::new(&regex::escape(query))
            .case_insensitive(true)
            .build()?;

        let mut matches = Vec::new();
        for file_path in self.passages.passage_files() {
            let file = std::fs::File::open(file_path)?;
            let reader = std::io::BufReader::new(file);
            use std::io::BufRead;
            for line in reader.lines() {
                let line = line?;
                if pattern.is_match(&line)
                    && let Ok(passage) = serde_json::from_str::<crate::passages::Passage>(&line)
                {
                    let count = pattern.find_iter(&passage.text).count();
                    matches.push(SearchResult::with_metadata(
                        passage.id,
                        count as f64,
                        passage.text,
                        passage.metadata,
                    ));
                }
            }
        }

        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(top_k);
        Ok(matches)
    }

    pub fn cleanup(&mut self) {
        // Cleanup resources (provider is Arc-dropped automatically)
    }
}

/// Search configuration options.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub complexity: usize,
    pub beam_width: usize,
    pub prune_ratio: f64,
    pub metadata_filters: Option<HashMap<String, HashMap<String, serde_json::Value>>>,
    pub batch_size: usize,
    pub use_grep: bool,
    /// Weight of vector search (0.0 = pure BM25, 1.0 = pure vector).
    pub gemma: f64,
    /// Pruning strategy: "global", "local", or "proportional".
    pub pruning_strategy: Option<String>,
    /// Provider options (e.g. prompt_template overrides) passed at query time.
    pub provider_options: Option<HashMap<String, serde_json::Value>>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            complexity: 64,
            beam_width: 1,
            prune_ratio: 0.0,
            metadata_filters: None,
            batch_size: 0,
            use_grep: false,
            gemma: 1.0,
            pruning_strategy: None,
            provider_options: None,
        }
    }
}

impl Drop for LeannSearcher {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_searcher_options_default() {
        let opts = SearcherOptions::default();
        assert!(!opts.enable_warmup);
        assert!(opts.recompute_embeddings.is_none());
    }
}
