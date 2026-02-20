use anyhow::Result;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use tracing::warn;

use crate::bm25::BM25Scorer;
use crate::embedding::client::EmbeddingClient;
use crate::hnsw::graph::HnswGraph;
use crate::hnsw::io::read_hnsw_index;
use crate::hnsw::search::{search_hnsw_recompute, SearchParams};
use crate::index::{DistanceMetric, IndexMeta, IndexPaths};
use crate::metadata_filter::MetadataFilters;
use crate::passages::{load_id_map, PassageManager};
use crate::search_result::SearchResult;

/// High-level searcher for LEANN indexes.
pub struct LeannSearcher {
    #[allow(dead_code)]
    meta: IndexMeta,
    passages: PassageManager,
    graph: HnswGraph,
    id_map: Vec<String>,
    distance_metric: DistanceMetric,
    recompute_embeddings: bool,
    #[allow(dead_code)]
    bm25: Option<BM25Scorer>,
    #[allow(dead_code)]
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

        Ok(Self {
            meta,
            passages,
            graph,
            id_map,
            distance_metric,
            recompute_embeddings: recompute,
            bm25: None,
            meta_path,
        })
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
        if config.gemma == 0.0 {
            let results = self.bm25_search(query, top_k)?;
            if let Some(ref filters) = config.metadata_filters {
                return Ok(self.passages.filter_search_results(&results, filters));
            }
            return Ok(results);
        }

        // Handle grep search
        if config.use_grep {
            let results = self.grep_search(query, top_k)?;
            if let Some(ref filters) = config.metadata_filters {
                return Ok(self.passages.filter_search_results(&results, filters));
            }
            return Ok(results);
        }

        // Vector search requires an embedding client
        // For now, we need the embedding server to compute query embeddings
        let zmq_port = config.zmq_port.unwrap_or(5557);
        let client = EmbeddingClient::new(zmq_port);

        // Compute query embedding
        let query_embedding = client.compute_text_embeddings(&[query.to_string()])?;
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

        let params = SearchParams {
            ef_search: config.complexity,
            beam_size: config.beam_width,
            prune_ratio: config.prune_ratio,
            recompute_embeddings: self.recompute_embeddings,
            zmq_port: Some(zmq_port),
            batch_size: config.batch_size,
            ..Default::default()
        };

        // Search
        let (labels, distances) = if self.recompute_embeddings {
            let client = EmbeddingClient::new(zmq_port);
            search_hnsw_recompute(
                &self.graph,
                &query_vec,
                top_k,
                &params,
                |node_ids, q, out| {
                    let dists = client
                        .compute_distances(node_ids, q)
                        .unwrap_or_else(|_| vec![1e9; node_ids.len()]);
                    out[..dists.len()].copy_from_slice(&dists);
                },
            )
        } else {
            // Non-recompute: we'd need stored vectors
            // For now, fall back to recompute
            let client = EmbeddingClient::new(zmq_port);
            search_hnsw_recompute(
                &self.graph,
                &query_vec,
                top_k,
                &params,
                |node_ids, q, out| {
                    let dists = client
                        .compute_distances(node_ids, q)
                        .unwrap_or_else(|_| vec![1e9; node_ids.len()]);
                    out[..dists.len()].copy_from_slice(&dists);
                },
            )
        };

        // Map labels to string IDs and enrich with passages
        let mut results = Vec::new();
        for (label, dist) in labels.iter().zip(distances.iter()) {
            let string_id = self.map_label(*label);
            match self.passages.get_passage(&string_id) {
                Ok(passage) => {
                    results.push(SearchResult::with_metadata(
                        string_id,
                        *dist as f64,
                        passage.text,
                        passage.metadata.clone(),
                    ));
                }
                Err(e) => {
                    warn!("Passage not found for ID '{}': {}", string_id, e);
                }
            }
        }

        // Apply metadata filters
        if let Some(ref filters) = config.metadata_filters {
            let filtered = self.passages.filter_search_results(&results, filters);
            return Ok(filtered);
        }

        // Handle hybrid search
        if config.gemma < 1.0 {
            let bm25_results = self.bm25_search(query, top_k)?;
            let bm25_weight = 1.0 - config.gemma;

            let mut hybrid_scores: HashMap<String, f64> = HashMap::new();

            for r in &results {
                *hybrid_scores.entry(r.id.clone()).or_default() += config.gemma * r.score;
            }
            for r in &bm25_results {
                *hybrid_scores.entry(r.id.clone()).or_default() += bm25_weight * r.score;
            }

            let mut sorted: Vec<(String, f64)> = hybrid_scores.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sorted.truncate(top_k);

            let mut hybrid_results = Vec::new();
            for (id, score) in sorted {
                let text = results
                    .iter()
                    .find(|r| r.id == id)
                    .map(|r| r.text.clone())
                    .unwrap_or_default();
                let metadata = results
                    .iter()
                    .find(|r| r.id == id)
                    .map(|r| r.metadata.clone())
                    .unwrap_or_default();
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

    fn bm25_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        // TODO: Initialize BM25 lazily on first use
        // For now, create a fresh one each time (not ideal for perf)
        let mut scorer = BM25Scorer::default();

        let mut documents = Vec::new();
        for file_path in self.passages.passage_files() {
            let file = std::fs::File::open(file_path)?;
            let reader = std::io::BufReader::new(file);
            use std::io::BufRead;
            for line in reader.lines() {
                let line = line?;
                if let Ok(passage) = serde_json::from_str::<crate::passages::Passage>(&line) {
                    documents.push((passage.id, passage.text));
                }
            }
        }

        scorer.fit(&documents);
        Ok(scorer.search(query, top_k))
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
                if pattern.is_match(&line) {
                    if let Ok(passage) = serde_json::from_str::<crate::passages::Passage>(&line) {
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
        // Cleanup embedding server resources
    }
}

/// Search configuration options.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub complexity: usize,
    pub beam_width: usize,
    pub prune_ratio: f64,
    pub metadata_filters: Option<MetadataFilters>,
    pub batch_size: usize,
    pub use_grep: bool,
    /// Weight of vector search (0.0 = pure BM25, 1.0 = pure vector).
    pub gemma: f64,
    pub zmq_port: Option<u16>,
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
            zmq_port: None,
        }
    }
}

impl Drop for LeannSearcher {
    fn drop(&mut self) {
        self.cleanup();
    }
}
