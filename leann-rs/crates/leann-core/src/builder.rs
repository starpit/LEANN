use anyhow::Result;
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::embedding::EmbeddingProvider;
use crate::hnsw::build::build_hnsw_with_threads;
use crate::hnsw::csr::convert_to_csr;
use crate::hnsw::graph::{HnswConfig, VectorStorage};
use crate::hnsw::io::{write_hnsw_compact, write_hnsw_standard};
use crate::hnsw::simd::normalize_l2_inplace;
use crate::index::{DistanceMetric, IndexMeta, IndexPaths, PassageSource};
use crate::passages::{Passage, write_id_map, write_passages};

/// Builder for creating LEANN indexes.
pub struct LeannBuilder {
    embedding_model: String,
    dimensions: Option<usize>,
    embedding_mode: String,
    config: HnswConfig,
    num_threads: usize,
    chunks: Vec<Passage>,
    embedding_options: HashMap<String, serde_json::Value>,
}

impl LeannBuilder {
    pub fn new(embedding_model: &str, dimensions: Option<usize>, embedding_mode: &str) -> Self {
        Self {
            embedding_model: embedding_model.to_string(),
            dimensions,
            embedding_mode: embedding_mode.to_string(),
            config: HnswConfig::default(),
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            chunks: Vec::new(),
            embedding_options: HashMap::new(),
        }
    }

    /// Configure HNSW parameters.
    pub fn with_config(mut self, config: HnswConfig) -> Self {
        self.config = config;
        self
    }

    /// Set M parameter (number of bi-directional links).
    pub fn with_m(mut self, m: usize) -> Self {
        self.config.m = m;
        self
    }

    /// Set efConstruction parameter.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.config.ef_construction = ef;
        self
    }

    /// Set the distance metric.
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.config.distance_metric = metric;
        self
    }

    /// Set compact mode.
    pub fn with_compact(mut self, compact: bool) -> Self {
        self.config.is_compact = compact;
        self
    }

    /// Set recompute mode.
    pub fn with_recompute(mut self, recompute: bool) -> Self {
        self.config.is_recompute = recompute;
        self
    }

    /// Set the number of threads for HNSW construction.
    pub fn with_num_threads(mut self, n: usize) -> Self {
        self.num_threads = n.max(1);
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
        if self.config.distance_metric == DistanceMetric::Cosine {
            normalize_l2_inplace(&mut embeddings);
        }

        // Build HNSW graph
        info!(
            "Building HNSW graph (M={}, efConstruction={})",
            self.config.m, self.config.ef_construction
        );
        let mut graph = build_hnsw_with_threads(&embeddings, &self.config, self.num_threads)?;

        // Store vectors if not using recompute
        if !self.config.is_recompute {
            let flat: Vec<f32> = embeddings.iter().copied().collect();
            // Create FAISS-compatible IndexFlat storage
            // For now, we store as raw bytes
            let storage_bytes = flat
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>();

            // FourCC for IndexFlatIP or IndexFlatL2
            let fourcc = match self.config.distance_metric {
                DistanceMetric::L2 => u32::from_le_bytes(*b"IxFl"),
                _ => u32::from_le_bytes(*b"IxFI"),
            };

            graph.vector_storage = VectorStorage::Raw {
                fourcc,
                data: storage_bytes,
            };
        }

        // Convert to CSR if compact mode
        let graph = if self.config.is_compact {
            info!("Converting to compact CSR format");
            convert_to_csr(&graph)?
        } else {
            graph
        };

        // Write index file
        let index_file = paths.index_file_path();
        let mut file = std::fs::File::create(&index_file)?;
        if graph.is_compact() {
            write_hnsw_compact(&mut file, &graph)?;
        } else {
            write_hnsw_standard(&mut file, &graph)?;
        }

        // Write metadata
        let meta = IndexMeta {
            version: "1.0".to_string(),
            backend_name: "hnsw".to_string(),
            embedding_model: self.embedding_model.clone(),
            dimensions,
            backend_kwargs: {
                let mut kwargs = HashMap::new();
                kwargs.insert("M".to_string(), serde_json::json!(self.config.m));
                kwargs.insert(
                    "efConstruction".to_string(),
                    serde_json::json!(self.config.ef_construction),
                );
                kwargs.insert(
                    "distance_metric".to_string(),
                    serde_json::json!(match self.config.distance_metric {
                        DistanceMetric::L2 => "l2",
                        DistanceMetric::Cosine => "cosine",
                        DistanceMetric::Mips => "mips",
                    }),
                );
                kwargs.insert(
                    "is_compact".to_string(),
                    serde_json::json!(self.config.is_compact),
                );
                kwargs.insert(
                    "is_recompute".to_string(),
                    serde_json::json!(self.config.is_recompute),
                );
                kwargs
            },
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
            is_compact: Some(self.config.is_compact),
            is_pruned: Some(self.config.is_recompute),
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
        if self.config.distance_metric == DistanceMetric::Cosine {
            normalize_l2_inplace(&mut emb);
        }

        let graph = build_hnsw_with_threads(&emb, &self.config, self.num_threads)?;

        if self.config.is_compact {
            let csr = convert_to_csr(&graph)?;
            let mut file = std::fs::File::create(paths.index_file_path())?;
            write_hnsw_compact(&mut file, &csr)?;
        } else {
            let mut file = std::fs::File::create(paths.index_file_path())?;
            write_hnsw_standard(&mut file, &graph)?;
        }

        let meta = IndexMeta {
            version: "1.0".to_string(),
            backend_name: "hnsw".to_string(),
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
            is_compact: Some(self.config.is_compact),
            is_pruned: Some(self.config.is_recompute),
            total_passages: Some(self.chunks.len()),
            built_from_precomputed_embeddings: Some(true),
            embeddings_source: None,
        };

        meta.save(&paths.meta_path())?;
        Ok(())
    }
}
