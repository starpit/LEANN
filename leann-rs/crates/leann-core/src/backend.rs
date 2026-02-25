//! Backend abstraction for LEANN index engines.
//!
//! Dispatches build, read, and search operations to the appropriate backend
//! (currently HNSW only) via enum matching. Zero runtime overhead — the
//! compiler devirtualizes each match arm.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;

use anyhow::Result;
use ndarray::Array2;
use tracing::info;

use crate::hnsw::build::build_hnsw_with_threads;
use crate::hnsw::csr::convert_to_csr;
use crate::hnsw::graph::{HnswConfig, HnswGraph, VectorStorage};
use crate::hnsw::io::{read_hnsw_index, write_hnsw_compact, write_hnsw_standard};
use crate::hnsw::search::{SearchParams, search_hnsw, search_hnsw_recompute};
use crate::index::DistanceMetric;

// Re-export search types so callers don't need to reach into hnsw::search.
pub use crate::hnsw::search::PruningStrategy;

// ---------------------------------------------------------------------------
// BackendConfig
// ---------------------------------------------------------------------------

/// Backend-specific build configuration.
///
/// Each variant holds all parameters needed to build an index with that backend.
#[derive(Debug)]
pub enum BackendConfig {
    Hnsw {
        m: usize,
        ef_construction: usize,
        distance_metric: DistanceMetric,
        is_compact: bool,
        is_recompute: bool,
        num_threads: usize,
        seed: Option<u64>,
    },
    // Future: Ivf { nlist: usize, distance_metric: DistanceMetric },
}

impl BackendConfig {
    /// Default HNSW configuration (matches `HnswConfig::default()`).
    pub fn hnsw_default() -> Self {
        let defaults = HnswConfig::default();
        Self::Hnsw {
            m: defaults.m,
            ef_construction: defaults.ef_construction,
            distance_metric: defaults.distance_metric,
            is_compact: defaults.is_compact,
            is_recompute: defaults.is_recompute,
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            seed: defaults.seed,
        }
    }

    /// Create a default config for the given backend name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "hnsw" => Ok(Self::hnsw_default()),
            other => anyhow::bail!(
                "Backend '{}' is not supported. Available backends: hnsw",
                other
            ),
        }
    }

    /// Backend name string (for serialization into `IndexMeta`).
    pub fn name(&self) -> &str {
        match self {
            Self::Hnsw { .. } => "hnsw",
        }
    }

    /// Distance metric for this backend configuration.
    pub fn distance_metric(&self) -> DistanceMetric {
        match self {
            Self::Hnsw {
                distance_metric, ..
            } => *distance_metric,
        }
    }

    /// Set the distance metric.
    pub fn set_distance_metric(&mut self, metric: DistanceMetric) {
        match self {
            Self::Hnsw {
                distance_metric, ..
            } => *distance_metric = metric,
        }
    }

    /// Set M parameter (HNSW only).
    pub fn set_m(&mut self, val: usize) {
        match self {
            Self::Hnsw { m, .. } => *m = val,
        }
    }

    /// Set efConstruction parameter (HNSW only).
    pub fn set_ef_construction(&mut self, val: usize) {
        match self {
            Self::Hnsw {
                ef_construction, ..
            } => *ef_construction = val,
        }
    }

    /// Set compact mode (HNSW only).
    pub fn set_compact(&mut self, val: bool) {
        match self {
            Self::Hnsw { is_compact, .. } => *is_compact = val,
        }
    }

    /// Set recompute mode (HNSW only).
    pub fn set_recompute(&mut self, val: bool) {
        match self {
            Self::Hnsw { is_recompute, .. } => *is_recompute = val,
        }
    }

    /// Set number of build threads (HNSW only).
    pub fn set_num_threads(&mut self, val: usize) {
        match self {
            Self::Hnsw { num_threads, .. } => *num_threads = val.max(1),
        }
    }

    /// Convert to `backend_kwargs` map for `IndexMeta` serialization.
    pub fn to_backend_kwargs(&self) -> HashMap<String, serde_json::Value> {
        match self {
            Self::Hnsw {
                m,
                ef_construction,
                distance_metric,
                is_compact,
                is_recompute,
                ..
            } => {
                let mut kwargs = HashMap::new();
                kwargs.insert("M".to_string(), serde_json::json!(m));
                kwargs.insert(
                    "efConstruction".to_string(),
                    serde_json::json!(ef_construction),
                );
                kwargs.insert(
                    "distance_metric".to_string(),
                    serde_json::json!(match distance_metric {
                        DistanceMetric::L2 => "l2",
                        DistanceMetric::Cosine => "cosine",
                        DistanceMetric::Mips => "mips",
                    }),
                );
                kwargs.insert("is_compact".to_string(), serde_json::json!(is_compact));
                kwargs.insert("is_recompute".to_string(), serde_json::json!(is_recompute));
                kwargs
            }
        }
    }

    /// Extract an `HnswConfig` from this configuration (panics if not HNSW).
    pub fn to_hnsw_config(&self) -> HnswConfig {
        match self {
            Self::Hnsw {
                m,
                ef_construction,
                distance_metric,
                is_compact,
                is_recompute,
                seed,
                ..
            } => HnswConfig {
                m: *m,
                ef_construction: *ef_construction,
                ef_search: 64, // search-time param, not stored in build config
                distance_metric: *distance_metric,
                is_compact: *is_compact,
                is_recompute: *is_recompute,
                seed: *seed,
            },
        }
    }

    /// Whether this config uses compact storage.
    pub fn is_compact(&self) -> bool {
        match self {
            Self::Hnsw { is_compact, .. } => *is_compact,
        }
    }

    /// Whether this config uses recompute mode.
    pub fn is_recompute(&self) -> bool {
        match self {
            Self::Hnsw { is_recompute, .. } => *is_recompute,
        }
    }
}

// ---------------------------------------------------------------------------
// BackendIndex
// ---------------------------------------------------------------------------

/// A loaded backend index, ready for search.
pub enum BackendIndex {
    Hnsw(HnswGraph),
    // Future: Ivf { ... },
}

impl std::fmt::Debug for BackendIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hnsw(g) => f
                .debug_struct("BackendIndex::Hnsw")
                .field("ntotal", &g.ntotal)
                .field("dimensions", &g.dimensions)
                .finish(),
        }
    }
}

impl BackendIndex {
    /// Total number of indexed vectors.
    pub fn ntotal(&self) -> usize {
        match self {
            Self::Hnsw(g) => g.ntotal,
        }
    }

    /// Dimensionality of indexed vectors.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Hnsw(g) => g.dimensions,
        }
    }

    /// Whether vector storage has been pruned (recompute mode).
    pub fn is_pruned(&self) -> bool {
        match self {
            Self::Hnsw(g) => g.is_pruned(),
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch functions
// ---------------------------------------------------------------------------

/// Build an index and write it to disk.
///
/// Handles graph construction, optional CSR compaction, optional vector storage,
/// and serialization — the full pipeline from embeddings to on-disk index.
pub fn build_backend(
    config: &BackendConfig,
    embeddings: &Array2<f32>,
    index_file: &Path,
) -> Result<()> {
    match config {
        BackendConfig::Hnsw {
            num_threads,
            is_recompute,
            is_compact,
            distance_metric,
            ..
        } => {
            let hnsw_config = config.to_hnsw_config();

            info!(
                "Building HNSW graph (M={}, efConstruction={})",
                hnsw_config.m, hnsw_config.ef_construction
            );
            let mut graph = build_hnsw_with_threads(embeddings, &hnsw_config, *num_threads)?;

            // Store vectors if not using recompute
            if !is_recompute {
                let flat: Vec<f32> = embeddings.iter().copied().collect();
                let storage_bytes = flat
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<u8>>();

                let fourcc = match distance_metric {
                    DistanceMetric::L2 => u32::from_le_bytes(*b"IxFl"),
                    _ => u32::from_le_bytes(*b"IxFI"),
                };

                graph.vector_storage = VectorStorage::Raw {
                    fourcc,
                    data: storage_bytes,
                };
            }

            // Convert to CSR if compact mode
            let graph = if *is_compact {
                info!("Converting to compact CSR format");
                convert_to_csr(&graph)?
            } else {
                graph
            };

            // Write index file
            let mut file = std::fs::File::create(index_file)?;
            if graph.is_compact() {
                write_hnsw_compact(&mut file, &graph)?;
            } else {
                write_hnsw_standard(&mut file, &graph)?;
            }

            Ok(())
        }
    }
}

/// Read an index from disk.
pub fn read_backend_index(backend_name: &str, index_file: &Path) -> Result<BackendIndex> {
    match backend_name {
        "hnsw" => {
            let index_data = std::fs::read(index_file)?;
            let mut cursor = Cursor::new(index_data);
            let graph = read_hnsw_index(&mut cursor)?;
            Ok(BackendIndex::Hnsw(graph))
        }
        other => anyhow::bail!("Unknown backend '{}' — cannot read index", other),
    }
}

/// Search using stored vectors.
pub fn search_backend(
    index: &BackendIndex,
    query: &[f32],
    top_k: usize,
    params: &SearchParams,
) -> (Vec<usize>, Vec<f32>) {
    match index {
        BackendIndex::Hnsw(graph) => {
            match &graph.vector_storage {
                VectorStorage::Raw { data, .. } => {
                    let flat_vectors: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    search_hnsw(graph, query, top_k, &flat_vectors, params)
                }
                VectorStorage::Null => {
                    // No stored vectors — return empty results.
                    // Caller should use search_backend_recompute instead.
                    (Vec::new(), Vec::new())
                }
            }
        }
    }
}

/// Search using recomputed distances (embedding provider callback).
pub fn search_backend_recompute<F>(
    index: &BackendIndex,
    query: &[f32],
    top_k: usize,
    params: &SearchParams,
    compute_distance: F,
) -> (Vec<usize>, Vec<f32>)
where
    F: FnMut(&[usize], &[f32], &mut [f32]),
{
    match index {
        BackendIndex::Hnsw(graph) => {
            search_hnsw_recompute(graph, query, top_k, params, compute_distance)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_hnsw_default() {
        let cfg = BackendConfig::hnsw_default();
        assert_eq!(cfg.name(), "hnsw");
        assert_eq!(cfg.distance_metric(), DistanceMetric::Mips);
        assert!(cfg.is_compact());
        assert!(cfg.is_recompute());
    }

    #[test]
    fn test_backend_config_from_name() {
        assert!(BackendConfig::from_name("hnsw").is_ok());
        assert!(BackendConfig::from_name("ivf").is_err());
        assert!(BackendConfig::from_name("unknown").is_err());
    }

    #[test]
    fn test_backend_config_setters() {
        let mut cfg = BackendConfig::hnsw_default();
        cfg.set_m(16);
        cfg.set_ef_construction(100);
        cfg.set_compact(false);
        cfg.set_recompute(false);
        cfg.set_distance_metric(DistanceMetric::L2);
        cfg.set_num_threads(4);

        assert!(!cfg.is_compact());
        assert!(!cfg.is_recompute());
        assert_eq!(cfg.distance_metric(), DistanceMetric::L2);

        let hnsw = cfg.to_hnsw_config();
        assert_eq!(hnsw.m, 16);
        assert_eq!(hnsw.ef_construction, 100);
        assert!(!hnsw.is_compact);
        assert!(!hnsw.is_recompute);
        assert_eq!(hnsw.distance_metric, DistanceMetric::L2);
    }

    #[test]
    fn test_backend_kwargs_serialization() {
        let cfg = BackendConfig::hnsw_default();
        let kwargs = cfg.to_backend_kwargs();
        assert_eq!(kwargs["M"], serde_json::json!(32));
        assert_eq!(kwargs["efConstruction"], serde_json::json!(200));
        assert_eq!(kwargs["distance_metric"], serde_json::json!("mips"));
        assert_eq!(kwargs["is_compact"], serde_json::json!(true));
        assert_eq!(kwargs["is_recompute"], serde_json::json!(true));
    }

    #[test]
    fn test_read_backend_index_unknown() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = read_backend_index("unknown", tmp.path());
        assert!(result.is_err());
    }
}
