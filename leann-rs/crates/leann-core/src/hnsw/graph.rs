use crate::index::DistanceMetric;
use serde::{Deserialize, Serialize};

/// Configuration parameters for an HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of bi-directional links per element (higher = better recall, more memory).
    pub m: usize,
    /// Size of the dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search.
    pub ef_search: usize,
    /// Distance metric for similarity computation.
    pub distance_metric: DistanceMetric,
    /// Whether to use compact CSR storage.
    pub is_compact: bool,
    /// Whether the index uses embedding recomputation (pruned storage).
    pub is_recompute: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 32,
            ef_construction: 200,
            ef_search: 64,
            distance_metric: DistanceMetric::Mips,
            is_compact: true,
            is_recompute: true,
        }
    }
}

/// The HNSW graph structure.
///
/// Supports two storage modes:
/// 1. **Standard (non-compact)**: Flat adjacency lists with offset arrays (FAISS-compatible).
/// 2. **Compact (CSR)**: Compressed sparse row format with separate level pointers.
pub struct HnswGraph {
    /// Number of vectors indexed.
    pub ntotal: usize,
    /// Dimensionality of the vectors.
    pub dimensions: usize,
    /// Entry point node for search.
    pub entry_point: i32,
    /// Maximum level in the graph.
    pub max_level: i32,
    /// Level assignment for each node (levels[i] = number of levels for node i).
    pub levels: Vec<i32>,
    /// Probability of level assignment.
    pub assign_probas: Vec<f64>,
    /// Cumulative number of neighbors per level.
    pub cum_nneighbor_per_level: Vec<i32>,
    /// Configuration used during build.
    pub config: HnswConfig,
    /// FAISS metric type (1 = INNER_PRODUCT, 0 = L2).
    pub metric_type: i32,
    /// Metric argument (for custom metrics).
    pub metric_arg: f32,
    /// Graph storage (either standard or compact).
    pub storage: GraphStorage,
    /// The storage section data from the FAISS file (IndexFlat data or null).
    pub vector_storage: VectorStorage,
}

/// Graph adjacency storage.
pub enum GraphStorage {
    /// Standard FAISS format: offsets[ntotal+1] + neighbors[total_links].
    Standard {
        offsets: Vec<u64>,
        neighbors: Vec<i32>,
    },
    /// Compact CSR format.
    Compact {
        /// Pointers into the neighbors array for each level of each node.
        level_ptr: Vec<u64>,
        /// Offsets into level_ptr for each node.
        node_offsets: Vec<u64>,
        /// Compact neighbor data (only valid neighbors, no -1 padding).
        neighbors: Vec<i32>,
    },
}

/// Vector storage section from the FAISS index file.
pub enum VectorStorage {
    /// No vector storage (pruned/recompute mode). FourCC = "null".
    Null,
    /// Raw vector storage bytes with its FourCC identifier.
    Raw { fourcc: u32, data: Vec<u8> },
}

// FourCC constants matching the FAISS format.
pub const FOURCC_HNSW_FLAT: u32 = u32::from_le_bytes(*b"IHNf");
pub const FOURCC_NULL: u32 = u32::from_le_bytes(*b"null");

impl HnswGraph {
    /// Get the number of neighbors at a given level using cumulative neighbor counts.
    pub fn neighbors_at_level(&self, level: usize) -> usize {
        if level == 0 {
            if self.cum_nneighbor_per_level.is_empty() {
                return 0;
            }
            self.cum_nneighbor_per_level[0] as usize
        } else if level < self.cum_nneighbor_per_level.len() {
            (self.cum_nneighbor_per_level[level] - self.cum_nneighbor_per_level[level - 1]) as usize
        } else {
            0
        }
    }

    /// Get neighbors of a node at a specific level (standard format).
    pub fn get_neighbors(&self, node: usize, level: usize) -> &[i32] {
        match &self.storage {
            GraphStorage::Standard { offsets, neighbors } => {
                let offset = offsets[node] as usize;
                let cum_begin = if level == 0 {
                    0
                } else {
                    self.cum_nneighbor_per_level[level - 1] as usize
                };
                let cum_end = if level < self.cum_nneighbor_per_level.len() {
                    self.cum_nneighbor_per_level[level] as usize
                } else {
                    return &[];
                };
                let begin = offset + cum_begin;
                let end = offset + cum_end;
                if end <= neighbors.len() {
                    &neighbors[begin..end]
                } else {
                    &[]
                }
            }
            GraphStorage::Compact {
                level_ptr,
                node_offsets,
                neighbors,
            } => {
                let base = node_offsets[node] as usize;
                let ptr_idx = base + level;
                if ptr_idx + 1 >= level_ptr.len() {
                    return &[];
                }
                let begin = level_ptr[ptr_idx] as usize;
                let end = level_ptr[ptr_idx + 1] as usize;
                if end <= neighbors.len() {
                    &neighbors[begin..end]
                } else {
                    &[]
                }
            }
        }
    }

    /// Check if the graph uses compact storage.
    pub fn is_compact(&self) -> bool {
        matches!(self.storage, GraphStorage::Compact { .. })
    }

    /// Check if vector storage has been pruned.
    pub fn is_pruned(&self) -> bool {
        matches!(self.vector_storage, VectorStorage::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_config_default() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 32);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 64);
        assert_eq!(config.distance_metric, DistanceMetric::Mips);
    }

    #[test]
    fn test_fourcc_constants() {
        assert_eq!(FOURCC_HNSW_FLAT, u32::from_le_bytes(*b"IHNf"));
        assert_eq!(FOURCC_NULL, u32::from_le_bytes(*b"null"));
    }
}
