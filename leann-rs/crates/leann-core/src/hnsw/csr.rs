use anyhow::Result;

use super::graph::*;

/// Convert a standard HNSW graph to compact CSR format.
///
/// This mirrors the Python `convert_hnsw_graph_to_csr()` function.
/// The compact format stores only valid (non-negative) neighbors,
/// removing the -1 padding used in the standard format.
pub fn convert_to_csr(graph: &HnswGraph) -> Result<HnswGraph> {
    let (offsets, neighbors) = match &graph.storage {
        GraphStorage::Standard { offsets, neighbors } => (offsets, neighbors),
        GraphStorage::Compact { .. } => {
            anyhow::bail!("Graph is already in compact format");
        }
    };

    let ntotal = graph.ntotal;
    let mut compact_neighbors: Vec<i32> = Vec::new();
    let mut compact_level_ptr: Vec<u64> = Vec::new();
    let mut compact_node_offsets: Vec<u64> = vec![0u64; ntotal + 1];

    let mut current_level_ptr_idx: u64 = 0;
    let mut current_data_idx: u64 = 0;

    for i in 0..ntotal {
        let node_max_level = graph.levels[i] - 1;
        let node_max_level = node_max_level.max(-1);

        let node_ptr_start_index = current_level_ptr_idx;
        compact_node_offsets[i] = node_ptr_start_index;

        let num_pointers_expected = (node_max_level + 1 + 1) as u64;
        let original_offset_start = offsets[i];

        for level in 0..=(node_max_level as usize) {
            compact_level_ptr.push(current_data_idx);

            let cum_begin = if level == 0 {
                0
            } else {
                graph.cum_nneighbor_per_level[level - 1] as u64
            };
            let cum_end = if level < graph.cum_nneighbor_per_level.len() {
                graph.cum_nneighbor_per_level[level] as u64
            } else {
                cum_begin
            };

            let begin = (original_offset_start + cum_begin) as usize;
            let end = (original_offset_start + cum_end) as usize;

            let begin = begin.min(neighbors.len());
            let end = end.min(neighbors.len()).max(begin);

            if begin < end {
                let level_neighbors = &neighbors[begin..end];
                for &nb in level_neighbors {
                    if nb >= 0 {
                        compact_neighbors.push(nb);
                        current_data_idx += 1;
                    }
                }
            }
        }

        // End pointer for last level
        compact_level_ptr.push(current_data_idx);
        current_level_ptr_idx += num_pointers_expected;
    }

    compact_node_offsets[ntotal] = current_level_ptr_idx;

    // Determine vector storage
    let vector_storage = match &graph.vector_storage {
        VectorStorage::Null => VectorStorage::Null,
        VectorStorage::Raw { .. } => VectorStorage::Null, // Prune by default during CSR conversion
    };

    Ok(HnswGraph {
        ntotal: graph.ntotal,
        dimensions: graph.dimensions,
        entry_point: graph.entry_point,
        max_level: graph.max_level,
        levels: graph.levels.clone(),
        assign_probas: graph.assign_probas.clone(),
        cum_nneighbor_per_level: graph.cum_nneighbor_per_level.clone(),
        config: HnswConfig {
            is_compact: true,
            ..graph.config.clone()
        },
        metric_type: graph.metric_type,
        metric_arg: graph.metric_arg,
        storage: GraphStorage::Compact {
            level_ptr: compact_level_ptr,
            node_offsets: compact_node_offsets,
            neighbors: compact_neighbors,
        },
        vector_storage,
    })
}

/// Prune vector storage from an HNSW graph (set storage to Null).
/// This is used for recompute mode where vectors are recomputed on the fly.
pub fn prune_embeddings(graph: &mut HnswGraph) {
    graph.vector_storage = VectorStorage::Null;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::build::build_hnsw;
    use ndarray::Array2;

    #[test]
    fn test_convert_to_csr() {
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5,
            ],
        )
        .unwrap();

        let config = HnswConfig {
            m: 4,
            ef_construction: 16,
            ef_search: 16,
            distance_metric: crate::index::DistanceMetric::L2,
            is_compact: false,
            is_recompute: false,
            seed: None,
        };

        let graph = build_hnsw(&data, &config).unwrap();
        assert!(!graph.is_compact());

        let csr_graph = convert_to_csr(&graph).unwrap();
        assert!(csr_graph.is_compact());
        assert_eq!(csr_graph.ntotal, 5);
        assert_eq!(csr_graph.dimensions, 3);

        // Verify all nodes have valid neighbors in the compact format
        if let GraphStorage::Compact { neighbors, .. } = &csr_graph.storage {
            for &nb in neighbors {
                assert!(nb >= 0, "Compact format should have no -1 entries");
            }
        }
    }
}
