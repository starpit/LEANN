use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::graph::*;
use super::simd::{inner_product_distance, l2_distance, VisitedList};

/// Search parameters for HNSW search.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Size of the dynamic candidate list during search (efSearch).
    pub ef_search: usize,
    /// Number of parallel search paths / beam size.
    pub beam_size: usize,
    /// Ratio of neighbors to prune via approximate distance (0.0 - 1.0).
    pub prune_ratio: f64,
    /// Whether to recompute embeddings via ZMQ server.
    pub recompute_embeddings: bool,
    /// Pruning strategy: "global", "local", or "proportional".
    pub pruning_strategy: PruningStrategy,
    /// ZMQ port for embedding server communication.
    pub zmq_port: Option<u16>,
    /// Batch size for neighbor processing (0 = disabled).
    pub batch_size: usize,
    /// Whether to check relative distance for early termination.
    pub check_relative_distance: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningStrategy {
    Global,
    Local,
    Proportional,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            ef_search: 64,
            beam_size: 1,
            prune_ratio: 0.0,
            recompute_embeddings: true,
            pruning_strategy: PruningStrategy::Global,
            zmq_port: None,
            batch_size: 0,
            check_relative_distance: true,
        }
    }
}

/// A neighbor candidate with distance.
#[derive(Debug, Clone, Copy)]
struct SearchCandidate {
    distance: f32,
    id: usize,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}
impl Eq for SearchCandidate {}

// Min-heap ordering (smallest distance first)
impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap ordering for result set (largest distance first, so we can evict worst).
#[derive(Debug, Clone, Copy)]
struct MaxCandidate {
    distance: f32,
    id: usize,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}
impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Result of an HNSW search: lists of labels and distances for each query.
#[derive(Debug)]
pub struct SearchResults {
    /// labels[batch][k] = node ID of k-th nearest neighbor for query batch
    pub labels: Vec<Vec<usize>>,
    /// distances[batch][k] = distance of k-th nearest neighbor
    pub distances: Vec<Vec<f32>>,
}

/// Pre-allocated buffers for HNSW search. Reuse across multiple queries
/// to avoid repeated O(n) allocation of the visited list.
pub struct SearchBuffers {
    visited: VisitedList,
}

impl SearchBuffers {
    /// Create new search buffers for a graph with `ntotal` nodes.
    pub fn new(ntotal: usize) -> Self {
        Self {
            visited: VisitedList::new(ntotal),
        }
    }
}

/// Search the HNSW graph for nearest neighbors using stored vectors.
///
/// This function searches when all vectors are available in memory (non-recompute mode).
pub fn search_hnsw(
    graph: &HnswGraph,
    query: &[f32],
    top_k: usize,
    vectors: &[f32], // flat array of all indexed vectors, row-major [ntotal * d]
    params: &SearchParams,
) -> (Vec<usize>, Vec<f32>) {
    let mut buffers = SearchBuffers::new(graph.ntotal);
    search_hnsw_buf(graph, query, top_k, vectors, params, &mut buffers)
}

/// Search the HNSW graph, reusing pre-allocated buffers.
/// Use this when searching the same graph multiple times to avoid
/// O(n) VisitedList allocation per call.
pub fn search_hnsw_buf(
    graph: &HnswGraph,
    query: &[f32],
    top_k: usize,
    vectors: &[f32],
    params: &SearchParams,
    buffers: &mut SearchBuffers,
) -> (Vec<usize>, Vec<f32>) {
    // Dispatch on metric to monomorphize — allows compiler to inline
    // the SIMD distance function into the search loop.
    match graph.config.distance_metric {
        crate::index::DistanceMetric::L2 => search_hnsw_inner(
            graph,
            query,
            top_k,
            vectors,
            params,
            &mut buffers.visited,
            l2_distance,
        ),
        _ => search_hnsw_inner(
            graph,
            query,
            top_k,
            vectors,
            params,
            &mut buffers.visited,
            inner_product_distance,
        ),
    }
}

/// Get a vector slice without bounds checking.
///
/// # Safety
/// Caller must ensure `id * dim + dim <= vectors.len()`.
#[inline(always)]
unsafe fn get_vec(vectors: &[f32], id: usize, dim: usize) -> &[f32] {
    debug_assert!(id * dim + dim <= vectors.len());
    vectors.get_unchecked(id * dim..id * dim + dim)
}

/// Monomorphized search implementation. The generic `D` parameter ensures
/// the distance function is inlined into the loop body rather than called
/// through an indirect function pointer.
fn search_hnsw_inner<D: Fn(&[f32], &[f32]) -> f32>(
    graph: &HnswGraph,
    query: &[f32],
    top_k: usize,
    vectors: &[f32],
    params: &SearchParams,
    visited: &mut VisitedList,
    dist_fn: D,
) -> (Vec<usize>, Vec<f32>) {
    let d = graph.dimensions;
    let ef = params.ef_search.max(top_k);

    // Safety invariants: vectors and visited list are large enough for all node IDs.
    assert!(vectors.len() >= graph.ntotal * d);
    assert!(visited.len() >= graph.ntotal);

    // Phase 1: Greedy search from top level to level 1
    let mut curr = graph.entry_point as usize;
    let mut d_curr = unsafe { dist_fn(query, get_vec(vectors, curr, d)) };
    for level in (1..=graph.max_level as usize).rev() {
        loop {
            let mut changed = false;
            let neighbors = graph.get_neighbors(curr, level);
            for &nb in neighbors {
                if nb < 0 {
                    break; // valid neighbors are packed at front
                }
                let nb = nb as usize;
                let d_nb = unsafe { dist_fn(query, get_vec(vectors, nb, d)) };
                if d_nb < d_curr {
                    curr = nb;
                    d_curr = d_nb;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    // Phase 2: Search at level 0 with ef candidates
    let mut candidates = BinaryHeap::with_capacity(ef * 2); // min-heap
    let mut results = BinaryHeap::with_capacity(ef); // max-heap (for eviction)
    visited.reset();

    let d_entry = unsafe { dist_fn(query, get_vec(vectors, curr, d)) };
    candidates.push(SearchCandidate {
        distance: d_entry,
        id: curr,
    });
    results.push(MaxCandidate {
        distance: d_entry,
        id: curr,
    });
    visited.set(curr);

    // Cache worst distance to avoid heap peek on every neighbor check.
    let mut worst_dist = d_entry;

    while let Some(cand) = candidates.pop() {
        // If candidate is worse than worst result and we have enough results, stop
        if results.len() >= ef && cand.distance > worst_dist {
            break;
        }

        let neighbors = graph.get_neighbors(cand.id, 0);
        for &nb in neighbors {
            if nb < 0 {
                break; // valid neighbors are packed at front
            }
            let nb = nb as usize;
            if visited.is_visited(nb) {
                continue;
            }
            visited.set(nb);

            let d_nb = unsafe { dist_fn(query, get_vec(vectors, nb, d)) };

            // Add to results if better than worst, or if not full yet
            if results.len() < ef {
                candidates.push(SearchCandidate {
                    distance: d_nb,
                    id: nb,
                });
                results.push(MaxCandidate {
                    distance: d_nb,
                    id: nb,
                });
                if results.len() == ef {
                    worst_dist = results.peek().unwrap().distance;
                }
            } else if d_nb < worst_dist {
                candidates.push(SearchCandidate {
                    distance: d_nb,
                    id: nb,
                });
                results.pop();
                results.push(MaxCandidate {
                    distance: d_nb,
                    id: nb,
                });
                worst_dist = results.peek().unwrap().distance;
            }
        }
    }

    // Collect results — into_sorted_vec() returns ascending distance order
    let result_vec = results.into_sorted_vec();

    let mut labels = Vec::with_capacity(top_k);
    let mut distances = Vec::with_capacity(top_k);
    for c in result_vec.into_iter().take(top_k) {
        labels.push(c.id);
        distances.push(c.distance);
    }

    (labels, distances)
}

/// Search the HNSW graph using recomputed distances via a callback.
///
/// The `compute_distance` callback takes (node_ids, query) and returns distances.
/// This is used when vectors are not stored locally and must be recomputed on the fly.
pub fn search_hnsw_recompute<F>(
    graph: &HnswGraph,
    query: &[f32],
    top_k: usize,
    params: &SearchParams,
    mut compute_distance: F,
) -> (Vec<usize>, Vec<f32>)
where
    F: FnMut(&[usize], &[f32]) -> Vec<f32>,
{
    let ef = params.ef_search.max(top_k);
    let max_neighbors = graph.neighbors_at_level(0);

    // Pre-allocate scratch buffers reused across iterations
    let mut node_buf = Vec::with_capacity(max_neighbors + 1);

    // Phase 1: Greedy search from top level to level 1
    let mut curr = graph.entry_point as usize;
    for level in (1..=graph.max_level as usize).rev() {
        loop {
            let mut changed = false;
            let neighbors = graph.get_neighbors(curr, level);

            node_buf.clear();
            for &nb in neighbors {
                if nb >= 0 {
                    node_buf.push(nb as usize);
                }
            }

            if node_buf.is_empty() {
                break;
            }

            node_buf.push(curr);
            let distances = compute_distance(&node_buf, query);

            let curr_dist = *distances.last().unwrap();
            let valid_count = node_buf.len() - 1;
            for i in 0..valid_count {
                if distances[i] < curr_dist {
                    curr = node_buf[i];
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    // Phase 2: ef-search at level 0
    let mut candidates = BinaryHeap::with_capacity(ef * 2);
    let mut results = BinaryHeap::with_capacity(ef);
    let mut visited = VisitedList::new(graph.ntotal);
    visited.reset();

    let entry_dists = compute_distance(&[curr], query);
    let d_entry = entry_dists[0];
    candidates.push(SearchCandidate {
        distance: d_entry,
        id: curr,
    });
    results.push(MaxCandidate {
        distance: d_entry,
        id: curr,
    });
    visited.set(curr);

    while let Some(cand) = candidates.pop() {
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if cand.distance > worst.distance {
                    break;
                }
            }
        }

        let neighbors = graph.get_neighbors(cand.id, 0);

        node_buf.clear();
        for &nb in neighbors {
            if nb >= 0 {
                let nb = nb as usize;
                if !visited.is_visited(nb) {
                    visited.set(nb);
                    node_buf.push(nb);
                }
            }
        }

        if node_buf.is_empty() {
            continue;
        }

        let distances = compute_distance(&node_buf, query);

        for (i, &nb) in node_buf.iter().enumerate() {
            let d_nb = distances[i];

            if results.len() < ef {
                candidates.push(SearchCandidate {
                    distance: d_nb,
                    id: nb,
                });
                results.push(MaxCandidate {
                    distance: d_nb,
                    id: nb,
                });
            } else if let Some(worst) = results.peek() {
                if d_nb < worst.distance {
                    candidates.push(SearchCandidate {
                        distance: d_nb,
                        id: nb,
                    });
                    results.pop();
                    results.push(MaxCandidate {
                        distance: d_nb,
                        id: nb,
                    });
                }
            }
        }
    }

    // Collect results — into_sorted_vec() returns ascending distance order
    let result_vec = results.into_sorted_vec();

    let mut labels = Vec::with_capacity(top_k);
    let mut distances = Vec::with_capacity(top_k);
    for c in result_vec.into_iter().take(top_k) {
        labels.push(c.id);
        distances.push(c.distance);
    }

    (labels, distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::build::build_hnsw;
    use ndarray::Array2;

    #[test]
    fn test_search_with_stored_vectors() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0],
        )
        .unwrap();

        let config = HnswConfig {
            m: 4,
            ef_construction: 16,
            ef_search: 16,
            distance_metric: crate::index::DistanceMetric::L2,
            is_compact: false,
            is_recompute: false,
        };

        let graph = build_hnsw(&data, &config).unwrap();
        let flat_vectors: Vec<f32> = data.iter().copied().collect();

        let query = vec![1.0, 0.0, 0.0];
        let params = SearchParams {
            ef_search: 16,
            ..Default::default()
        };

        let (labels, distances) = search_hnsw(&graph, &query, 2, &flat_vectors, &params);
        assert_eq!(labels.len(), 2);
        assert_eq!(distances.len(), 2);
        // Node 0 should be the nearest to [1, 0, 0]
        assert_eq!(labels[0], 0);
        assert!((distances[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_with_recompute() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0],
        )
        .unwrap();

        let config = HnswConfig {
            m: 4,
            ef_construction: 16,
            ef_search: 16,
            distance_metric: crate::index::DistanceMetric::L2,
            is_compact: false,
            is_recompute: true,
        };

        let graph = build_hnsw(&data, &config).unwrap();
        let flat_vectors: Vec<f32> = data.iter().copied().collect();
        let d = 3;

        let query = vec![1.0, 0.0, 0.0];
        let params = SearchParams {
            ef_search: 16,
            recompute_embeddings: true,
            ..Default::default()
        };

        let (labels, distances) =
            search_hnsw_recompute(&graph, &query, 2, &params, |node_ids, q| {
                node_ids
                    .iter()
                    .map(|&id| {
                        let vec = &flat_vectors[id * d..(id + 1) * d];
                        vec.iter()
                            .zip(q.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum()
                    })
                    .collect()
            });

        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], 0);
    }
}
