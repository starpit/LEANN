use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::graph::*;

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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
    let d = graph.dimensions;
    let ef = params.ef_search.max(top_k);

    let dist_fn: Box<dyn Fn(&[f32], &[f32]) -> f32> = match graph.config.distance_metric {
        crate::index::DistanceMetric::L2 => Box::new(|a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
        }),
        _ => Box::new(|a: &[f32], b: &[f32]| -> f32 {
            -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
        }),
    };

    // Phase 1: Greedy search from top level to level 1
    let mut curr = graph.entry_point as usize;
    for level in (1..=graph.max_level as usize).rev() {
        loop {
            let mut changed = false;
            let neighbors = graph.get_neighbors(curr, level);
            for &nb in neighbors {
                if nb < 0 {
                    continue;
                }
                let nb = nb as usize;
                let d_nb = dist_fn(query, &vectors[nb * d..(nb + 1) * d]);
                let d_curr = dist_fn(query, &vectors[curr * d..(curr + 1) * d]);
                if d_nb < d_curr {
                    curr = nb;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    // Phase 2: Search at level 0 with ef candidates
    let mut candidates = BinaryHeap::new(); // min-heap
    let mut results = BinaryHeap::new(); // max-heap (for eviction)
    let mut visited = vec![false; graph.ntotal];

    let d_entry = dist_fn(query, &vectors[curr * d..(curr + 1) * d]);
    candidates.push(SearchCandidate {
        distance: d_entry,
        id: curr,
    });
    results.push(MaxCandidate {
        distance: d_entry,
        id: curr,
    });
    visited[curr] = true;

    while let Some(cand) = candidates.pop() {
        // If candidate is worse than worst result and we have enough results, stop
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if cand.distance > worst.distance {
                    break;
                }
            }
        }

        let neighbors = graph.get_neighbors(cand.id, 0);
        for &nb in neighbors {
            if nb < 0 {
                continue;
            }
            let nb = nb as usize;
            if visited[nb] {
                continue;
            }
            visited[nb] = true;

            let d_nb = dist_fn(query, &vectors[nb * d..(nb + 1) * d]);

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

    // Collect and sort results
    let mut result_vec: Vec<(usize, f32)> = results
        .into_sorted_vec()
        .into_iter()
        .map(|c| (c.id, c.distance))
        .collect();

    result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    result_vec.truncate(top_k);

    let labels: Vec<usize> = result_vec.iter().map(|(id, _)| *id).collect();
    let distances: Vec<f32> = result_vec.iter().map(|(_, d)| *d).collect();

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

    // Phase 1: Greedy search from top level to level 1
    let mut curr = graph.entry_point as usize;
    for level in (1..=graph.max_level as usize).rev() {
        loop {
            let mut changed = false;
            let neighbors = graph.get_neighbors(curr, level);
            let valid_neighbors: Vec<usize> = neighbors
                .iter()
                .filter(|&&nb| nb >= 0)
                .map(|&nb| nb as usize)
                .collect();

            if valid_neighbors.is_empty() {
                break;
            }

            let mut all_nodes = valid_neighbors.clone();
            all_nodes.push(curr);
            let distances = compute_distance(&all_nodes, query);

            let curr_dist = *distances.last().unwrap();
            for (i, &nb) in valid_neighbors.iter().enumerate() {
                if distances[i] < curr_dist {
                    curr = nb;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    // Phase 2: ef-search at level 0
    let mut candidates = BinaryHeap::new();
    let mut results = BinaryHeap::new();
    let mut visited = vec![false; graph.ntotal];

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
    visited[curr] = true;

    while let Some(cand) = candidates.pop() {
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if cand.distance > worst.distance {
                    break;
                }
            }
        }

        let neighbors = graph.get_neighbors(cand.id, 0);
        let unvisited: Vec<usize> = neighbors
            .iter()
            .filter(|&&nb| nb >= 0)
            .map(|&nb| nb as usize)
            .filter(|&nb| {
                if visited[nb] {
                    false
                } else {
                    visited[nb] = true;
                    true
                }
            })
            .collect();

        if unvisited.is_empty() {
            continue;
        }

        let distances = compute_distance(&unvisited, query);

        for (i, &nb) in unvisited.iter().enumerate() {
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

    // Collect results
    let mut result_vec: Vec<(usize, f32)> = results
        .into_sorted_vec()
        .into_iter()
        .map(|c| (c.id, c.distance))
        .collect();

    result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    result_vec.truncate(top_k);

    let labels: Vec<usize> = result_vec.iter().map(|(id, _)| *id).collect();
    let distances: Vec<f32> = result_vec.iter().map(|(_, d)| *d).collect();

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
