use anyhow::Result;
use ndarray::Array2;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::graph::*;
use super::simd::{inner_product_distance, l2_distance, VisitedList};
use crate::index::DistanceMetric;

/// A candidate neighbor during search/build.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    distance: f32,
    id: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse the comparison so smallest distance is popped first
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Select the appropriate distance function based on metric.
#[inline]
fn select_dist_fn(metric: DistanceMetric) -> fn(&[f32], &[f32]) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_distance,
        DistanceMetric::Mips | DistanceMetric::Cosine => inner_product_distance,
    }
}

/// Build an HNSW graph from dense vectors.
pub fn build_hnsw(data: &Array2<f32>, config: &HnswConfig) -> Result<HnswGraph> {
    let n = data.nrows();
    let d = data.ncols();

    if n == 0 {
        anyhow::bail!("Cannot build HNSW from empty data");
    }

    // Compute level assignment probabilities
    let m = config.m;
    let ml = 1.0 / (m as f64).ln();

    // Assign levels
    let mut rng = rand::thread_rng();
    let mut levels = Vec::with_capacity(n);
    let mut max_level: i32 = 0;

    for _ in 0..n {
        let r: f64 = rng.gen::<f64>();
        let level = (-r.ln() * ml).floor() as i32;
        let level = level.max(0);
        if level > max_level {
            max_level = level;
        }
        levels.push(level + 1); // FAISS convention: levels[i] = max_level + 1
    }

    // Build cumulative neighbor counts per level
    // Level 0 has 2*M neighbors, upper levels have M neighbors
    let num_levels = (max_level + 1) as usize;
    let mut cum_nneighbor_per_level = Vec::with_capacity(num_levels);
    let mut cum = 0i32;
    for l in 0..num_levels {
        let nb_neighbors = if l == 0 { 2 * m } else { m };
        cum += nb_neighbors as i32;
        cum_nneighbor_per_level.push(cum);
    }

    // Total neighbor slots per node
    let _total_neighbors_per_node = cum as usize;

    // Build offsets and allocate neighbors
    let mut offsets = Vec::with_capacity(n + 1);
    let mut current_offset = 0u64;
    for i in 0..n {
        offsets.push(current_offset);
        let node_levels = levels[i] as usize;
        let node_neighbors = if node_levels == 0 {
            0
        } else {
            // Use cumulative count up to node's max level
            let idx = (node_levels - 1).min(cum_nneighbor_per_level.len() - 1);
            cum_nneighbor_per_level[idx] as usize
        };
        current_offset += node_neighbors as u64;
    }
    offsets.push(current_offset);

    let total_neighbor_slots = current_offset as usize;
    let mut neighbors = vec![-1i32; total_neighbor_slots];

    // Insert nodes one by one
    let mut entry_point: i32 = 0;

    // Select SIMD-accelerated distance function
    let dist_fn = select_dist_fn(config.distance_metric);

    // Allocate a single visited list, reused across all levels/nodes
    let mut visited = VisitedList::new(n);

    // Pre-allocate reusable buffers outside the hot loop to avoid repeated alloc/realloc.
    // ef_construction bounds the result size; candidates can grow larger during search.
    let ef = config.ef_construction;
    let mut candidates = BinaryHeap::with_capacity(ef * 2 * m);
    let mut result: Vec<Candidate> = Vec::with_capacity(ef);

    for i in 0..n {
        let node_level = levels[i] - 1; // max level for this node

        if i == 0 {
            entry_point = 0;
            continue;
        }

        let query = data.row(i);
        let query_slice = query.as_slice().unwrap();

        // Phase 1: Traverse from top level down to node_level+1 (greedy search to find entry point)
        let mut curr_entry = entry_point as usize;
        for level in (node_level as usize + 1..=max_level as usize).rev() {
            let mut d_curr = dist_fn(query_slice, data.row(curr_entry).as_slice().unwrap());
            loop {
                let mut changed = false;
                let neighbor_slice = get_neighbors_mut_slice(
                    &neighbors,
                    &offsets,
                    &cum_nneighbor_per_level,
                    curr_entry,
                    level,
                );
                for &nb in neighbor_slice {
                    if nb < 0 {
                        continue;
                    }
                    let nb = nb as usize;
                    let d_nb = dist_fn(query_slice, data.row(nb).as_slice().unwrap());
                    if d_nb < d_curr {
                        curr_entry = nb;
                        d_curr = d_nb;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Phase 2: For each level from min(node_level, max_level) down to 0,
        // do greedy search to find ef_construction nearest neighbors, then connect
        for level in (0..=node_level as usize).rev() {
            let max_neighbors = if level == 0 { 2 * m } else { m };

            // Reuse pre-allocated buffers (clear is O(1) for len, keeps capacity)
            candidates.clear();
            result.clear();
            visited.reset();
            visited.set(i);

            let d_entry = dist_fn(query_slice, data.row(curr_entry).as_slice().unwrap());
            candidates.push(Candidate {
                distance: d_entry,
                id: curr_entry,
            });
            visited.set(curr_entry);

            while let Some(cand) = candidates.pop() {
                // If we have enough results and the candidate is worse than the worst result, stop
                if result.len() >= ef {
                    if let Some(worst) = result.last() {
                        if cand.distance > worst.distance {
                            break;
                        }
                    }
                }

                result.push(cand);

                // Explore neighbors
                let nb_slice = get_neighbors_mut_slice(
                    &neighbors,
                    &offsets,
                    &cum_nneighbor_per_level,
                    cand.id,
                    level,
                );
                for &nb in nb_slice {
                    if nb < 0 {
                        continue;
                    }
                    let nb = nb as usize;
                    if visited.is_visited(nb) {
                        continue;
                    }
                    visited.set(nb);
                    let d_nb = dist_fn(query_slice, data.row(nb).as_slice().unwrap());
                    candidates.push(Candidate {
                        distance: d_nb,
                        id: nb,
                    });
                }
            }

            // Sort results by distance and take top max_neighbors
            result.sort_unstable_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
            result.truncate(max_neighbors);

            // Connect node i to its neighbors at this level
            let neighbors_range = get_neighbor_range(&offsets, &cum_nneighbor_per_level, i, level);
            for (slot, cand) in neighbors_range.zip(result.iter()) {
                neighbors[slot] = cand.id as i32;
            }

            // Add reverse connections
            for cand in &result {
                add_reverse_connection(
                    &mut neighbors,
                    &offsets,
                    &cum_nneighbor_per_level,
                    cand.id,
                    i as i32,
                    level,
                    max_neighbors,
                    &data,
                    dist_fn,
                );
            }

            if !result.is_empty() {
                curr_entry = result[0].id;
            }
        }

        // Update entry point if this node has a higher level
        if node_level > levels[entry_point as usize] - 1 {
            entry_point = i as i32;
        }
    }

    // Build assign_probas
    let mut assign_probas = Vec::with_capacity(num_levels);
    for l in 0..num_levels {
        let p = if l == 0 {
            1.0 - (-1.0 / ml).exp()
        } else {
            (-((l as f64) / ml)).exp() - (-(((l + 1) as f64) / ml)).exp()
        };
        assign_probas.push(p);
    }

    let graph = HnswGraph {
        ntotal: n,
        dimensions: d,
        entry_point,
        max_level,
        levels,
        assign_probas,
        cum_nneighbor_per_level,
        config: config.clone(),
        metric_type: match config.distance_metric {
            DistanceMetric::L2 => 0,
            DistanceMetric::Mips | DistanceMetric::Cosine => 1,
        },
        metric_arg: 0.0,
        storage: GraphStorage::Standard { offsets, neighbors },
        vector_storage: VectorStorage::Null, // Will be set by caller if needed
    };

    Ok(graph)
}

// Helper functions for neighbor access during build

fn get_neighbor_range(
    offsets: &[u64],
    cum_nn: &[i32],
    node: usize,
    level: usize,
) -> std::ops::Range<usize> {
    let offset = offsets[node] as usize;
    let begin = if level == 0 {
        0
    } else {
        cum_nn[level - 1] as usize
    };
    let end = cum_nn[level] as usize;
    (offset + begin)..(offset + end)
}

fn get_neighbors_mut_slice<'a>(
    neighbors: &'a [i32],
    offsets: &[u64],
    cum_nn: &[i32],
    node: usize,
    level: usize,
) -> &'a [i32] {
    let range = get_neighbor_range(offsets, cum_nn, node, level);
    if range.end <= neighbors.len() {
        &neighbors[range]
    } else {
        &[]
    }
}

fn add_reverse_connection(
    neighbors: &mut [i32],
    offsets: &[u64],
    cum_nn: &[i32],
    target: usize,
    source: i32,
    level: usize,
    max_neighbors: usize,
    data: &Array2<f32>,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) {
    let range = get_neighbor_range(offsets, cum_nn, target, level);
    if range.end > neighbors.len() {
        return;
    }

    // Find an empty slot
    for idx in range.clone() {
        if neighbors[idx] < 0 {
            neighbors[idx] = source;
            return;
        }
    }

    // All slots full - replace the worst neighbor if the new one is better
    let target_vec = data.row(target);
    let target_slice = target_vec.as_slice().unwrap();
    let source_dist = dist_fn(target_slice, data.row(source as usize).as_slice().unwrap());

    let mut worst_idx = range.start;
    let mut worst_dist = f32::NEG_INFINITY;

    for idx in range {
        let nb = neighbors[idx];
        if nb < 0 {
            continue;
        }
        let d = dist_fn(target_slice, data.row(nb as usize).as_slice().unwrap());
        if d > worst_dist {
            worst_dist = d;
            worst_idx = idx;
        }
    }

    if source_dist < worst_dist {
        neighbors[worst_idx] = source;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_build_small_graph() {
        let data = Array2::from_shape_vec(
            (5, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                0.5, 0.5, 0.0, 0.0,
            ],
        )
        .unwrap();

        let config = HnswConfig {
            m: 4,
            ef_construction: 16,
            ef_search: 16,
            distance_metric: DistanceMetric::L2,
            is_compact: false,
            is_recompute: false,
        };

        let graph = build_hnsw(&data, &config).unwrap();
        assert_eq!(graph.ntotal, 5);
        assert_eq!(graph.dimensions, 4);
        assert!(graph.entry_point >= 0);
    }
}
