use anyhow::Result;
use ndarray::Array2;
use rand::Rng;
use std::sync::atomic::{AtomicI32, Ordering as AtomicOrdering};

use rayon::prelude::*;

use super::graph::*;
use super::search::{FlatMaxHeap, FlatMinHeap};
use super::simd::{
    inner_product_distance, inner_product_distance_batch_4, l2_distance, l2_distance_batch_4,
    VisitedList,
};
use crate::index::DistanceMetric;

/// Build an HNSW graph from dense vectors (single-threaded).
pub fn build_hnsw(data: &Array2<f32>, config: &HnswConfig) -> Result<HnswGraph> {
    build_hnsw_serial(data, config)
}

/// Build an HNSW graph with the specified number of threads.
/// Creates a new rayon thread pool per call. If you're building multiple
/// indexes, use [`build_hnsw_with_pool`] to reuse a single pool.
/// Falls back to the serial path when `num_threads <= 1`.
pub fn build_hnsw_with_threads(
    data: &Array2<f32>,
    config: &HnswConfig,
    num_threads: usize,
) -> Result<HnswGraph> {
    if num_threads <= 1 {
        build_hnsw_serial(data, config)
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()?;
        build_hnsw_with_pool(data, config, &pool)
    }
}

/// Build an HNSW graph using an existing rayon thread pool.
/// Avoids the ~0.5-1ms cost of creating a new pool per call.
pub fn build_hnsw_with_pool(
    data: &Array2<f32>,
    config: &HnswConfig,
    pool: &rayon::ThreadPool,
) -> Result<HnswGraph> {
    build_hnsw_parallel(data, config, pool)
}

/// Serial HNSW build (original implementation).
fn build_hnsw_serial(data: &Array2<f32>, config: &HnswConfig) -> Result<HnswGraph> {
    // Dispatch on metric to monomorphize — inlines SIMD distance into build loops.
    match config.distance_metric {
        DistanceMetric::L2 => {
            build_hnsw_serial_inner(data, config, l2_distance, l2_distance_batch_4)
        }
        DistanceMetric::Mips | DistanceMetric::Cosine => build_hnsw_serial_inner(
            data,
            config,
            inner_product_distance,
            inner_product_distance_batch_4,
        ),
    }
}

/// Monomorphized serial build. The generic `D` and `B` parameters let the
/// compiler inline the distance functions into every call site.
fn build_hnsw_serial_inner<D, B>(
    data: &Array2<f32>,
    config: &HnswConfig,
    dist_fn: D,
    dist_batch_4: B,
) -> Result<HnswGraph>
where
    D: Fn(&[f32], &[f32]) -> f32,
    B: Fn(&[f32], &[f32], &[f32], &[f32], &[f32]) -> [f32; 4],
{
    let n = data.nrows();
    let d = data.ncols();

    if n == 0 {
        anyhow::bail!("Cannot build HNSW from empty data");
    }

    // Pre-flatten data to avoid per-call data.row().as_slice() overhead.
    let flat: Vec<f32> = data.iter().copied().collect();
    assert!(flat.len() >= n * d);

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

    // Allocate a single visited list, reused across all levels/nodes
    let mut visited = VisitedList::new(n);

    // Pre-allocate reusable flat heaps (matching FAISS's search_neighbors_to_add)
    let ef = config.ef_construction;
    let mut candidates = FlatMinHeap::new(ef * 2 * m);
    let mut results = FlatMaxHeap::new(ef);

    // Scratch buffer for batching 4 unvisited neighbors
    let mut saved: [u32; 4] = [0; 4];

    for i in 0..n {
        let node_level = levels[i] - 1; // max level for this node

        if i == 0 {
            entry_point = 0;
            continue;
        }

        let query_slice = &flat[i * d..][..d];

        // Phase 1: Traverse from top level down to node_level+1 (greedy search to find entry point)
        let mut curr_entry = entry_point as usize;
        for level in (node_level as usize + 1..=max_level as usize).rev() {
            let mut d_curr = dist_fn(query_slice, &flat[curr_entry * d..][..d]);
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
                    let d_nb = dist_fn(query_slice, &flat[nb * d..][..d]);
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
        // search for ef_construction nearest neighbors, then connect.
        // Uses FAISS-style bounded max-heap + batch_4 distance.
        for level in (0..=node_level as usize).rev() {
            let max_neighbors = if level == 0 { 2 * m } else { m };

            // Reuse pre-allocated buffers (clear is O(1), keeps capacity)
            candidates.clear();
            results.clear();
            visited.reset();
            visited.set(i);

            let d_entry = dist_fn(query_slice, &flat[curr_entry * d..][..d]);
            candidates.push(d_entry, curr_entry as u32);
            results.push(d_entry, curr_entry as u32);
            visited.set(curr_entry);

            while !candidates.is_empty() {
                let (cand_dist, cand_id) = candidates.pop();

                // FAISS termination: stop when best candidate > worst result
                if results.len() >= ef && cand_dist > results.peek_max_dis() {
                    break;
                }

                // Explore neighbors with batch_4 distance
                let nb_slice = get_neighbors_mut_slice(
                    &neighbors,
                    &offsets,
                    &cum_nneighbor_per_level,
                    cand_id as usize,
                    level,
                );

                let mut counter = 0;
                for &nb in nb_slice {
                    if nb < 0 {
                        break;
                    }
                    if !visited.check_and_set(nb as usize) {
                        continue;
                    }

                    saved[counter] = nb as u32;
                    counter += 1;

                    if counter == 4 {
                        let dists = unsafe {
                            dist_batch_4(
                                query_slice,
                                &flat[saved[0] as usize * d..][..d],
                                &flat[saved[1] as usize * d..][..d],
                                &flat[saved[2] as usize * d..][..d],
                                &flat[saved[3] as usize * d..][..d],
                            )
                        };

                        for k in 0..4 {
                            let nb_id = saved[k];
                            let d_nb = dists[k];
                            if results.len() < ef {
                                candidates.push(d_nb, nb_id);
                                results.push(d_nb, nb_id);
                            } else if d_nb < results.peek_max_dis() {
                                candidates.push(d_nb, nb_id);
                                results.replace_max(d_nb, nb_id);
                            }
                        }
                        counter = 0;
                    }
                }

                // Process remainder (1-3 leftover neighbors)
                for k in 0..counter {
                    let nb_id = saved[k];
                    let d_nb = dist_fn(query_slice, &flat[nb_id as usize * d..][..d]);
                    if results.len() < ef {
                        candidates.push(d_nb, nb_id);
                        results.push(d_nb, nb_id);
                    } else if d_nb < results.peek_max_dis() {
                        candidates.push(d_nb, nb_id);
                        results.replace_max(d_nb, nb_id);
                    }
                }
            }

            // Extract top max_neighbors from the bounded results heap
            // Pop all from max-heap into a vec, sort, take top max_neighbors
            let mut result_vec: Vec<(f32, u32)> = Vec::with_capacity(results.len());
            while results.len() > 0 {
                let (d, id) = results.pop_max();
                result_vec.push((d, id));
            }
            result_vec.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            result_vec.truncate(max_neighbors);

            // Connect node i to its neighbors at this level
            let neighbors_range = get_neighbor_range(&offsets, &cum_nneighbor_per_level, i, level);
            for (slot, &(_, id)) in neighbors_range.zip(result_vec.iter()) {
                neighbors[slot] = id as i32;
            }

            // Add reverse connections
            for &(_, id) in &result_vec {
                add_reverse_connection(
                    &mut neighbors,
                    &offsets,
                    &cum_nneighbor_per_level,
                    id as usize,
                    i as i32,
                    level,
                    max_neighbors,
                    &flat,
                    d,
                    &dist_fn,
                );
            }

            if !result_vec.is_empty() {
                curr_entry = result_vec[0].1 as usize;
            }
        }

        // Update entry point if this node has a higher level
        if node_level > levels[entry_point as usize] - 1 {
            entry_point = i as i32;
        }
    }

    finalize_graph(
        data,
        config,
        entry_point,
        max_level,
        levels,
        cum_nneighbor_per_level,
        offsets,
        neighbors,
    )
}

/// Parallel HNSW build using rayon and lock-free atomics.
fn build_hnsw_parallel(
    data: &Array2<f32>,
    config: &HnswConfig,
    pool: &rayon::ThreadPool,
) -> Result<HnswGraph> {
    // Dispatch on metric to monomorphize — inlines SIMD distance into build loops.
    match config.distance_metric {
        DistanceMetric::L2 => {
            build_hnsw_parallel_inner(data, config, pool, l2_distance, l2_distance_batch_4)
        }
        DistanceMetric::Mips | DistanceMetric::Cosine => build_hnsw_parallel_inner(
            data,
            config,
            pool,
            inner_product_distance,
            inner_product_distance_batch_4,
        ),
    }
}

/// Monomorphized parallel build. Generic `D` and `B` ensure the distance
/// functions are inlined into the hot loops rather than called through
/// function pointers.
fn build_hnsw_parallel_inner<D, B>(
    data: &Array2<f32>,
    config: &HnswConfig,
    pool: &rayon::ThreadPool,
    dist_fn: D,
    dist_batch_4: B,
) -> Result<HnswGraph>
where
    D: Fn(&[f32], &[f32]) -> f32 + Sync,
    B: Fn(&[f32], &[f32], &[f32], &[f32], &[f32]) -> [f32; 4] + Sync,
{
    let n = data.nrows();
    let d = data.ncols();

    if n == 0 {
        anyhow::bail!("Cannot build HNSW from empty data");
    }

    let m = config.m;
    let ml = 1.0 / (m as f64).ln();
    let ef = config.ef_construction;

    // Pre-flatten data to avoid per-call data.row().as_slice() overhead.
    let flat: Vec<f32> = data.iter().copied().collect();
    assert!(flat.len() >= n * d);

    // Assign levels (sequential, fast)
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
        levels.push(level + 1);
    }

    let num_levels = (max_level + 1) as usize;
    let mut cum_nneighbor_per_level = Vec::with_capacity(num_levels);
    let mut cum = 0i32;
    for l in 0..num_levels {
        let nb_neighbors = if l == 0 { 2 * m } else { m };
        cum += nb_neighbors as i32;
        cum_nneighbor_per_level.push(cum);
    }

    // Build offsets (sequential, fast)
    let mut offsets = Vec::with_capacity(n + 1);
    let mut current_offset = 0u64;
    for i in 0..n {
        offsets.push(current_offset);
        let node_levels = levels[i] as usize;
        let node_neighbors = if node_levels == 0 {
            0
        } else {
            let idx = (node_levels - 1).min(cum_nneighbor_per_level.len() - 1);
            cum_nneighbor_per_level[idx] as usize
        };
        current_offset += node_neighbors as u64;
    }
    offsets.push(current_offset);

    let total_neighbor_slots = current_offset as usize;

    // Atomic neighbor array — each slot is independently CAS-able
    let neighbors: Vec<AtomicI32> = (0..total_neighbor_slots)
        .map(|_| AtomicI32::new(-1))
        .collect();

    // Atomic entry point (updated via CAS when a higher-level node appears)
    let entry_point = AtomicI32::new(0);

    pool.install(|| {
        let dist_ref = &dist_fn;
        let batch_ref = &dist_batch_4;
        (1..n).into_par_iter().for_each_init(
            // Per-thread buffer allocation (runs once per rayon worker)
            || {
                (
                    VisitedList::new(n),
                    FlatMinHeap::new(ef * 2 * m),
                    FlatMaxHeap::new(ef),
                )
            },
            |(visited, candidates, results), i| {
                let node_level = levels[i] - 1;

                let query_slice = &flat[i * d..][..d];

                // Phase 1: greedy descent from top level to node_level+1
                let mut curr_entry = entry_point.load(AtomicOrdering::Relaxed) as usize;
                for level in (node_level as usize + 1..=max_level as usize).rev() {
                    let mut d_curr = dist_ref(query_slice, &flat[curr_entry * d..][..d]);
                    loop {
                        let mut changed = false;
                        let nb_slice = get_neighbors_atomic_slice(
                            &neighbors,
                            &offsets,
                            &cum_nneighbor_per_level,
                            curr_entry,
                            level,
                        );
                        for atom in nb_slice {
                            let nb = atom.load(AtomicOrdering::Relaxed);
                            if nb < 0 {
                                continue;
                            }
                            let nb = nb as usize;
                            let d_nb = dist_ref(query_slice, &flat[nb * d..][..d]);
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

                // Phase 2: search & connect at each level from node_level down to 0
                // Uses FAISS-style bounded max-heap + batch_4 distance.
                let mut saved: [u32; 4] = [0; 4];

                for level in (0..=node_level as usize).rev() {
                    let max_neighbors = if level == 0 { 2 * m } else { m };

                    candidates.clear();
                    results.clear();
                    visited.reset();
                    visited.set(i);

                    let d_entry = dist_ref(query_slice, &flat[curr_entry * d..][..d]);
                    candidates.push(d_entry, curr_entry as u32);
                    results.push(d_entry, curr_entry as u32);
                    visited.set(curr_entry);

                    while !candidates.is_empty() {
                        let (cand_dist, cand_id) = candidates.pop();

                        if results.len() >= ef && cand_dist > results.peek_max_dis() {
                            break;
                        }

                        let nb_slice = get_neighbors_atomic_slice(
                            &neighbors,
                            &offsets,
                            &cum_nneighbor_per_level,
                            cand_id as usize,
                            level,
                        );

                        let mut counter = 0;
                        for atom in nb_slice {
                            let nb = atom.load(AtomicOrdering::Relaxed);
                            if nb < 0 {
                                break;
                            }
                            if !visited.check_and_set(nb as usize) {
                                continue;
                            }

                            saved[counter] = nb as u32;
                            counter += 1;

                            if counter == 4 {
                                let dists = unsafe {
                                    batch_ref(
                                        query_slice,
                                        &flat[saved[0] as usize * d..][..d],
                                        &flat[saved[1] as usize * d..][..d],
                                        &flat[saved[2] as usize * d..][..d],
                                        &flat[saved[3] as usize * d..][..d],
                                    )
                                };

                                for k in 0..4 {
                                    let nb_id = saved[k];
                                    let d_nb = dists[k];
                                    if results.len() < ef {
                                        candidates.push(d_nb, nb_id);
                                        results.push(d_nb, nb_id);
                                    } else if d_nb < results.peek_max_dis() {
                                        candidates.push(d_nb, nb_id);
                                        results.replace_max(d_nb, nb_id);
                                    }
                                }
                                counter = 0;
                            }
                        }

                        // Process remainder (1-3 leftover neighbors)
                        for k in 0..counter {
                            let nb_id = saved[k];
                            let d_nb =
                                dist_ref(query_slice, &flat[nb_id as usize * d..][..d]);
                            if results.len() < ef {
                                candidates.push(d_nb, nb_id);
                                results.push(d_nb, nb_id);
                            } else if d_nb < results.peek_max_dis() {
                                candidates.push(d_nb, nb_id);
                                results.replace_max(d_nb, nb_id);
                            }
                        }
                    }

                    // Extract top max_neighbors from bounded results heap
                    let mut result_vec: Vec<(f32, u32)> =
                        Vec::with_capacity(results.len());
                    while results.len() > 0 {
                        let (dd, id) = results.pop_max();
                        result_vec.push((dd, id));
                    }
                    result_vec.sort_unstable_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    result_vec.truncate(max_neighbors);

                    // Forward connections: only this thread writes to node i's slots
                    let range =
                        get_neighbor_range(&offsets, &cum_nneighbor_per_level, i, level);
                    for (slot, &(_, id)) in range.zip(result_vec.iter()) {
                        neighbors[slot].store(id as i32, AtomicOrdering::Relaxed);
                    }

                    // Reverse connections via CAS
                    for &(_, id) in result_vec.iter() {
                        add_reverse_connection_atomic(
                            &neighbors,
                            &offsets,
                            &cum_nneighbor_per_level,
                            id as usize,
                            i as i32,
                            level,
                            &flat,
                            d,
                            dist_ref,
                        );
                    }

                    if !result_vec.is_empty() {
                        curr_entry = result_vec[0].1 as usize;
                    }
                }

                // CAS-update entry point if this node has a higher level
                loop {
                    let ep = entry_point.load(AtomicOrdering::Relaxed);
                    if node_level <= levels[ep as usize] - 1 {
                        break;
                    }
                    if entry_point
                        .compare_exchange_weak(
                            ep,
                            i as i32,
                            AtomicOrdering::Relaxed,
                            AtomicOrdering::Relaxed,
                        )
                        .is_ok()
                    {
                        break;
                    }
                }
            },
        );
    });

    // Convert AtomicI32 → i32 (zero-cost: same layout, into_inner consumes)
    let neighbors_i32: Vec<i32> = neighbors.into_iter().map(|a| a.into_inner()).collect();
    let final_entry_point = entry_point.into_inner();

    finalize_graph(
        data,
        config,
        final_entry_point,
        max_level,
        levels,
        cum_nneighbor_per_level,
        offsets,
        neighbors_i32,
    )
}

/// Shared graph finalization: build assign_probas and construct HnswGraph.
fn finalize_graph(
    data: &Array2<f32>,
    config: &HnswConfig,
    entry_point: i32,
    max_level: i32,
    levels: Vec<i32>,
    cum_nneighbor_per_level: Vec<i32>,
    offsets: Vec<u64>,
    neighbors: Vec<i32>,
) -> Result<HnswGraph> {
    let n = data.nrows();
    let d = data.ncols();
    let m = config.m;
    let ml = 1.0 / (m as f64).ln();
    let num_levels = (max_level + 1) as usize;

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
        vector_storage: VectorStorage::Null,
    };

    Ok(graph)
}

// ---------------------------------------------------------------------------
// Helper functions for neighbor access
// ---------------------------------------------------------------------------

#[inline(always)]
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

#[inline(always)]
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

/// Read a neighbor slice from atomic storage (parallel path).
#[inline(always)]
fn get_neighbors_atomic_slice<'a>(
    neighbors: &'a [AtomicI32],
    offsets: &[u64],
    cum_nn: &[i32],
    node: usize,
    level: usize,
) -> &'a [AtomicI32] {
    let range = get_neighbor_range(offsets, cum_nn, node, level);
    if range.end <= neighbors.len() {
        &neighbors[range]
    } else {
        &[]
    }
}

fn add_reverse_connection<D: Fn(&[f32], &[f32]) -> f32>(
    neighbors: &mut [i32],
    offsets: &[u64],
    cum_nn: &[i32],
    target: usize,
    source: i32,
    level: usize,
    _max_neighbors: usize,
    flat: &[f32],
    dim: usize,
    dist_fn: &D,
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
    let target_slice = &flat[target * dim..][..dim];
    let source_dist = dist_fn(target_slice, &flat[source as usize * dim..][..dim]);

    let mut worst_idx = range.start;
    let mut worst_dist = f32::NEG_INFINITY;

    for idx in range {
        let nb = neighbors[idx];
        if nb < 0 {
            continue;
        }
        let d = dist_fn(target_slice, &flat[nb as usize * dim..][..dim]);
        if d > worst_dist {
            worst_dist = d;
            worst_idx = idx;
        }
    }

    if source_dist < worst_dist {
        neighbors[worst_idx] = source;
    }
}

/// Lock-free reverse connection using CAS (parallel path).
/// Empty slot: CAS(-1, source). Replace worst: single CAS attempt — if lost, give up.
/// HNSW is robust to slight suboptimality from lost races.
fn add_reverse_connection_atomic<D: Fn(&[f32], &[f32]) -> f32>(
    neighbors: &[AtomicI32],
    offsets: &[u64],
    cum_nn: &[i32],
    target: usize,
    source: i32,
    level: usize,
    flat: &[f32],
    dim: usize,
    dist_fn: &D,
) {
    let range = get_neighbor_range(offsets, cum_nn, target, level);
    if range.end > neighbors.len() {
        return;
    }

    // Try to claim an empty slot via CAS
    for idx in range.clone() {
        if neighbors[idx]
            .compare_exchange(-1, source, AtomicOrdering::Relaxed, AtomicOrdering::Relaxed)
            .is_ok()
        {
            return;
        }
    }

    // All slots occupied — try to replace the worst neighbor
    let target_slice = &flat[target * dim..][..dim];
    let source_dist = dist_fn(target_slice, &flat[source as usize * dim..][..dim]);

    let mut worst_idx = range.start;
    let mut worst_dist = f32::NEG_INFINITY;
    let mut worst_val = -1i32;

    for idx in range {
        let nb = neighbors[idx].load(AtomicOrdering::Relaxed);
        if nb < 0 {
            // Slot freed by another thread — try to claim it
            if neighbors[idx]
                .compare_exchange(-1, source, AtomicOrdering::Relaxed, AtomicOrdering::Relaxed)
                .is_ok()
            {
                return;
            }
            continue;
        }
        let d = dist_fn(target_slice, &flat[nb as usize * dim..][..dim]);
        if d > worst_dist {
            worst_dist = d;
            worst_idx = idx;
            worst_val = nb;
        }
    }

    if source_dist < worst_dist && worst_val >= 0 {
        // Single CAS attempt — if another thread changed the slot, give up
        let _ = neighbors[worst_idx].compare_exchange(
            worst_val,
            source,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        );
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

    #[test]
    fn test_build_parallel_small_graph() {
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

        let graph = build_hnsw_with_threads(&data, &config, 2).unwrap();
        assert_eq!(graph.ntotal, 5);
        assert_eq!(graph.dimensions, 4);
        assert!(graph.entry_point >= 0);

        // Verify neighbors are populated (at least some non-negative entries)
        if let GraphStorage::Standard { neighbors, .. } = &graph.storage {
            let connected = neighbors.iter().filter(|&&n| n >= 0).count();
            assert!(connected > 0, "Graph should have some connections");
        } else {
            panic!("Expected Standard storage");
        }
    }

    #[test]
    fn test_parallel_larger_graph() {
        // 100 random vectors in 16 dimensions
        let mut rng = rand::thread_rng();
        let n = 100;
        let d = 16;
        let data_vec: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
        let data = Array2::from_shape_vec((n, d), data_vec).unwrap();

        let config = HnswConfig {
            m: 8,
            ef_construction: 32,
            ef_search: 32,
            distance_metric: DistanceMetric::L2,
            is_compact: false,
            is_recompute: false,
        };

        let graph = build_hnsw_with_threads(&data, &config, 4).unwrap();
        assert_eq!(graph.ntotal, n);
        assert_eq!(graph.dimensions, d);
        assert!(graph.entry_point >= 0);
        assert!((graph.entry_point as usize) < n);
    }
}
