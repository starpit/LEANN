use anyhow::Result;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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
    let flat_ptr = flat.as_ptr();

    // Compute level assignment probabilities
    let m = config.m;
    let ml = 1.0 / (m as f64).ln();

    // Assign levels — seed from config, or default deterministic seed
    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or(42));
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
    let mut candidates = FlatMinHeap::new(ef * 2);
    let mut results = FlatMaxHeap::new(ef + 1);

    // Scratch buffers
    let mut saved: [u32; 4] = [0; 4];
    let mut result_vec: Vec<(f32, u32)> = Vec::with_capacity(ef);
    let mut shrink_out: Vec<(f32, u32)> = Vec::with_capacity(2 * m);
    let mut link_scratch: Vec<(f32, u32)> = Vec::with_capacity(2 * m + 1);
    let mut link_shrink: Vec<(f32, u32)> = Vec::with_capacity(2 * m);

    for i in 0..n {
        let node_level = levels[i] - 1; // max level for this node

        if i == 0 {
            entry_point = 0;
            continue;
        }

        let query_slice = unsafe { get_flat(flat_ptr, i, d) };

        // Phase 1: Traverse from top level down to node_level+1 (greedy search to find entry point)
        let mut curr_entry = entry_point as usize;
        for level in (node_level as usize + 1..=max_level as usize).rev() {
            let mut d_curr = dist_fn(query_slice, unsafe { get_flat(flat_ptr, curr_entry, d) });
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
                    let d_nb = dist_fn(query_slice, unsafe { get_flat(flat_ptr, nb, d) });
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
        // FAISS-style: conditional candidate push + diversity pruning.
        for level in (0..=node_level as usize).rev() {
            let max_neighbors = if level == 0 { 2 * m } else { m };

            // Reuse pre-allocated buffers (clear is O(1), keeps capacity)
            candidates.clear();
            results.clear();
            visited.reset();
            visited.set(i);

            let d_entry = dist_fn(query_slice, unsafe { get_flat(flat_ptr, curr_entry, d) });
            candidates.push(d_entry, curr_entry as u32);
            results.push(d_entry, curr_entry as u32);
            visited.set(curr_entry);

            while !candidates.is_empty() {
                let (cand_dist, _cand_id) = candidates.peek();

                // FAISS termination: stop when best candidate > worst result
                if cand_dist > results.peek_max_dis() {
                    break;
                }
                let (_cand_dist, cand_id) = candidates.pop();

                // Explore neighbors with prefetch + batch_4 distance
                let nb_slice = get_neighbors_mut_slice(
                    &neighbors,
                    &offsets,
                    &cum_nneighbor_per_level,
                    cand_id as usize,
                    level,
                );

                // Pass 1: prefetch visited table entries
                for &nb in nb_slice {
                    if nb < 0 { break; }
                    visited.prefetch(nb as usize);
                }

                // Pass 2: check visited + batch distance
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
                                get_flat(flat_ptr, saved[0] as usize, d),
                                get_flat(flat_ptr, saved[1] as usize, d),
                                get_flat(flat_ptr, saved[2] as usize, d),
                                get_flat(flat_ptr, saved[3] as usize, d),
                            )
                        };

                        for k in 0..4 {
                            let nb_id = saved[k];
                            let d_nb = dists[k];
                            // FAISS-style: conditional push to both heaps
                            if results.len() < ef || d_nb < results.peek_max_dis() {
                                candidates.push(d_nb, nb_id);
                                results.push(d_nb, nb_id);
                                if results.len() > ef {
                                    results.pop_max();
                                }
                            }
                        }
                        counter = 0;
                    }
                }

                // Process remainder (1-3 leftover neighbors)
                for k in 0..counter {
                    let nb_id = saved[k];
                    let d_nb = dist_fn(query_slice, unsafe { get_flat(flat_ptr, nb_id as usize, d) });
                    if results.len() < ef || d_nb < results.peek_max_dis() {
                        candidates.push(d_nb, nb_id);
                        results.push(d_nb, nb_id);
                        if results.len() > ef {
                            results.pop_max();
                        }
                    }
                }
            }

            // Extract results sorted by ascending distance for shrink_neighbor_list
            result_vec.clear();
            while results.len() > 0 {
                let (dd, id) = results.pop_max();
                result_vec.push((dd, id));
            }
            result_vec.reverse(); // now ascending distance (closest first)

            // FAISS-style diversity pruning: select diverse neighbors
            shrink_neighbor_list(&result_vec, &mut shrink_out, max_neighbors, flat_ptr, d, &dist_fn);

            // Forward connections: write selected neighbors to node i's slots
            let fwd_range = get_neighbor_range(&offsets, &cum_nneighbor_per_level, i, level);
            for (slot, &(_, id)) in fwd_range.zip(shrink_out.iter()) {
                neighbors[slot] = id as i32;
            }

            // Reverse connections via add_link (with diversity pruning)
            for &(_, id) in shrink_out.iter() {
                add_link(
                    &mut neighbors,
                    &offsets,
                    &cum_nneighbor_per_level,
                    id as usize,
                    i as i32,
                    level,
                    flat_ptr,
                    d,
                    &dist_fn,
                    &mut link_scratch,
                    &mut link_shrink,
                );
            }

            if !shrink_out.is_empty() {
                curr_entry = shrink_out[0].1 as usize;
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
    // Store base address as usize so it's Send+Sync for rayon closures.
    // Safety: flat lives until after pool.install() returns.
    let flat_addr = flat.as_ptr() as usize;

    // Assign levels — seed from config, or default deterministic seed
    let mut rng = StdRng::seed_from_u64(config.seed.unwrap_or(42));
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
                    FlatMinHeap::new(ef * 2),
                    FlatMaxHeap::new(ef + 1),
                    Vec::<(f32, u32)>::with_capacity(ef),
                    Vec::<(f32, u32)>::with_capacity(2 * m),  // shrink_out
                    Vec::<(f32, u32)>::with_capacity(2 * m + 1),  // link_scratch
                    Vec::<(f32, u32)>::with_capacity(2 * m),  // link_shrink
                )
            },
            |(visited, candidates, results, result_vec, shrink_out, link_scratch, link_shrink), i| {
                let flat_ptr = flat_addr as *const f32;
                let node_level = levels[i] - 1;

                let query_slice = unsafe { get_flat(flat_ptr, i, d) };

                // Phase 1: greedy descent from top level to node_level+1
                let mut curr_entry = entry_point.load(AtomicOrdering::Relaxed) as usize;
                for level in (node_level as usize + 1..=max_level as usize).rev() {
                    let mut d_curr = dist_ref(query_slice, unsafe { get_flat(flat_ptr, curr_entry, d) });
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
                            let d_nb = dist_ref(query_slice, unsafe { get_flat(flat_ptr, nb, d) });
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
                // FAISS-style: conditional candidate push + diversity pruning.
                let mut saved: [u32; 4] = [0; 4];

                for level in (0..=node_level as usize).rev() {
                    let max_neighbors = if level == 0 { 2 * m } else { m };

                    candidates.clear();
                    results.clear();
                    visited.reset();
                    visited.set(i);

                    let d_entry = dist_ref(query_slice, unsafe { get_flat(flat_ptr, curr_entry, d) });
                    candidates.push(d_entry, curr_entry as u32);
                    results.push(d_entry, curr_entry as u32);
                    visited.set(curr_entry);

                    while !candidates.is_empty() {
                        let (cand_dist, _) = candidates.peek();

                        if cand_dist > results.peek_max_dis() {
                            break;
                        }
                        let (_cand_dist, cand_id) = candidates.pop();

                        let nb_slice = get_neighbors_atomic_slice(
                            &neighbors,
                            &offsets,
                            &cum_nneighbor_per_level,
                            cand_id as usize,
                            level,
                        );

                        // Pass 1: prefetch visited table entries
                        for atom in nb_slice {
                            let nb = atom.load(AtomicOrdering::Relaxed);
                            if nb < 0 { break; }
                            visited.prefetch(nb as usize);
                        }

                        // Pass 2: check visited + batch distance
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
                                        get_flat(flat_ptr, saved[0] as usize, d),
                                        get_flat(flat_ptr, saved[1] as usize, d),
                                        get_flat(flat_ptr, saved[2] as usize, d),
                                        get_flat(flat_ptr, saved[3] as usize, d),
                                    )
                                };

                                for k in 0..4 {
                                    let nb_id = saved[k];
                                    let d_nb = dists[k];
                                    if results.len() < ef || d_nb < results.peek_max_dis() {
                                        candidates.push(d_nb, nb_id);
                                        results.push(d_nb, nb_id);
                                        if results.len() > ef {
                                            results.pop_max();
                                        }
                                    }
                                }
                                counter = 0;
                            }
                        }

                        // Process remainder (1-3 leftover neighbors)
                        for k in 0..counter {
                            let nb_id = saved[k];
                            let d_nb =
                                dist_ref(query_slice, unsafe { get_flat(flat_ptr, nb_id as usize, d) });
                            if results.len() < ef || d_nb < results.peek_max_dis() {
                                candidates.push(d_nb, nb_id);
                                results.push(d_nb, nb_id);
                                if results.len() > ef {
                                    results.pop_max();
                                }
                            }
                        }
                    }

                    // Extract results sorted ascending for shrink_neighbor_list
                    result_vec.clear();
                    while results.len() > 0 {
                        let (dd, id) = results.pop_max();
                        result_vec.push((dd, id));
                    }
                    result_vec.reverse();

                    // FAISS-style diversity pruning
                    shrink_neighbor_list(result_vec, shrink_out, max_neighbors, flat_ptr, d, dist_ref);

                    // Forward connections: only this thread writes to node i's slots
                    let range =
                        get_neighbor_range(&offsets, &cum_nneighbor_per_level, i, level);
                    for (slot, &(_, id)) in range.zip(shrink_out.iter()) {
                        neighbors[slot].store(id as i32, AtomicOrdering::Relaxed);
                    }

                    // Reverse connections via add_link_atomic (with diversity pruning)
                    for &(_, id) in shrink_out.iter() {
                        add_link_atomic(
                            &neighbors,
                            &offsets,
                            &cum_nneighbor_per_level,
                            id as usize,
                            i as i32,
                            level,
                            flat_ptr,
                            d,
                            dist_ref,
                            link_scratch,
                            link_shrink,
                        );
                    }

                    if !shrink_out.is_empty() {
                        curr_entry = shrink_out[0].1 as usize;
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
// Helper functions for neighbor and vector access
// ---------------------------------------------------------------------------

/// Get a vector slice from a raw pointer without bounds checking.
///
/// # Safety
/// Caller must ensure `id * dim + dim` does not exceed the flat array length.
#[inline(always)]
unsafe fn get_flat<'a>(ptr: *const f32, id: usize, dim: usize) -> &'a [f32] {
    std::slice::from_raw_parts(ptr.add(id * dim), dim)
}

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

/// FAISS-style diversity-aware neighbor pruning.
///
/// From a list of candidates sorted by ascending distance, selects up to
/// `max_size` neighbors that cover diverse directions. A candidate is rejected
/// if it is closer to an already-selected neighbor than to the query node.
///
/// `candidates` must be sorted by ascending distance (closest first).
/// Results are written into `output` (cleared first).
#[inline]
fn shrink_neighbor_list<D: Fn(&[f32], &[f32]) -> f32>(
    candidates: &[(f32, u32)],
    output: &mut Vec<(f32, u32)>,
    max_size: usize,
    flat_ptr: *const f32,
    dim: usize,
    dist_fn: &D,
) {
    output.clear();
    for &(dist_to_query, cand_id) in candidates {
        let mut good = true;
        let cand_vec = unsafe { get_flat(flat_ptr, cand_id as usize, dim) };
        for &(_, selected_id) in output.iter() {
            let dist_to_selected =
                dist_fn(cand_vec, unsafe { get_flat(flat_ptr, selected_id as usize, dim) });
            if dist_to_selected < dist_to_query {
                good = false;
                break;
            }
        }
        if good {
            output.push((dist_to_query, cand_id));
            if output.len() >= max_size {
                return;
            }
        }
    }
}

/// FAISS-style add_link: add a reverse connection with diversity pruning.
///
/// If there's an empty slot, just insert. If full, rebuild the entire
/// neighbor list using `shrink_neighbor_list` with the new link included.
fn add_link<D: Fn(&[f32], &[f32]) -> f32>(
    neighbors: &mut [i32],
    offsets: &[u64],
    cum_nn: &[i32],
    target: usize,
    source: i32,
    level: usize,
    flat_ptr: *const f32,
    dim: usize,
    dist_fn: &D,
    scratch: &mut Vec<(f32, u32)>,
    shrink_out: &mut Vec<(f32, u32)>,
) {
    let range = get_neighbor_range(offsets, cum_nn, target, level);
    if range.end > neighbors.len() {
        return;
    }
    let max_neighbors = range.end - range.start;

    // Check for empty slot (scan from end, matching FAISS)
    if range.end > range.start && neighbors[range.end - 1] == -1 {
        let mut i = range.end;
        while i > range.start && neighbors[i - 1] == -1 {
            i -= 1;
        }
        neighbors[i] = source;
        return;
    }

    // All slots full — rebuild with diversity pruning
    let target_vec = unsafe { get_flat(flat_ptr, target, dim) };

    // Collect all current neighbors + new source into scratch, sorted by distance
    scratch.clear();
    let source_dist = dist_fn(target_vec, unsafe { get_flat(flat_ptr, source as usize, dim) });
    scratch.push((source_dist, source as u32));
    for idx in range.clone() {
        let nb = neighbors[idx];
        if nb >= 0 {
            let d = dist_fn(target_vec, unsafe { get_flat(flat_ptr, nb as usize, dim) });
            scratch.push((d, nb as u32));
        }
    }
    scratch.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Select diverse neighbors
    shrink_neighbor_list(scratch, shrink_out, max_neighbors, flat_ptr, dim, dist_fn);

    // Write back
    let mut i = range.start;
    for &(_, id) in shrink_out.iter() {
        neighbors[i] = id as i32;
        i += 1;
    }
    // Clear remaining slots
    while i < range.end {
        neighbors[i] = -1;
        i += 1;
    }
}

/// Lock-free add_link with diversity pruning (parallel path).
///
/// If there's an empty slot, CAS to claim it. Otherwise, snapshot all current
/// neighbors, run shrink_neighbor_list with the new source included, and
/// write back via CAS. Lost races are tolerated — HNSW is robust to them.
fn add_link_atomic<D: Fn(&[f32], &[f32]) -> f32>(
    neighbors: &[AtomicI32],
    offsets: &[u64],
    cum_nn: &[i32],
    target: usize,
    source: i32,
    level: usize,
    flat_ptr: *const f32,
    dim: usize,
    dist_fn: &D,
    scratch: &mut Vec<(f32, u32)>,
    shrink_out: &mut Vec<(f32, u32)>,
) {
    let range = get_neighbor_range(offsets, cum_nn, target, level);
    if range.end > neighbors.len() {
        return;
    }
    let max_neighbors = range.end - range.start;

    // Try to claim an empty slot via CAS
    for idx in range.clone() {
        if neighbors[idx]
            .compare_exchange(-1, source, AtomicOrdering::Relaxed, AtomicOrdering::Relaxed)
            .is_ok()
        {
            return;
        }
    }

    // All slots occupied — rebuild with diversity pruning
    let target_vec = unsafe { get_flat(flat_ptr, target, dim) };

    // Snapshot current neighbors + source into scratch, sorted by distance
    scratch.clear();
    let source_dist = dist_fn(target_vec, unsafe { get_flat(flat_ptr, source as usize, dim) });
    scratch.push((source_dist, source as u32));
    for idx in range.clone() {
        let nb = neighbors[idx].load(AtomicOrdering::Relaxed);
        if nb >= 0 {
            let d = dist_fn(target_vec, unsafe { get_flat(flat_ptr, nb as usize, dim) });
            scratch.push((d, nb as u32));
        }
    }
    scratch.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Select diverse neighbors
    shrink_neighbor_list(scratch, shrink_out, max_neighbors, flat_ptr, dim, dist_fn);

    // Write back via store (best-effort, races tolerated)
    let mut i = range.start;
    for &(_, id) in shrink_out.iter() {
        neighbors[i].store(id as i32, AtomicOrdering::Relaxed);
        i += 1;
    }
    while i < range.end {
        neighbors[i].store(-1, AtomicOrdering::Relaxed);
        i += 1;
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
            seed: None,
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
            seed: None,
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
            seed: None,
        };

        let graph = build_hnsw_with_threads(&data, &config, 4).unwrap();
        assert_eq!(graph.ntotal, n);
        assert_eq!(graph.dimensions, d);
        assert!(graph.entry_point >= 0);
        assert!((graph.entry_point as usize) < n);
    }
}
