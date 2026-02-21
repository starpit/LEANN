use super::graph::*;
use super::simd::{
    VisitedList, inner_product_distance, inner_product_distance_batch_4, l2_distance,
    l2_distance_batch_4,
};

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

// ── Flat-array min-heap for candidates ───────────────────────────────

/// Min-heap using separate flat arrays for distances and IDs.
/// Direct f32 comparison, no Ord trait, no bounds checking in sift.
pub(crate) struct FlatMinHeap {
    dis: Vec<f32>,
    ids: Vec<u32>,
    len: usize,
}

impl FlatMinHeap {
    #[inline]
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            dis: Vec::with_capacity(capacity),
            ids: Vec::with_capacity(capacity),
            len: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Peek at the minimum element without removing it.
    #[inline(always)]
    pub(crate) fn peek(&self) -> (f32, u32) {
        debug_assert!(self.len > 0);
        unsafe { (*self.dis.get_unchecked(0), *self.ids.get_unchecked(0)) }
    }

    #[inline]
    pub(crate) fn push(&mut self, dis: f32, id: u32) {
        let pos = self.len;
        if pos == self.dis.len() {
            self.dis.push(dis);
            self.ids.push(id);
        } else {
            unsafe {
                *self.dis.get_unchecked_mut(pos) = dis;
                *self.ids.get_unchecked_mut(pos) = id;
            }
        }
        self.len += 1;
        self.sift_up(pos);
    }

    /// Pop the minimum element. Caller must check is_empty() first.
    #[inline]
    pub(crate) fn pop(&mut self) -> (f32, u32) {
        debug_assert!(self.len > 0);
        unsafe {
            let dis = *self.dis.get_unchecked(0);
            let id = *self.ids.get_unchecked(0);
            self.len -= 1;
            if self.len > 0 {
                *self.dis.get_unchecked_mut(0) = *self.dis.get_unchecked(self.len);
                *self.ids.get_unchecked_mut(0) = *self.ids.get_unchecked(self.len);
                self.sift_down(0);
            }
            (dis, id)
        }
    }

    #[inline]
    fn sift_up(&mut self, mut pos: usize) {
        unsafe {
            let d = *self.dis.get_unchecked(pos);
            let id = *self.ids.get_unchecked(pos);
            while pos > 0 {
                let parent = (pos - 1) >> 1;
                let pd = *self.dis.get_unchecked(parent);
                if d < pd {
                    *self.dis.get_unchecked_mut(pos) = pd;
                    *self.ids.get_unchecked_mut(pos) = *self.ids.get_unchecked(parent);
                    pos = parent;
                } else {
                    break;
                }
            }
            *self.dis.get_unchecked_mut(pos) = d;
            *self.ids.get_unchecked_mut(pos) = id;
        }
    }

    #[inline]
    fn sift_down(&mut self, mut pos: usize) {
        let n = self.len;
        unsafe {
            let d = *self.dis.get_unchecked(pos);
            let id = *self.ids.get_unchecked(pos);
            loop {
                let left = 2 * pos + 1;
                if left >= n {
                    break;
                }
                let right = left + 1;
                let mut smallest = left;
                if right < n && *self.dis.get_unchecked(right) < *self.dis.get_unchecked(left) {
                    smallest = right;
                }
                let sd = *self.dis.get_unchecked(smallest);
                if sd < d {
                    *self.dis.get_unchecked_mut(pos) = sd;
                    *self.ids.get_unchecked_mut(pos) = *self.ids.get_unchecked(smallest);
                    pos = smallest;
                } else {
                    break;
                }
            }
            *self.dis.get_unchecked_mut(pos) = d;
            *self.ids.get_unchecked_mut(pos) = id;
        }
    }
}

// ── Flat-array max-heap for results ──────────────────────────────────

/// Max-heap using separate flat arrays. Worst distance at root for O(1) rejection.
pub(crate) struct FlatMaxHeap {
    dis: Vec<f32>,
    ids: Vec<u32>,
    len: usize,
}

impl FlatMaxHeap {
    #[inline]
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            dis: Vec::with_capacity(capacity),
            ids: Vec::with_capacity(capacity),
            len: 0,
        }
    }

    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Peek at the maximum distance (heap root).
    #[inline(always)]
    pub(crate) fn peek_max_dis(&self) -> f32 {
        debug_assert!(self.len > 0);
        unsafe { *self.dis.get_unchecked(0) }
    }

    /// Pop the maximum element. Caller must check len() > 0 first.
    #[inline]
    pub(crate) fn pop_max(&mut self) -> (f32, u32) {
        debug_assert!(self.len > 0);
        unsafe {
            let dis = *self.dis.get_unchecked(0);
            let id = *self.ids.get_unchecked(0);
            self.len -= 1;
            if self.len > 0 {
                *self.dis.get_unchecked_mut(0) = *self.dis.get_unchecked(self.len);
                *self.ids.get_unchecked_mut(0) = *self.ids.get_unchecked(self.len);
                self.sift_down(0);
            }
            (dis, id)
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, dis: f32, id: u32) {
        let pos = self.len;
        if pos == self.dis.len() {
            self.dis.push(dis);
            self.ids.push(id);
        } else {
            unsafe {
                *self.dis.get_unchecked_mut(pos) = dis;
                *self.ids.get_unchecked_mut(pos) = id;
            }
        }
        self.len += 1;
        self.sift_up(pos);
    }

    /// Replace the maximum (root) with a new element and sift down.
    /// More efficient than pop + push.
    #[inline]
    pub(crate) fn replace_max(&mut self, dis: f32, id: u32) {
        debug_assert!(self.len > 0);
        self.dis[0] = dis;
        self.ids[0] = id;
        self.sift_down(0);
    }

    /// In-place heap-sort: sort entries by ascending distance with zero allocations.
    /// Returns slices into the internal arrays. The heap property is destroyed;
    /// call `clear()` before reusing.
    pub(crate) fn drain_sorted(&mut self) -> (&[u32], &[f32]) {
        let n = self.len;
        // Heap-sort: repeatedly swap root (max) with last and shrink heap.
        // Produces ascending order in dis[0..n] and ids[0..n].
        while self.len > 1 {
            self.len -= 1;
            self.dis.swap(0, self.len);
            self.ids.swap(0, self.len);
            self.sift_down(0);
        }
        self.len = n;
        (&self.ids[..n], &self.dis[..n])
    }

    #[inline]
    fn sift_up(&mut self, mut pos: usize) {
        unsafe {
            let d = *self.dis.get_unchecked(pos);
            let id = *self.ids.get_unchecked(pos);
            while pos > 0 {
                let parent = (pos - 1) >> 1;
                let pd = *self.dis.get_unchecked(parent);
                if d > pd {
                    *self.dis.get_unchecked_mut(pos) = pd;
                    *self.ids.get_unchecked_mut(pos) = *self.ids.get_unchecked(parent);
                    pos = parent;
                } else {
                    break;
                }
            }
            *self.dis.get_unchecked_mut(pos) = d;
            *self.ids.get_unchecked_mut(pos) = id;
        }
    }

    #[inline]
    fn sift_down(&mut self, mut pos: usize) {
        let n = self.len;
        unsafe {
            let d = *self.dis.get_unchecked(pos);
            let id = *self.ids.get_unchecked(pos);
            loop {
                let left = 2 * pos + 1;
                if left >= n {
                    break;
                }
                let right = left + 1;
                let mut largest = left;
                if right < n && *self.dis.get_unchecked(right) > *self.dis.get_unchecked(left) {
                    largest = right;
                }
                let ld = *self.dis.get_unchecked(largest);
                if ld > d {
                    *self.dis.get_unchecked_mut(pos) = ld;
                    *self.ids.get_unchecked_mut(pos) = *self.ids.get_unchecked(largest);
                    pos = largest;
                } else {
                    break;
                }
            }
            *self.dis.get_unchecked_mut(pos) = d;
            *self.ids.get_unchecked_mut(pos) = id;
        }
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
/// to avoid repeated heap and visited-list allocation.
pub struct SearchBuffers {
    visited: VisitedList,
    candidates: FlatMinHeap,
    results: FlatMaxHeap,
}

impl SearchBuffers {
    /// Create new search buffers for a graph with `ntotal` nodes.
    /// Heaps start empty and grow on first use; subsequent searches reuse capacity.
    pub fn new(ntotal: usize) -> Self {
        Self {
            visited: VisitedList::new(ntotal),
            candidates: FlatMinHeap::new(0),
            results: FlatMaxHeap::new(0),
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
/// repeated allocation of the visited list and search heaps.
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
            buffers,
            l2_distance,
            l2_distance_batch_4,
        ),
        _ => search_hnsw_inner(
            graph,
            query,
            top_k,
            vectors,
            params,
            buffers,
            inner_product_distance,
            inner_product_distance_batch_4,
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
    unsafe { vectors.get_unchecked(id * dim..id * dim + dim) }
}

/// Optimized HNSW search with dual flat-array heaps and batched 4x distance.
///
/// Key optimizations:
/// - Custom flat-array heaps with u32 IDs and direct f32 comparison (no Ord trait)
/// - Batched 4x distance: query loaded once, reused across 4 database vectors
/// - Combined check_and_set: single cache access for visited check + mark
/// - replace_max on result heap: single sift_down instead of pop + push
#[allow(clippy::too_many_arguments)]
fn search_hnsw_inner<D, B>(
    graph: &HnswGraph,
    query: &[f32],
    top_k: usize,
    vectors: &[f32],
    params: &SearchParams,
    buffers: &mut SearchBuffers,
    dist_fn: D,
    dist_batch_4: B,
) -> (Vec<usize>, Vec<f32>)
where
    D: Fn(&[f32], &[f32]) -> f32,
    B: Fn(&[f32], &[f32], &[f32], &[f32], &[f32]) -> [f32; 4],
{
    let d = graph.dimensions;
    let ef = params.ef_search.max(top_k);

    // Destructure for disjoint mutable borrows on each field.
    let SearchBuffers {
        visited,
        candidates,
        results,
    } = buffers;

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
                    break;
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

    // Phase 2: Search at level 0 — reuse pre-allocated heaps
    candidates.clear();
    results.clear();
    visited.reset();

    let d_entry = unsafe { dist_fn(query, get_vec(vectors, curr, d)) };
    candidates.push(d_entry, curr as u32);
    results.push(d_entry, curr as u32);
    visited.set(curr);

    // Cache worst distance to avoid heap peek on every neighbor check.
    let mut worst_dist = d_entry;

    // Scratch buffer for batching unvisited neighbors
    let mut saved: [u32; 4] = [0; 4];

    while !candidates.is_empty() {
        let (cand_dist, cand_id) = candidates.pop();

        // If candidate is worse than worst result and we have enough results, stop
        if results.len() >= ef && cand_dist > worst_dist {
            break;
        }

        let neighbors = graph.get_neighbors(cand_id as usize, 0);

        // Pass 1: prefetch visited-list entries into L2 cache
        for &nb in neighbors {
            if nb < 0 {
                break;
            }
            visited.prefetch_l2(nb as usize);
        }

        // Pass 2: check visited + accumulate batches of 4 for distance
        let mut counter = 0;
        for &nb in neighbors {
            if nb < 0 {
                break;
            }
            if !visited.check_and_set(nb as usize) {
                continue;
            }

            saved[counter] = nb as u32;
            counter += 1;

            // Prefetch this vector's data so it's in cache by the time batch_4 runs.
            // First 2 cache lines (128 bytes = 32 floats); the HW prefetcher handles the rest.
            unsafe {
                let vptr = vectors.as_ptr().add(nb as usize * d) as *const u8;
                #[cfg(target_arch = "aarch64")]
                {
                    std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) vptr, options(nostack, preserves_flags));
                    std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) vptr.add(64), options(nostack, preserves_flags));
                }
                #[cfg(target_arch = "x86_64")]
                {
                    std::arch::x86_64::_mm_prefetch(
                        vptr as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                    std::arch::x86_64::_mm_prefetch(
                        vptr.add(64) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            if counter == 4 {
                let dists = unsafe {
                    dist_batch_4(
                        query,
                        get_vec(vectors, saved[0] as usize, d),
                        get_vec(vectors, saved[1] as usize, d),
                        get_vec(vectors, saved[2] as usize, d),
                        get_vec(vectors, saved[3] as usize, d),
                    )
                };

                for k in 0..4 {
                    let nb_id = saved[k];
                    let d_nb = dists[k];

                    if results.len() < ef {
                        candidates.push(d_nb, nb_id);
                        results.push(d_nb, nb_id);
                        if results.len() == ef {
                            worst_dist = results.peek_max_dis();
                        }
                    } else if d_nb < worst_dist {
                        candidates.push(d_nb, nb_id);
                        results.replace_max(d_nb, nb_id);
                        worst_dist = results.peek_max_dis();
                    }
                }
                counter = 0;
            }
        }

        // Process remainder (1-3 leftover neighbors)
        for &nb_id in &saved[..counter] {
            let d_nb = unsafe { dist_fn(query, get_vec(vectors, nb_id as usize, d)) };

            if results.len() < ef {
                candidates.push(d_nb, nb_id);
                results.push(d_nb, nb_id);
                if results.len() == ef {
                    worst_dist = results.peek_max_dis();
                }
            } else if d_nb < worst_dist {
                candidates.push(d_nb, nb_id);
                results.replace_max(d_nb, nb_id);
                worst_dist = results.peek_max_dis();
            }
        }
    }

    // In-place heap-sort, then collect top_k into return Vecs
    let (sorted_ids, sorted_dis) = results.drain_sorted();
    let n = top_k.min(sorted_ids.len());
    let labels: Vec<usize> = sorted_ids[..n].iter().map(|&id| id as usize).collect();
    let distances: Vec<f32> = sorted_dis[..n].to_vec();

    (labels, distances)
}

/// Search the HNSW graph using recomputed distances via a callback.
///
/// Convenience wrapper that allocates fresh buffers. For repeated searches,
/// use [`search_hnsw_recompute_buf`] to reuse allocations.
pub fn search_hnsw_recompute<F>(
    graph: &HnswGraph,
    query: &[f32],
    top_k: usize,
    params: &SearchParams,
    compute_distance: F,
) -> (Vec<usize>, Vec<f32>)
where
    F: FnMut(&[usize], &[f32], &mut [f32]),
{
    let mut buffers = SearchBuffers::new(graph.ntotal);
    search_hnsw_recompute_buf(graph, query, top_k, params, &mut buffers, compute_distance)
}

/// Search the HNSW graph using recomputed distances, reusing pre-allocated buffers.
///
/// The `compute_distance` callback takes `(node_ids, query, out)` and writes
/// distances into the pre-allocated `out` slice. The callback should use
/// `l2_distance_batch_4` (or similar) internally to batch-compute distances with SIMD.
pub fn search_hnsw_recompute_buf<F>(
    graph: &HnswGraph,
    query: &[f32],
    top_k: usize,
    params: &SearchParams,
    buffers: &mut SearchBuffers,
    mut compute_distance: F,
) -> (Vec<usize>, Vec<f32>)
where
    F: FnMut(&[usize], &[f32], &mut [f32]),
{
    let ef = params.ef_search.max(top_k);
    let max_neighbors = graph.neighbors_at_level(0);

    let SearchBuffers {
        visited,
        candidates,
        results,
    } = buffers;

    // Pre-allocate scratch buffers reused across iterations
    let mut node_buf = Vec::with_capacity(max_neighbors + 1);
    let mut dist_buf = vec![0.0f32; max_neighbors + 1];

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
            let n = node_buf.len();
            compute_distance(&node_buf, query, &mut dist_buf[..n]);

            let curr_dist = dist_buf[n - 1];
            for (i, &nb) in node_buf.iter().enumerate().take(n - 1) {
                if dist_buf[i] < curr_dist {
                    curr = nb;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    // Phase 2: ef-search at level 0 — reuse pre-allocated heaps
    candidates.clear();
    results.clear();
    visited.reset();

    node_buf.clear();
    node_buf.push(curr);
    compute_distance(&node_buf, query, &mut dist_buf[..1]);
    let d_entry = dist_buf[0];
    candidates.push(d_entry, curr as u32);
    results.push(d_entry, curr as u32);
    visited.set(curr);

    let mut worst_dist = d_entry;

    while !candidates.is_empty() {
        let (cand_dist, cand_id) = candidates.pop();

        if results.len() >= ef && cand_dist > worst_dist {
            break;
        }

        let neighbors = graph.get_neighbors(cand_id as usize, 0);

        node_buf.clear();
        for &nb in neighbors {
            if nb >= 0 {
                let nb = nb as usize;
                if visited.check_and_set(nb) {
                    node_buf.push(nb);
                }
            }
        }

        if node_buf.is_empty() {
            continue;
        }

        let n = node_buf.len();
        compute_distance(&node_buf, query, &mut dist_buf[..n]);

        for i in 0..n {
            let d_nb = dist_buf[i];
            let nb = node_buf[i] as u32;

            if results.len() < ef {
                candidates.push(d_nb, nb);
                results.push(d_nb, nb);
                if results.len() == ef {
                    worst_dist = results.peek_max_dis();
                }
            } else if d_nb < worst_dist {
                candidates.push(d_nb, nb);
                results.replace_max(d_nb, nb);
                worst_dist = results.peek_max_dis();
            }
        }
    }

    // In-place heap-sort, then collect top_k into return Vecs
    let (sorted_ids, sorted_dis) = results.drain_sorted();
    let n = top_k.min(sorted_ids.len());
    let labels: Vec<usize> = sorted_ids[..n].iter().map(|&id| id as usize).collect();
    let distances: Vec<f32> = sorted_dis[..n].to_vec();

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
            seed: None,
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
            seed: None,
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

        let (labels, _distances) =
            search_hnsw_recompute(&graph, &query, 2, &params, |node_ids, q, out| {
                for (i, &id) in node_ids.iter().enumerate() {
                    let vec = &flat_vectors[id * d..(id + 1) * d];
                    out[i] = vec
                        .iter()
                        .zip(q.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                }
            });

        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], 0);
    }
}
