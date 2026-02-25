//! Fuzz-style tests for HNSW I/O.
//!
//! These run on stable Rust as regular `#[test]` functions.
//! They exercise two invariants:
//!
//! 1. **Arbitrary bytes → no panic**: `read_hnsw_index` must return `Err`,
//!    never panic, on any input that isn't a valid index.
//! 2. **Roundtrip**: write → read → assert structural equivalence for
//!    randomly parameterised graphs in both standard and compact formats.

use leann_core::hnsw::graph::*;
use leann_core::hnsw::io::{read_hnsw_index, write_hnsw_compact, write_hnsw_standard};
use leann_core::index::DistanceMetric;
use std::io::Cursor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic PRNG (xorshift64) — no extra dependency needed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo)
    }
}

/// Build a standard-format graph with the given parameters.
fn make_standard_graph(ntotal: usize, m: usize, dim: usize, metric: DistanceMetric) -> HnswGraph {
    let metric_type = if matches!(metric, DistanceMetric::L2) {
        0
    } else {
        1
    };
    let neighbors_per_node = 2 * m;
    let levels = vec![1i32; ntotal];
    let cum = vec![neighbors_per_node as i32];

    let mut offsets = Vec::with_capacity(ntotal + 1);
    for i in 0..=ntotal {
        offsets.push((i * neighbors_per_node) as u64);
    }
    let neighbors = vec![-1i32; ntotal * neighbors_per_node];

    HnswGraph {
        ntotal,
        dimensions: dim,
        entry_point: 0,
        max_level: 0,
        levels,
        assign_probas: vec![1.0],
        cum_nneighbor_per_level: cum,
        config: HnswConfig {
            m,
            ef_construction: 16,
            ef_search: 16,
            distance_metric: metric,
            is_compact: false,
            is_recompute: false,
            seed: None,
        },
        metric_type,
        metric_arg: 0.0,
        storage: GraphStorage::Standard { offsets, neighbors },
        vector_storage: VectorStorage::Null,
    }
}

/// Build a compact-format graph with the given parameters.
fn make_compact_graph(ntotal: usize, m: usize, dim: usize, metric: DistanceMetric) -> HnswGraph {
    let metric_type = if matches!(metric, DistanceMetric::L2) {
        0
    } else {
        1
    };

    let levels = vec![1i32; ntotal];
    let cum = vec![(2 * m) as i32];

    let mut level_ptr = Vec::with_capacity(2 * ntotal);
    let mut node_offsets = Vec::with_capacity(ntotal + 1);
    let mut neighbors = Vec::new();

    for i in 0..ntotal {
        node_offsets.push((i * 2) as u64);
        let nb_idx = neighbors.len() as u64;
        level_ptr.push(nb_idx);
        neighbors.push(((i + 1) % ntotal) as i32);
        level_ptr.push(nb_idx + 1);
    }
    node_offsets.push((ntotal * 2) as u64);

    HnswGraph {
        ntotal,
        dimensions: dim,
        entry_point: 0,
        max_level: 0,
        levels,
        assign_probas: vec![1.0],
        cum_nneighbor_per_level: cum,
        config: HnswConfig {
            m,
            ef_construction: 16,
            ef_search: 16,
            distance_metric: metric,
            is_compact: true,
            is_recompute: true,
            seed: None,
        },
        metric_type,
        metric_arg: 0.0,
        storage: GraphStorage::Compact {
            level_ptr,
            node_offsets,
            neighbors,
        },
        vector_storage: VectorStorage::Null,
    }
}

// ---------------------------------------------------------------------------
// 1. Arbitrary bytes → no panic
// ---------------------------------------------------------------------------

/// Feed random byte sequences to read_hnsw_index.
/// Must never panic — only Ok or Err.
#[test]
fn fuzz_read_random_bytes() {
    let mut rng = Rng::new(0xDEAD_BEEF);
    for _ in 0..10_000 {
        let len = rng.range(0, 512) as usize;
        let data: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        let mut cursor = Cursor::new(&data);
        let _ = read_hnsw_index(&mut cursor); // must not panic
    }
}

/// Feed all-zeros of various lengths — exercises allocation paths
/// with count=0 in read_vec.
#[test]
fn fuzz_read_zeros() {
    for len in 0..512 {
        let data = vec![0u8; len];
        let mut cursor = Cursor::new(&data);
        let _ = read_hnsw_index(&mut cursor);
    }
}

/// Feed bytes where every byte is 0xFF — maximises counts and triggers
/// overflow checks in read_vec.
#[test]
fn fuzz_read_all_ones() {
    for len in 0..512 {
        let data = vec![0xFFu8; len];
        let mut cursor = Cursor::new(&data);
        let _ = read_hnsw_index(&mut cursor);
    }
}

/// Valid HNSW header followed by garbage — tests that the parser
/// doesn't trust the header and then crash on body.
#[test]
fn fuzz_read_valid_header_garbage_body() {
    let graph = make_standard_graph(4, 4, 8, DistanceMetric::L2);
    let mut buf = Vec::new();
    write_hnsw_standard(&mut buf, &graph).unwrap();

    let mut rng = Rng::new(42);
    for _ in 0..2_000 {
        let mut corrupted = buf.clone();
        // Corrupt 1-8 random bytes in the body (after the 4-byte FourCC).
        let n_corruptions = rng.range(1, 9) as usize;
        for _ in 0..n_corruptions {
            let pos = rng.range(4, corrupted.len() as u64) as usize;
            corrupted[pos] = rng.next() as u8;
        }
        let mut cursor = Cursor::new(&corrupted);
        let _ = read_hnsw_index(&mut cursor); // must not panic
    }
}

/// Single bit-flips across a valid serialised index.
#[test]
fn fuzz_read_bitflip() {
    let graph = make_standard_graph(16, 4, 8, DistanceMetric::L2);
    let mut buf = Vec::new();
    write_hnsw_standard(&mut buf, &graph).unwrap();

    for byte_idx in 0..buf.len() {
        for bit in 0..8u8 {
            let mut corrupted = buf.clone();
            corrupted[byte_idx] ^= 1 << bit;
            let mut cursor = Cursor::new(&corrupted);
            let _ = read_hnsw_index(&mut cursor); // must not panic
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Roundtrip: write → read → verify
// ---------------------------------------------------------------------------

/// Roundtrip standard-format graphs with many ntotal values,
/// including those whose offsets count has LSB == 0x01
/// (the value that triggered the original false-positive compact detection).
#[test]
fn fuzz_roundtrip_standard() {
    let mut rng = Rng::new(0xCAFE);
    for _ in 0..500 {
        let ntotal = rng.range(1, 600) as usize;
        let m = rng.range(2, 33) as usize;
        let dim = rng.range(1, 257) as usize;
        let metric = if rng.next().is_multiple_of(2) {
            DistanceMetric::L2
        } else {
            DistanceMetric::Mips
        };

        let graph = make_standard_graph(ntotal, m, dim, metric);
        let mut buf = Vec::new();
        write_hnsw_standard(&mut buf, &graph).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = read_hnsw_index(&mut cursor).unwrap();

        assert_eq!(
            loaded.ntotal, ntotal,
            "ntotal mismatch for n={ntotal} m={m}"
        );
        assert_eq!(loaded.dimensions, dim);
        assert!(
            !loaded.is_compact(),
            "n={ntotal} m={m}: detected as compact"
        );
    }
}

/// Roundtrip compact-format graphs with many ntotal values.
#[test]
fn fuzz_roundtrip_compact() {
    let mut rng = Rng::new(0xBEEF);
    for _ in 0..500 {
        let ntotal = rng.range(1, 600) as usize;
        let m = rng.range(2, 33) as usize;
        let dim = rng.range(1, 257) as usize;
        let metric = if rng.next().is_multiple_of(2) {
            DistanceMetric::L2
        } else {
            DistanceMetric::Mips
        };

        let graph = make_compact_graph(ntotal, m, dim, metric);
        let mut buf = Vec::new();
        write_hnsw_compact(&mut buf, &graph).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = read_hnsw_index(&mut cursor).unwrap();

        assert_eq!(
            loaded.ntotal, ntotal,
            "ntotal mismatch for n={ntotal} m={m}"
        );
        assert_eq!(loaded.dimensions, dim);
        assert!(
            loaded.is_compact(),
            "n={ntotal} m={m}: detected as standard"
        );
    }
}

/// Specifically test ntotal values where (ntotal+1) % 256 == 1,
/// i.e. the offsets count LSB is 0x01 — the exact pattern that
/// caused the original compact-detection false positive.
#[test]
fn fuzz_roundtrip_standard_ambiguous_counts() {
    for &ntotal in &[256, 512, 1, 257, 513] {
        for &m in &[2, 4, 8, 16, 32] {
            let graph = make_standard_graph(ntotal, m, 8, DistanceMetric::L2);
            let mut buf = Vec::new();
            write_hnsw_standard(&mut buf, &graph).unwrap();

            let mut cursor = Cursor::new(&buf);
            let loaded = read_hnsw_index(&mut cursor)
                .unwrap_or_else(|e| panic!("failed for n={ntotal} m={m}: {e}"));

            assert_eq!(loaded.ntotal, ntotal);
            assert!(
                !loaded.is_compact(),
                "n={ntotal} m={m}: falsely detected as compact"
            );
        }
    }
}
