//! Standalone benchmark binary that outputs JSON results to stdout.
//!
//! This avoids parsing criterion's internal format and gives a clean JSON
//! that the comparison script can consume.
//!
//! Usage:
//!   cargo bench --package leann-core --bench bench_json_output -- [--skip-large]

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use leann_core::hnsw::build::{build_hnsw, build_hnsw_with_threads};
use leann_core::hnsw::graph::HnswConfig;
use leann_core::hnsw::io::{read_hnsw_index, write_hnsw_standard};
use leann_core::hnsw::search::{
    search_hnsw, search_hnsw_buf, search_hnsw_recompute, SearchBuffers, SearchParams,
};
use leann_core::hnsw::simd::{inner_product_distance, l2_distance};

fn gen_vectors(rng: &mut StdRng, n: usize, d: usize) -> Array2<f32> {
    let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
    Array2::from_shape_vec((n, d), data).unwrap()
}

fn gen_query(rng: &mut StdRng, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.gen::<f32>()).collect()
}

/// Run a benchmark: warmup iterations, then `repeats` timed iterations.
/// Returns (median_secs, mean_secs).
fn bench_fn<F: FnMut()>(mut f: F, warmup: usize, repeats: usize) -> (f64, f64) {
    // Warmup
    for _ in 0..warmup {
        f();
    }

    // Timed runs
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        f();
        times.push(start.elapsed());
    }

    times.sort();
    let median = times[times.len() / 2].as_secs_f64();
    let mean = times.iter().map(|t| t.as_secs_f64()).sum::<f64>() / times.len() as f64;
    (median, mean)
}

#[derive(serde::Serialize)]
struct BenchResult {
    median_s: f64,
    mean_s: f64,
    median_ms: f64,
    mean_ms: f64,
    median_us: f64,
    mean_us: f64,
    median_ns: f64,
    mean_ns: f64,
}

impl BenchResult {
    fn from_secs(median: f64, mean: f64) -> Self {
        Self {
            median_s: median,
            mean_s: mean,
            median_ms: median * 1e3,
            mean_ms: mean * 1e3,
            median_us: median * 1e6,
            mean_us: mean * 1e6,
            median_ns: median * 1e9,
            mean_ns: mean * 1e9,
        }
    }
}

fn main() {
    let skip_large = std::env::args().any(|a| a == "--skip-large");
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut results: BTreeMap<String, BenchResult> = BTreeMap::new();

    eprintln!("Using {num_threads} threads for parallel builds");

    eprintln!("=== LEANN Rust HNSW Benchmarks (JSON output) ===");

    // ── Distance computation ────────────────────────────────────────
    // Batch multiple calls per timing interval to overcome Instant::now() overhead
    // (~20-50ns on macOS), which would drown out sub-100ns operations.
    eprintln!("\n[1/5] Distance computation...");
    let dist_batch = 1000;
    for dim in [128, 384, 768] {
        let mut rng = StdRng::seed_from_u64(42);
        let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let (med, mean) = bench_fn(
            || {
                for _ in 0..dist_batch {
                    black_box(l2_distance(black_box(&a), black_box(&b)));
                }
            },
            100,
            1_000,
        );
        results.insert(
            format!("distance/l2/{dim}"),
            BenchResult::from_secs(med / dist_batch as f64, mean / dist_batch as f64),
        );

        let (med, mean) = bench_fn(
            || {
                for _ in 0..dist_batch {
                    black_box(inner_product_distance(black_box(&a), black_box(&b)));
                }
            },
            100,
            1_000,
        );
        results.insert(
            format!("distance/ip/{dim}"),
            BenchResult::from_secs(med / dist_batch as f64, mean / dist_batch as f64),
        );
    }

    // ── HNSW build ──────────────────────────────────────────────────
    eprintln!("[2/5] HNSW build...");
    let config = HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 64,
        distance_metric: leann_core::index::DistanceMetric::L2,
        is_compact: false,
        is_recompute: false,
    };

    let build_sizes: Vec<usize> = if skip_large {
        vec![100, 1_000, 10_000]
    } else {
        vec![100, 1_000, 10_000, 50_000]
    };

    for &n in &build_sizes {
        eprintln!("  Building {n} vectors...");
        let mut rng = StdRng::seed_from_u64(42);
        let data = gen_vectors(&mut rng, n, 384);

        let repeats = if n >= 50_000 {
            3
        } else if n >= 10_000 {
            5
        } else {
            10
        };
        let (med, mean) = bench_fn(
            || {
                let _ = build_hnsw_with_threads(&data, &config, num_threads).unwrap();
            },
            1,
            repeats,
        );
        results.insert(format!("hnsw_build/{n}"), BenchResult::from_secs(med, mean));
    }

    // ── HNSW search (stored vectors) ────────────────────────────────
    eprintln!("[3/5] HNSW search (stored vectors)...");
    {
        let n = 10_000;
        let d = 384;
        let top_k = 10;

        let mut rng = StdRng::seed_from_u64(42);
        let data = gen_vectors(&mut rng, n, d);
        let graph = build_hnsw(&data, &config).unwrap();
        let flat_vectors: Vec<f32> = data.iter().copied().collect();
        let query = gen_query(&mut rng, d);

        // Pre-allocate buffers once, reuse across all search calls
        let mut buffers = SearchBuffers::new(graph.ntotal);

        for ef in [16, 32, 64, 128, 256] {
            let params = SearchParams {
                ef_search: ef,
                ..Default::default()
            };
            let (med, mean) = bench_fn(
                || {
                    let _ = search_hnsw_buf(
                        &graph,
                        &query,
                        top_k,
                        &flat_vectors,
                        &params,
                        &mut buffers,
                    );
                },
                100,
                1000,
            );
            results.insert(
                format!("hnsw_search/ef{ef}"),
                BenchResult::from_secs(med, mean),
            );
        }
    }

    // ── HNSW search (recompute) ─────────────────────────────────────
    eprintln!("[4/5] HNSW search (recompute with in-memory callback)...");
    {
        let n = 10_000;
        let d = 384;
        let top_k = 10;

        let mut rng = StdRng::seed_from_u64(42);
        let data = gen_vectors(&mut rng, n, d);

        let recompute_config = HnswConfig {
            is_recompute: true,
            ..config.clone()
        };
        let graph = build_hnsw(&data, &recompute_config).unwrap();
        let flat_vectors: Vec<f32> = data.iter().copied().collect();
        let query = gen_query(&mut rng, d);

        for ef in [16, 32, 64, 128, 256] {
            let params = SearchParams {
                ef_search: ef,
                recompute_embeddings: true,
                ..Default::default()
            };
            let flat_ref = &flat_vectors;
            let (med, mean) = bench_fn(
                || {
                    let _ = search_hnsw_recompute(&graph, &query, top_k, &params, |node_ids, q| {
                        node_ids
                            .iter()
                            .map(|&id| {
                                let vec = &flat_ref[id * d..(id + 1) * d];
                                l2_distance(vec, q)
                            })
                            .collect()
                    });
                },
                100,
                1000,
            );
            results.insert(
                format!("hnsw_search_recompute/ef{ef}"),
                BenchResult::from_secs(med, mean),
            );
        }
    }

    // ── Full pipeline (build + write + read + search) ───────────────
    eprintln!("[5/5] Full pipeline...");
    {
        let d = 384;
        let top_k = 10;

        for n in [100, 1_000, 10_000] {
            eprintln!("  Pipeline {n} vectors...");
            let mut rng = StdRng::seed_from_u64(42);
            let data = gen_vectors(&mut rng, n, d);
            let query = gen_query(&mut rng, d);

            let repeats = if n >= 10_000 { 3 } else { 5 };
            let (med, mean) = bench_fn(
                || {
                    let graph = build_hnsw_with_threads(&data, &config, num_threads).unwrap();
                    let mut buf = Vec::new();
                    write_hnsw_standard(&mut buf, &graph).unwrap();
                    let mut cursor = std::io::Cursor::new(&buf);
                    let loaded = read_hnsw_index(&mut cursor).unwrap();
                    let flat_vectors: Vec<f32> = data.iter().copied().collect();
                    let params = SearchParams {
                        ef_search: 64,
                        ..Default::default()
                    };
                    let _ = search_hnsw(&loaded, &query, top_k, &flat_vectors, &params);
                },
                1,
                repeats,
            );
            results.insert(
                format!("full_pipeline/{n}"),
                BenchResult::from_secs(med, mean),
            );

            // Also measure index size
            let graph = build_hnsw_with_threads(&data, &config, num_threads).unwrap();
            let mut buf = Vec::new();
            write_hnsw_standard(&mut buf, &graph).unwrap();
            results.insert(
                format!("index_size_bytes/{n}"),
                BenchResult {
                    median_s: buf.len() as f64,
                    mean_s: buf.len() as f64,
                    median_ms: buf.len() as f64,
                    mean_ms: buf.len() as f64,
                    median_us: buf.len() as f64,
                    mean_us: buf.len() as f64,
                    median_ns: buf.len() as f64,
                    mean_ns: buf.len() as f64,
                },
            );
        }
    }

    eprintln!("\nDone! Writing JSON to stdout.");
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
