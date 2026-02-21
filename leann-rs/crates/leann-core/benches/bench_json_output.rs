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

use leann_core::hnsw::build::{build_hnsw, build_hnsw_with_pool};
use leann_core::hnsw::graph::HnswConfig;
use leann_core::hnsw::io::{read_hnsw_index, write_hnsw_standard};
use leann_core::hnsw::search::{
    SearchBuffers, SearchParams, search_hnsw, search_hnsw_buf, search_hnsw_recompute_buf,
};
use leann_core::hnsw::simd::{inner_product_distance, l2_distance, l2_distance_batch_4};

fn gen_vectors(rng: &mut StdRng, n: usize, d: usize) -> Array2<f32> {
    let data: Vec<f32> = (0..n * d).map(|_| rng.random::<f32>()).collect();
    Array2::from_shape_vec((n, d), data).unwrap()
}

fn gen_query(rng: &mut StdRng, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.random::<f32>()).collect()
}

/// Run a benchmark: warmup iterations, then `repeats` timed iterations.
/// Returns sorted Vec of elapsed seconds for each iteration.
fn bench_fn<F: FnMut()>(mut f: F, warmup: usize, repeats: usize) -> Vec<f64> {
    // Warmup
    for _ in 0..warmup {
        f();
    }

    // Timed runs
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times
}

/// Compute a quantile from a sorted slice (linear interpolation).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = lo + 1;
    let frac = pos - lo as f64;
    if hi >= sorted.len() {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[derive(serde::Serialize)]
struct BenchResult {
    // Quantiles in seconds
    p5_s: f64,
    p25_s: f64,
    p50_s: f64,
    p75_s: f64,
    p95_s: f64,
    mean_s: f64,
    n_iters: usize,
    // Convenience: median in other units
    median_s: f64,
    median_ms: f64,
    median_us: f64,
    median_ns: f64,
    mean_ms: f64,
    mean_us: f64,
    mean_ns: f64,
}

impl BenchResult {
    fn from_times(times: &[f64]) -> Self {
        let p5 = quantile(times, 0.05);
        let p25 = quantile(times, 0.25);
        let p50 = quantile(times, 0.50);
        let p75 = quantile(times, 0.75);
        let p95 = quantile(times, 0.95);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        Self {
            p5_s: p5,
            p25_s: p25,
            p50_s: p50,
            p75_s: p75,
            p95_s: p95,
            mean_s: mean,
            n_iters: times.len(),
            median_s: p50,
            median_ms: p50 * 1e3,
            median_us: p50 * 1e6,
            median_ns: p50 * 1e9,
            mean_ms: mean * 1e3,
            mean_us: mean * 1e6,
            mean_ns: mean * 1e9,
        }
    }

    /// Scale all time values by a constant (e.g. to convert batched distance to per-call).
    fn scaled(mut self, factor: f64) -> Self {
        self.p5_s *= factor;
        self.p25_s *= factor;
        self.p50_s *= factor;
        self.p75_s *= factor;
        self.p95_s *= factor;
        self.mean_s *= factor;
        self.median_s *= factor;
        self.median_ms *= factor;
        self.median_us *= factor;
        self.median_ns *= factor;
        self.mean_ms *= factor;
        self.mean_us *= factor;
        self.mean_ns *= factor;
        self
    }

    fn from_size(size: f64) -> Self {
        Self {
            p5_s: size,
            p25_s: size,
            p50_s: size,
            p75_s: size,
            p95_s: size,
            mean_s: size,
            n_iters: 1,
            median_s: size,
            median_ms: size,
            median_us: size,
            median_ns: size,
            mean_ms: size,
            mean_us: size,
            mean_ns: size,
        }
    }
}

fn main() {
    let skip_large = std::env::args().any(|a| a == "--skip-large");
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let hnsw_seed: Option<u64> = std::env::var("HNSW_SEED").ok().and_then(|s| s.parse().ok());
    let mut results: BTreeMap<String, BenchResult> = BTreeMap::new();

    // Create one thread pool, reused across all parallel builds.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    eprintln!("Using {num_threads} threads for parallel builds");
    if let Some(seed) = hnsw_seed {
        eprintln!("HNSW_SEED={seed}");
    }

    eprintln!("=== LEANN Rust HNSW Benchmarks (JSON output) ===");

    // ── Distance computation ────────────────────────────────────────
    // Batch multiple calls per timing interval to overcome Instant::now() overhead
    // (~20-50ns on macOS), which would drown out sub-100ns operations.
    eprintln!("\n[1/6] Distance computation...");
    let dist_batch = 1000;
    for dim in [128, 384, 768] {
        let mut rng = StdRng::seed_from_u64(42);
        let a: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        let times = bench_fn(
            || {
                for _ in 0..dist_batch {
                    black_box(l2_distance(black_box(&a), black_box(&b)));
                }
            },
            200,
            2_000,
        );
        results.insert(
            format!("distance/l2/{dim}"),
            BenchResult::from_times(&times).scaled(1.0 / dist_batch as f64),
        );

        let times = bench_fn(
            || {
                for _ in 0..dist_batch {
                    black_box(inner_product_distance(black_box(&a), black_box(&b)));
                }
            },
            200,
            2_000,
        );
        results.insert(
            format!("distance/ip/{dim}"),
            BenchResult::from_times(&times).scaled(1.0 / dist_batch as f64),
        );
    }

    // ── HNSW build ──────────────────────────────────────────────────
    eprintln!("[2/6] HNSW build...");
    let config = HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 64,
        distance_metric: leann_core::index::DistanceMetric::L2,
        is_compact: false,
        is_recompute: false,
        seed: hnsw_seed,
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
            5
        } else if n >= 10_000 {
            10
        } else {
            20
        };
        let times = bench_fn(
            || {
                let _ = build_hnsw_with_pool(&data, &config, &pool).unwrap();
            },
            1,
            repeats,
        );
        results.insert(format!("hnsw_build/{n}"), BenchResult::from_times(&times));
    }

    // ── HNSW search (stored vectors) ────────────────────────────────
    eprintln!("[3/6] HNSW search (stored vectors)...");
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
            let times = bench_fn(
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
                200,
                5000,
            );
            results.insert(
                format!("hnsw_search/ef{ef}"),
                BenchResult::from_times(&times),
            );
        }
    }

    // ── HNSW search (recompute) ─────────────────────────────────────
    eprintln!("[4/6] HNSW search (recompute with in-memory callback)...");
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

        // Pre-allocate buffers once, reuse across all recompute search calls
        let mut recompute_buffers = SearchBuffers::new(graph.ntotal);

        for ef in [16, 32, 64, 128, 256] {
            let params = SearchParams {
                ef_search: ef,
                recompute_embeddings: true,
                ..Default::default()
            };
            let flat_ref = &flat_vectors;
            let times = bench_fn(
                || {
                    let _ = search_hnsw_recompute_buf(
                        &graph,
                        &query,
                        top_k,
                        &params,
                        &mut recompute_buffers,
                        |node_ids, q, out| {
                            let n = node_ids.len();
                            let mut i = 0;
                            while i + 4 <= n {
                                let dists = l2_distance_batch_4(
                                    q,
                                    &flat_ref[node_ids[i] * d..(node_ids[i] + 1) * d],
                                    &flat_ref[node_ids[i + 1] * d..(node_ids[i + 1] + 1) * d],
                                    &flat_ref[node_ids[i + 2] * d..(node_ids[i + 2] + 1) * d],
                                    &flat_ref[node_ids[i + 3] * d..(node_ids[i + 3] + 1) * d],
                                );
                                out[i..i + 4].copy_from_slice(&dists);
                                i += 4;
                            }
                            while i < n {
                                out[i] = l2_distance(
                                    &flat_ref[node_ids[i] * d..(node_ids[i] + 1) * d],
                                    q,
                                );
                                i += 1;
                            }
                        },
                    );
                },
                200,
                5000,
            );
            results.insert(
                format!("hnsw_search_recompute/ef{ef}"),
                BenchResult::from_times(&times),
            );
        }
    }

    // ── Full pipeline (build + write + read + search) ───────────────
    eprintln!("[5/6] Full pipeline...");
    {
        let d = 384;
        let top_k = 10;

        for n in [100, 1_000, 10_000] {
            eprintln!("  Pipeline {n} vectors...");
            let mut rng = StdRng::seed_from_u64(42);
            let data = gen_vectors(&mut rng, n, d);
            let query = gen_query(&mut rng, d);

            let repeats = if n >= 10_000 { 5 } else { 10 };
            let times = bench_fn(
                || {
                    let graph = build_hnsw_with_pool(&data, &config, &pool).unwrap();
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
                BenchResult::from_times(&times),
            );

            // Also measure index size
            let graph = build_hnsw_with_pool(&data, &config, &pool).unwrap();
            let mut buf = Vec::new();
            write_hnsw_standard(&mut buf, &graph).unwrap();
            results.insert(
                format!("index_size_bytes/{n}"),
                BenchResult::from_size(buf.len() as f64),
            );
        }
    }

    // ── Passage lookup ────────────────────────────────────────────────
    eprintln!("[6/6] Passage lookup...");
    {
        use leann_core::index::PassageSource;
        use leann_core::passages::{Passage, PassageManager, write_id_map, write_passages};

        for n in [1_000, 10_000] {
            eprintln!("  Passage lookup n={n}...");
            let dir = tempfile::tempdir().unwrap();
            let passages_path = dir.path().join("test.passages.jsonl");
            let offset_path = dir.path().join("test.passages.idx");
            let id_map_path = dir.path().join("test.ids.txt");

            // Create passages with realistic-ish text
            let passages: Vec<Passage> = (0..n)
                .map(|i| {
                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert("doc_num".to_string(), serde_json::json!(i));
                    metadata.insert(
                        "topic".to_string(),
                        serde_json::json!(format!("topic_{}", i % 10)),
                    );
                    Passage {
                        id: i.to_string(),
                        text: format!(
                            "This is document number {} about topic {}. It contains some text \
                             that is representative of a typical passage in a RAG system.",
                            i,
                            i % 10
                        ),
                        metadata,
                    }
                })
                .collect();

            let ids: Vec<String> = passages.iter().map(|p| p.id.clone()).collect();
            write_passages(&passages, &passages_path, &offset_path).unwrap();
            write_id_map(&ids, &id_map_path).unwrap();

            let sources = vec![PassageSource {
                source_type: "jsonl".to_string(),
                path: passages_path.to_string_lossy().to_string(),
                index_path: offset_path.to_string_lossy().to_string(),
                path_relative: None,
                index_path_relative: None,
            }];

            let manager = PassageManager::load(&sources, None).unwrap();

            // Generate random lookup indices
            let mut rng = StdRng::seed_from_u64(42);
            let lookup_indices: Vec<usize> = (0..1000)
                .map(|_| (rng.random::<f32>() * n as f32) as usize % n)
                .collect();

            let times = bench_fn(
                || {
                    for &idx in &lookup_indices {
                        black_box(manager.get_passage_by_index(idx).unwrap());
                    }
                },
                5,
                100,
            );
            results.insert(
                format!("passage_lookup/{n}"),
                BenchResult::from_times(&times).scaled(1.0 / lookup_indices.len() as f64),
            );
        }
    }

    eprintln!("\nDone! Writing JSON to stdout.");
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
