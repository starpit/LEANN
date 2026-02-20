use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

use leann_core::hnsw::build::build_hnsw;
use leann_core::hnsw::graph::HnswConfig;
use leann_core::hnsw::io::{read_hnsw_index, write_hnsw_standard};
use leann_core::hnsw::search::{search_hnsw, search_hnsw_recompute, SearchParams};
use leann_core::hnsw::simd::{inner_product_distance, l2_distance, l2_distance_batch_4};

fn gen_vectors(rng: &mut StdRng, n: usize, d: usize) -> Array2<f32> {
    let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
    Array2::from_shape_vec((n, d), data).unwrap()
}

fn gen_query(rng: &mut StdRng, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.gen::<f32>()).collect()
}

// ── Distance computation benchmarks ─────────────────────────────────

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [128, 384, 768] {
        let a: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        group.bench_with_input(BenchmarkId::new("l2", dim), &dim, |bench, _| {
            bench.iter(|| l2_distance(&a, &b));
        });

        group.bench_with_input(BenchmarkId::new("inner_product", dim), &dim, |bench, _| {
            bench.iter(|| inner_product_distance(&a, &b));
        });
    }

    group.finish();
}

// ── HNSW build benchmarks ───────────────────────────────────────────

fn bench_hnsw_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_build");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let config = HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 64,
        distance_metric: leann_core::index::DistanceMetric::L2,
        is_compact: false,
        is_recompute: false,
        seed: None,
    };

    for n in [100, 1_000, 10_000, 50_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let data = gen_vectors(&mut rng, n, 384);

        group.bench_with_input(BenchmarkId::new("vectors", n), &n, |bench, _| {
            bench.iter(|| build_hnsw(&data, &config).unwrap());
        });
    }

    group.finish();
}

// ── HNSW search (stored vectors) benchmarks ─────────────────────────

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    group.sample_size(50);

    let n = 10_000;
    let d = 384;
    let top_k = 10;

    let mut rng = StdRng::seed_from_u64(42);
    let data = gen_vectors(&mut rng, n, d);

    let config = HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 64,
        distance_metric: leann_core::index::DistanceMetric::L2,
        is_compact: false,
        is_recompute: false,
        seed: None,
    };

    let graph = build_hnsw(&data, &config).unwrap();
    let flat_vectors: Vec<f32> = data.iter().copied().collect();
    let query = gen_query(&mut rng, d);

    for ef in [16, 32, 64, 128, 256] {
        group.bench_with_input(BenchmarkId::new("ef_search", ef), &ef, |bench, &ef| {
            let params = SearchParams {
                ef_search: ef,
                ..Default::default()
            };
            bench.iter(|| search_hnsw(&graph, &query, top_k, &flat_vectors, &params));
        });
    }

    group.finish();
}

// ── HNSW search (recompute with in-memory callback) ─────────────────

fn bench_hnsw_search_recompute(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_recompute");
    group.sample_size(50);

    let n = 10_000;
    let d = 384;
    let top_k = 10;

    let mut rng = StdRng::seed_from_u64(42);
    let data = gen_vectors(&mut rng, n, d);

    let config = HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 64,
        distance_metric: leann_core::index::DistanceMetric::L2,
        is_compact: false,
        is_recompute: true,
        seed: None,
    };

    let graph = build_hnsw(&data, &config).unwrap();
    let flat_vectors: Vec<f32> = data.iter().copied().collect();
    let query = gen_query(&mut rng, d);

    for ef in [16, 32, 64, 128, 256] {
        let flat_ref = &flat_vectors;
        group.bench_with_input(BenchmarkId::new("ef_search", ef), &ef, |bench, &ef| {
            let params = SearchParams {
                ef_search: ef,
                recompute_embeddings: true,
                ..Default::default()
            };
            bench.iter(|| {
                search_hnsw_recompute(&graph, &query, top_k, &params, |node_ids, q, out| {
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
                        out[i] = l2_distance(&flat_ref[node_ids[i] * d..(node_ids[i] + 1) * d], q);
                        i += 1;
                    }
                })
            });
        });
    }

    group.finish();
}

// ── Full pipeline benchmarks (build + write + read + search) ────────

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let d = 384;
    let top_k = 10;

    let config = HnswConfig {
        m: 32,
        ef_construction: 200,
        ef_search: 64,
        distance_metric: leann_core::index::DistanceMetric::L2,
        is_compact: false,
        is_recompute: false,
        seed: None,
    };

    for n in [100, 1_000, 10_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let data = gen_vectors(&mut rng, n, d);
        let query = gen_query(&mut rng, d);

        group.bench_with_input(BenchmarkId::new("vectors", n), &n, |bench, _| {
            bench.iter(|| {
                // Build
                let graph = build_hnsw(&data, &config).unwrap();

                // Write to buffer
                let mut buf = Vec::new();
                write_hnsw_standard(&mut buf, &graph).unwrap();

                // Read back
                let mut cursor = std::io::Cursor::new(&buf);
                let loaded = read_hnsw_index(&mut cursor).unwrap();

                // Search
                let flat_vectors: Vec<f32> = data.iter().copied().collect();
                let params = SearchParams {
                    ef_search: 64,
                    ..Default::default()
                };
                let _ = search_hnsw(&loaded, &query, top_k, &flat_vectors, &params);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_distance,
    bench_hnsw_build,
    bench_hnsw_search,
    bench_hnsw_search_recompute,
    bench_full_pipeline,
);
criterion_main!(benches);
