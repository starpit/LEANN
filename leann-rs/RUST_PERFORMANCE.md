# Rust vs Python (FAISS C++) HNSW Performance

Pure-Rust HNSW engine benchmarked against the Python LEANN backend (FAISS C++ with SWIG bindings). All benchmarks use d=384, M=32, efConstruction=200 unless noted.

## Latest Results (2026-02-20)

**Geometric mean speedup: 3.41x. Rust faster in 18/23 benchmarks, 5 ties, 0 Python wins.**

Verdict uses IQR (p25-p75) overlap: non-overlapping = clear winner, overlapping = ~Tie.

### Distance (raw SIMD vs FAISS)

| Metric | Rust p50 | Python p50 | Speedup |
|--------|----------|------------|---------|
| L2 d=128 | 8.2 ns | 1.7 us | **204x** |
| L2 d=384 | 24.8 ns | 1.8 us | **71x** |
| L2 d=768 | 50.7 ns | 1.9 us | **38x** |
| IP d=128 | 8.3 ns | 417 ns | **50x** |
| IP d=384 | 23.3 ns | 416 ns | **18x** |
| IP d=768 | 46.7 ns | 458 ns | **10x** |

### HNSW Build (parallel)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 100 | 373 us | 1.80 ms | **4.8x** |
| 1,000 | 12.7 ms | 20.9 ms | **1.6x** |
| 10,000 | 784 ms | 1.32 s | **1.7x** |
| 50,000 | 14.0 s | 19.7 s | **1.4x** |

### HNSW Search (stored vectors, 10K vectors, top_k=10)

| ef | Rust p50 | Python p50 | Verdict |
|----|----------|------------|---------|
| 16 | 36.0 us | 35.7 us | ~Tie |
| 32 | 60.7 us | 63.2 us | Rust 1.04x |
| 64 | 108 us | 121 us | Rust 1.1x |
| 128 | 243 us | 252 us | ~Tie |
| 256 | 445 us | 438 us | ~Tie |

### HNSW Search Recompute (callback, 10K vectors, top_k=10)

| ef | Rust p50 | Python p50 | Verdict |
|----|----------|------------|---------|
| 16 | 33.5 us | 35.3 us | Rust 1.1x |
| 32 | 58.1 us | 62.2 us | Rust 1.1x |
| 64 | 98.8 us | 115 us | Rust 1.2x |
| 128 | 250 us | 225 us | ~Tie |
| 256 | 410 us | 462 us | ~Tie |

### Full Pipeline (build + write + read + search)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 100 | 515 us | 2.52 ms | **4.9x** |
| 1,000 | 14.3 ms | 22.1 ms | **1.5x** |
| 10,000 | 780 ms | 1.31 s | **1.7x** |

### Index Size (compact CSR vs FAISS flat)

| Size | Rust | Python (FAISS) | Ratio |
|------|------|----------------|-------|
| 100 | 27.4 KB | 180.9 KB | **0.15x** |
| 1,000 | 271.6 KB | 1.8 MB | **0.15x** |
| 10,000 | 2.7 MB | 18.1 MB | **0.15x** |

## Key Optimizations

The Rust HNSW engine uses several techniques to match or exceed FAISS C++ performance:

- **SIMD distance**: NEON (aarch64) and AVX2 (x86_64) implementations of L2 and inner product, with batch-4 variants that load the query vector once and compute against 4 database vectors simultaneously
- **Monomorphized distance**: Distance functions are generic (not function pointers), enabling full inlining at each call site
- **Parallel build**: Rayon thread pool with lock-free reverse connections (CAS)
- **Flat heaps**: `FlatMinHeap`/`FlatMaxHeap` replace `BinaryHeap` for cache-friendly candidate management in search and build hot loops
- **VisitedList**: Generation-counter visited set for O(1) reset between searches (no HashSet reallocation)
- **Early termination**: Build stops neighbor search when improvement is unlikely

## How to Reproduce

### Quick: all-in-one script

From the repository root:

```bash
bash benchmarks/compare_rust_python.sh

# Skip 50K-vector benchmarks for faster runs (~5 min instead of ~15 min)
SKIP_LARGE=1 bash benchmarks/compare_rust_python.sh
```

This runs both Rust and Python benchmarks and prints a comparison table.

### Manual: step by step

```bash
# 1. Generate Rust results (from leann-rs/)
cd leann-rs
mkdir -p ../benchmarks/results
cargo bench --package leann-core --bench bench_json_output > ../benchmarks/results/rust_results.json

# 2. Generate Python (FAISS C++) results (from repo root)
cd ..
uv run python benchmarks/rust_vs_python.py --json > benchmarks/results/python_results.json

# 3. Compare
uv run python benchmarks/compare_results.py
```

### Criterion benchmarks only (no Python comparison)

```bash
# Run all Criterion benchmarks (HTML reports in target/criterion/)
cargo bench --package leann-core

# Run a specific benchmark group
cargo bench --package leann-core -- "distance"
cargo bench --package leann-core -- "hnsw_build"
cargo bench --package leann-core -- "hnsw_search"
```

### Benchmark groups

| Group | What it measures |
|-------|-----------------|
| `distance` | SIMD (NEON/AVX2) L2 and inner product at dims 128, 384, 768 |
| `hnsw_build` | Graph construction at 100, 1K, 10K, 50K vectors (M=32, efConstruction=200) |
| `hnsw_search` | Stored-vector search at ef_search 16/32/64/128/256 (10K vectors, top_k=10) |
| `hnsw_search_recompute` | Recompute-mode search with in-memory callback at same ef values |
| `full_pipeline` | Build + write + read + search at 100, 1K, 10K vectors |

### Notes

- Search benchmarks at low ef (16, 32) have high variance (~30%). ef 64+ is much more stable.
- Build n=100 shows the largest Rust advantage (4.8x) because FAISS has higher fixed overhead.
- Index size comparison reflects compact CSR format (Rust) vs FAISS flat format (Python). Both store the same graph; Rust strips padding and uses a denser layout.
- Results will vary by machine. The numbers above were collected on Apple M-series (aarch64/NEON).
