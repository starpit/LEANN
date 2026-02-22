# Rust vs Python (FAISS C++) HNSW Performance

Pure-Rust HNSW engine benchmarked against the Python LEANN backend (FAISS C++ with SWIG bindings). All benchmarks use d=384, M=32, efConstruction=200 unless noted.

## Latest Results (2026-02-21)

**Geometric mean speedup: 3.31x. Rust faster in 19/25 benchmarks, 3 ties, 3 Python wins.**

Verdict uses IQR (p25-p75) overlap: non-overlapping = clear winner, overlapping = ~Tie.

### Distance (raw SIMD vs FAISS)

| Metric | Rust p50 | Python p50 | Speedup |
|--------|----------|------------|---------|
| L2 d=128 | 14.6 ns | 1.6 us | **111x** |
| L2 d=384 | 24.0 ns | 1.8 us | **73x** |
| L2 d=768 | 50.3 ns | 2.0 us | **39x** |
| IP d=128 | 10.3 ns | 416 ns | **40x** |
| IP d=384 | 21.9 ns | 416 ns | **19x** |
| IP d=768 | 45.2 ns | 417 ns | **9.2x** |

### HNSW Build (parallel)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 100 | 307 us | 1.79 ms | **5.8x** |
| 1,000 | 11.4 ms | 20.3 ms | **1.8x** |
| 10,000 | 719 ms | 1.14 s | **1.6x** |
| 50,000 | 12.8 s | 18.2 s | **1.4x** |

### HNSW Search (stored vectors, 10K vectors, top_k=10)

| ef | Rust p50 | Python p50 | Verdict |
|----|----------|------------|---------|
| 16 | 43.0 us | 39.6 us | Python 1.1x |
| 32 | 64.5 us | 63.2 us | Python 1.0x |
| 64 | 111 us | 123 us | **Rust 1.1x** |
| 128 | 233 us | 236 us | ~Tie |
| 256 | 436 us | 499 us | **Rust 1.1x** |

### HNSW Search Recompute (callback, 10K vectors, top_k=10)

| ef | Rust p50 | Python p50 | Verdict |
|----|----------|------------|---------|
| 16 | 36.8 us | 39.5 us | **Rust 1.1x** |
| 32 | 68.5 us | 63.1 us | Python 1.1x |
| 64 | 100 us | 124 us | **Rust 1.2x** |
| 128 | 250 us | 247 us | ~Tie |
| 256 | 465 us | 438 us | ~Tie |

### Passage Lookup (seek + JSON parse, per-call)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 1.5 us | 3.8 us | **Rust 2.5x** |
| 10,000 | 1.4 us | 4.0 us | **Rust 2.8x** |

### Full Pipeline (build + write + read + search)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 100 | 354 us | 2.23 ms | **6.3x** |
| 1,000 | 11.2 ms | 21.8 ms | **1.9x** |
| 10,000 | 711 ms | 1.18 s | **1.7x** |

### Index Size (compact CSR vs FAISS flat)

| Size | Rust | Python (FAISS) | Ratio |
|------|------|----------------|-------|
| 100 | 27.3 KB | 180.9 KB | **0.15x** |
| 1,000 | 271.2 KB | 1.8 MB | **0.15x** |
| 10,000 | 2.7 MB | 18.1 MB | **0.15x** |

## Key Optimizations

The Rust HNSW engine uses several techniques to match or exceed FAISS C++ performance:

- **SIMD distance**: NEON (aarch64) and AVX2 (x86_64) implementations of L2 and inner product, with batch-4 variants that load the query vector once and compute against 4 database vectors simultaneously
- **Monomorphized distance**: Distance functions are generic (not function pointers), enabling full inlining at each call site
- **Parallel build**: Rayon thread pool with lock-free reverse connections (CAS)
- **Flat heaps**: `FlatMinHeap`/`FlatMaxHeap` replace `BinaryHeap` for cache-friendly candidate management in search and build hot loops
- **VisitedList**: Generation-counter visited set for O(1) reset between searches (no HashSet reallocation)
- **Early termination**: Build stops neighbor search when improvement is unlikely
- **Cached file handles**: PassageManager opens passage files once at load time and reuses handles for all lookups (seek + read, no open/close per call)

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

### Feature flags for benchmarking

Benchmarks only exercise the HNSW core (build, search, SIMD distance, I/O) and do not use chat, embedding providers, BM25, or the ZMQ server. The only feature required beyond the bare minimum is `parallel` (for `build_hnsw_with_pool`):

```bash
# Minimal feature set for benchmarks — skips compiling reqwest, tokio, zeromq, etc.
cargo bench --package leann-core --no-default-features --features parallel
```

With default features the benchmarks compile and run identically; the flag just trims ~200 transitive crates from the build.

### Benchmark groups

| Group | What it measures |
|-------|-----------------|
| `distance` | SIMD (NEON/AVX2) L2 and inner product at dims 128, 384, 768 |
| `hnsw_build` | Graph construction at 100, 1K, 10K, 50K vectors (M=32, efConstruction=200) |
| `hnsw_search` | Stored-vector search at ef_search 16/32/64/128/256 (10K vectors, top_k=10) |
| `hnsw_search_recompute` | Recompute-mode search with in-memory callback at same ef values |
| `full_pipeline` | Build + write + read + search at 100, 1K, 10K vectors |
| `passage_lookup` | Random passage retrieval (seek + JSON parse) at 1K, 10K passages |

### Notes

- Search benchmarks at low ef (16, 32) have high variance (~30%). ef 64+ is much more stable.
- Build n=100 shows the largest Rust advantage (5.8x) because FAISS has higher fixed overhead.
- Index size comparison reflects compact CSR format (Rust) vs FAISS flat format (Python). Both store the same graph; Rust strips padding and uses a denser layout.
- Results will vary by machine. The numbers above were collected on Apple M-series (aarch64/NEON).
