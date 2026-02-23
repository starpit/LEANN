# Rust vs Python (FAISS C++) Performance

Pure-Rust LEANN engine benchmarked against the Python LEANN backend (FAISS C++ with SWIG bindings). HNSW benchmarks use d=384, M=32, efConstruction=200 unless noted.

## Latest Results (2026-02-23)

**Geometric mean speedup: 5.53x. Rust faster in 42/44 benchmarks, 1 tie, 1 Python win.**

Verdict uses IQR (p25-p75) overlap: non-overlapping = clear winner, overlapping = ~Tie.

### Distance (raw SIMD vs NumPy)

| Metric | Rust p50 | Python p50 | Speedup |
|--------|----------|------------|---------|
| L2 d=128 | 8.5 ns | 1.7 us | **201x** |
| L2 d=384 | 23.6 ns | 1.8 us | **74x** |
| L2 d=768 | 52.1 ns | 2.0 us | **39x** |
| IP d=128 | 8.6 ns | 375 ns | **44x** |
| IP d=384 | 22.4 ns | 416 ns | **19x** |
| IP d=768 | 48.1 ns | 417 ns | **8.7x** |

### HNSW Build (parallel)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 100 | 272 us | 1.77 ms | **6.5x** |
| 1,000 | 12.4 ms | 21.0 ms | **1.7x** |
| 10,000 | 817 ms | 1.30 s | **1.6x** |
| 50,000 | 14.8 s | 20.0 s | **1.4x** |

### HNSW Search (stored vectors, 10K vectors, top_k=10)

| ef | Rust p50 | Python p50 | Verdict |
|----|----------|------------|---------|
| 16 | 43.3 us | 47.3 us | **Rust 1.1x** |
| 32 | 64.7 us | 74.5 us | **Rust 1.2x** |
| 64 | 109 us | 116 us | **Rust 1.1x** |
| 128 | 236 us | 287 us | ~Tie |
| 256 | 390 us | 444 us | **Rust 1.1x** |

### HNSW Search Recompute (callback, 10K vectors, top_k=10)

| ef | Rust p50 | Python p50 | Verdict |
|----|----------|------------|---------|
| 16 | 35.3 us | 46.5 us | **Rust 1.3x** |
| 32 | 66.5 us | 70.9 us | **Rust 1.1x** |
| 64 | 95.5 us | 116 us | **Rust 1.2x** |
| 128 | 195 us | 225 us | **Rust 1.2x** |
| 256 | 388 us | 439 us | **Rust 1.1x** |

### Text Chunking (sentence split + chunk with overlap)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 10 KB | 33.0 us | 557 us | **16.9x** |
| 100 KB | 311 us | 5.66 ms | **18.2x** |
| 1 MB | 3.08 ms | 56.6 ms | **18.4x** |

### Sentence Splitting

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 10 KB | 25.3 us | 557 us | **22.0x** |
| 100 KB | 236 us | 5.43 ms | **23.0x** |
| 1 MB | 2.35 ms | 55.0 ms | **23.4x** |

### BM25 Fit (index build)

| Size | Rust p50 | Python p50 | Verdict |
|------|----------|------------|---------|
| 1,000 | 5.27 ms | 4.98 ms | Python 1.1x |
| 10,000 | 51.9 ms | 55.8 ms | **Rust 1.1x** |

### BM25 Search

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 242 us | 1.37 ms | **5.7x** |
| 10,000 | 4.31 ms | 18.5 ms | **4.3x** |

### Metadata Filtering (compound filter)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 101 us | 271 us | **2.7x** |
| 10,000 | 1.25 ms | 2.68 ms | **2.1x** |

### Index I/O Write (graph serialization)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 7.0 us | 753 us | **108x** |
| 10,000 | 245 us | 3.79 ms | **15.5x** |

### Index I/O Read (graph deserialization)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 8.7 us | 102 us | **11.8x** |
| 10,000 | 106 us | 1.91 ms | **18.0x** |

### Passage Write (JSONL + offset index)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 599 us | 12.8 ms | **21.3x** |
| 10,000 | 5.06 ms | 122 ms | **24.1x** |

### Passage Lookup (seek + JSON parse, per-call)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 1.5 us | 3.8 us | **Rust 2.6x** |
| 10,000 | 1.5 us | 4.0 us | **Rust 2.7x** |

### Full Pipeline (build + write + read + search)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 100 | 352 us | 2.37 ms | **6.7x** |
| 1,000 | 11.9 ms | 24.4 ms | **2.1x** |
| 10,000 | 823 ms | 1.28 s | **1.6x** |

### CLI Startup

| | Rust p50 | Python p50 | Speedup |
|--|----------|------------|---------|
| `leann --help` | 8.0 ms | 37.2 ms | **4.6x** |

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
- **Buffered I/O**: Passage and index writes use `BufWriter` with manual offset tracking to minimize syscalls
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
cargo build --release -p leann-cli  # needed for cli_startup benchmark
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

Benchmarks exercise the HNSW core (build, search, SIMD distance, I/O), BM25 scoring, text chunking, and metadata filtering. They do not use chat, embedding providers, or the ZMQ server. The features required beyond the bare minimum are `parallel` (for `build_hnsw_with_pool`) and `bm25` (for `BM25Scorer`):

```bash
# Minimal feature set for benchmarks — skips compiling reqwest, tokio, zeromq, etc.
cargo bench --package leann-core --no-default-features --features parallel,bm25
```

With default features (which include both `parallel` and `bm25`) the benchmarks compile and run identically; the flag just trims ~200 transitive crates from the build.

### Benchmark groups

| Group | What it measures |
|-------|-----------------|
| `distance` | SIMD (NEON/AVX2) L2 and inner product at dims 128, 384, 768 |
| `hnsw_build` | Graph construction at 100, 1K, 10K, 50K vectors (M=32, efConstruction=200) |
| `hnsw_search` | Stored-vector search at ef_search 16/32/64/128/256 (10K vectors, top_k=10) |
| `hnsw_search_recompute` | Recompute-mode search with in-memory callback at same ef values |
| `full_pipeline` | Build + write + read + search at 100, 1K, 10K vectors |
| `passage_lookup` | Random passage retrieval (seek + JSON parse) at 1K, 10K passages |
| `text_chunking` | Sentence-splitting + chunking with overlap at 10KB, 100KB, 1MB text |
| `sentence_split` | Sentence boundary detection at 10KB, 100KB, 1MB text |
| `bm25_fit` | BM25 index build (tokenize + compute statistics) at 1K, 10K documents |
| `bm25_search` | BM25 query scoring at 1K, 10K documents |
| `metadata_filter` | Compound metadata filter evaluation at 1K, 10K results |
| `index_io_write` | HNSW graph serialization at 1K, 10K vectors |
| `index_io_read` | HNSW graph deserialization at 1K, 10K vectors |
| `passage_write` | Passage JSONL + offset index file write at 1K, 10K passages |
| `cli_startup` | Process startup time for `leann --help` (subprocess wall-clock) |

### Notes

- Search benchmarks at low ef (16, 32) have high variance (~30%). ef 64+ is much more stable.
- Build n=100 shows the largest Rust advantage (6.5x) because FAISS has higher fixed overhead.
- BM25 fit at 1K is a ~tie because the workload is dominated by regex tokenization and hash-map building, which are similarly optimized in both languages.
- BM25 search is 4-6x faster in Rust due to tighter iteration over the scored document set.
- Index size comparison reflects compact CSR format (Rust) vs FAISS flat format (Python). Both store the same graph; Rust strips padding and uses a denser layout.
- Results will vary by machine. The numbers above were collected on Apple M-series (aarch64/NEON).
