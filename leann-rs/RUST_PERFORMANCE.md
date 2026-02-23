# Rust vs Python (FAISS C++) Performance

Pure-Rust LEANN engine benchmarked against the Python LEANN backend (FAISS C++ with SWIG bindings). HNSW benchmarks use d=384, M=32, efConstruction=200 unless noted.

## Latest Results (2026-02-23)

**Geometric mean speedup: 6.07x. Rust faster in 40/44 benchmarks, 4 ties, 0 Python wins.**

Verdict uses IQR (p25-p75) overlap: non-overlapping = clear winner, overlapping = ~Tie.

### Distance (raw SIMD vs NumPy)

| Metric | Rust p50 | Python p50 | Speedup |
|--------|----------|------------|---------|
| L2 d=128 | 8.3 ns | 1.7 us | **206x** |
| L2 d=384 | 24.2 ns | 1.8 us | **72x** |
| L2 d=768 | 51.5 ns | 2.0 us | **40x** |
| IP d=128 | 8.4 ns | 375 ns | **45x** |
| IP d=384 | 23.8 ns | 416 ns | **18x** |
| IP d=768 | 47.1 ns | 417 ns | **8.9x** |

### HNSW Build (parallel)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 100 | 223 us | 1.77 ms | **7.9x** |
| 1,000 | 11.3 ms | 21.0 ms | **1.9x** |
| 10,000 | 759 ms | 1.30 s | **1.7x** |
| 50,000 | 13.4 s | 20.0 s | **1.5x** |

### HNSW Search (stored vectors, 10K vectors, top_k=10)

| ef | Rust p50 | Python p50 | Verdict |
|----|----------|------------|---------|
| 16 | 43.4 us | 47.3 us | **Rust 1.1x** |
| 32 | 65.2 us | 74.5 us | **Rust 1.1x** |
| 64 | 116 us | 116 us | ~Tie |
| 128 | 272 us | 287 us | ~Tie |
| 256 | 428 us | 444 us | ~Tie |

### HNSW Search Recompute (callback, 10K vectors, top_k=10)

| ef | Rust p50 | Python p50 | Verdict |
|----|----------|------------|---------|
| 16 | 35.2 us | 46.5 us | **Rust 1.3x** |
| 32 | 66.4 us | 70.9 us | **Rust 1.1x** |
| 64 | 96.4 us | 116 us | **Rust 1.2x** |
| 128 | 200 us | 225 us | **Rust 1.1x** |
| 256 | 474 us | 439 us | ~Tie |

### Text Chunking (sentence split + chunk with overlap)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 10 KB | 22.0 us | 557 us | **25.3x** |
| 100 KB | 204 us | 5.66 ms | **27.7x** |
| 1 MB | 2.10 ms | 56.6 ms | **27.0x** |

### Sentence Splitting

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 10 KB | 13.1 us | 557 us | **42.5x** |
| 100 KB | 136 us | 5.43 ms | **39.9x** |
| 1 MB | 1.33 ms | 55.0 ms | **41.3x** |

### BM25 Fit (index build)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 3.16 ms | 4.98 ms | **Rust 1.6x** |
| 10,000 | 32.5 ms | 55.8 ms | **Rust 1.7x** |

### BM25 Search

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 228 us | 1.37 ms | **6.0x** |
| 10,000 | 3.51 ms | 18.5 ms | **5.3x** |

### Metadata Filtering (compound filter)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 115 us | 271 us | **2.4x** |
| 10,000 | 1.28 ms | 2.68 ms | **2.1x** |

### Index I/O Write (graph serialization)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 6.9 us | 753 us | **110x** |
| 10,000 | 241 us | 3.79 ms | **15.8x** |

### Index I/O Read (graph deserialization)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 9.2 us | 102 us | **11.2x** |
| 10,000 | 103 us | 1.91 ms | **18.6x** |

### Passage Write (JSONL + offset index)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 560 us | 12.8 ms | **22.9x** |
| 10,000 | 5.03 ms | 122 ms | **24.2x** |

### Passage Lookup (seek + JSON parse, per-call)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 1,000 | 1.5 us | 3.8 us | **Rust 2.6x** |
| 10,000 | 1.5 us | 4.0 us | **Rust 2.6x** |

### Full Pipeline (build + write + read + search)

| Size | Rust p50 | Python p50 | Speedup |
|------|----------|------------|---------|
| 100 | 342 us | 2.37 ms | **6.9x** |
| 1,000 | 12.5 ms | 24.4 ms | **2.0x** |
| 10,000 | 860 ms | 1.28 s | **1.5x** |

### CLI Startup

| | Rust p50 | Python p50 | Speedup |
|--|----------|------------|---------|
| `leann --help` | 7.3 ms | 37.2 ms | **5.1x** |

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
- **Allocation-light BM25**: Single-pass tokenizer (no regex) builds frequency counts inline, reusing the token buffer for repeated words; search uses `&str` references and only clones the top-k result IDs
- **Byte-level sentence splitting**: Iterates `char_indices` with byte-level boundary detection instead of allocating a `Vec<char>` copy of the input
- **Cow<str> metadata filters**: `value_to_string` returns `Cow<str>`, borrowing for String/Bool/Null values; `matches_metadata` evaluates filters directly on `SearchResult.metadata` without round-tripping through HashMap
- **Buffered I/O**: Passage, index, and ID-map writes use `BufWriter` with manual offset tracking to minimize syscalls
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
- Build n=100 shows the largest Rust advantage (7.9x) because FAISS has higher fixed overhead.
- BM25 fit uses single-pass char-level tokenization (no regex) with buffer reuse for repeated tokens, giving 1.6-1.7x over Python.
- BM25 search is 5-6x faster in Rust due to reference-based scoring (only top-k IDs are cloned) and tighter iteration.
- Sentence splitting gained ~1.8x from eliminating `Vec<char>` allocation in favor of byte-level `char_indices` iteration.
- Index size comparison reflects compact CSR format (Rust) vs FAISS flat format (Python). Both store the same graph; Rust strips padding and uses a denser layout.
- Results will vary by machine. The numbers above were collected on Apple M-series (aarch64/NEON).
