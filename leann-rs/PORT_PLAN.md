# LEANN Rust Port Plan

> **Rust port currently based on Python commit [`1da64a9`](https://github.com/nickmccarty/LEANN/commit/1da64a9) (`main` as of 2026-02-20).** When updating the Rust code to match future Python changes, diff from this commit forward.

## Current Status (2026-02-20)

**11,300+ lines of Rust across 4 crates. 157 tests passing (64 unit + 82 integration + 11 CLI/server). 0 errors, 0 warnings.**

All 8 phases of the initial implementation are complete. Since then, major HNSW performance work has been done: SIMD-optimized distance functions (NEON/AVX2), batch-4 distance computation, parallel build with rayon thread pools, flat heaps for cache locality, early termination, and a `VisitedList` with generation-counter reset. The pure-Rust HNSW engine now matches or approaches FAISS C++ performance. Criterion benchmark suite and Rust-vs-Python comparison scripts validate this.

What remains is hardening: ONNX Runtime activation, Python example porting, CI setup, and remaining integration test gaps.

### Crate Status

| Crate | LOC | Status |
|-------|-----|--------|
| `leann-core` | 8,959 | Complete - all modules implemented with 64 unit tests; extensive HNSW optimization |
| `leann-cli` | 1,406 | Complete - all 8 commands wired up, aligned with Python CLI |
| `leann-server` | 420 | Complete - all endpoints functional with state management |
| `leann-python` | 349 | Complete - compiles with PyO3 0.25 (needs maturin build test) |

### File Inventory

```
leann-rs/
  Cargo.toml                         # workspace root
  crates/
    leann-core/benches/
      hnsw_benchmarks.rs       (236)  # Criterion benchmarks: distance, build, search, recompute, pipeline
      bench_json_output.rs     (396)  # Standalone JSON output benchmark (quantiles, RNG seed, build+search)
    leann-core/src/
      lib.rs                    (24)  # Module declarations + re-exports
      search_result.rs          (72)  # SearchResult struct [3 tests]
      settings.rs              (150)  # Env var resolution [4 tests]
      index.rs                 (252)  # IndexMeta, IndexPaths, DistanceMetric [5 tests]
      metadata_filter.rs       (387)  # 13 operators, AND logic [7 tests]
      passages.rs              (626)  # PassageManager, JSONL I/O, pickle parser [3 tests]
      bm25.rs                  (202)  # BM25Scorer [4 tests]
      builder.rs               (380)  # LeannBuilder (build + from_embeddings)
      searcher.rs              (339)  # LeannSearcher (vector/BM25/grep/hybrid search)
      chat.rs                  (274)  # LlmProvider trait + Ollama/OpenAI/Anthropic/Simulated
      react_agent.rs           (246)  # ReAct agent [2 tests]
      sync.rs                  (279)  # MerkleTree + FileSynchronizer [2 tests]
      hnsw/
        mod.rs                   (9)
        graph.rs               (190)  # HnswGraph, GraphStorage, VectorStorage [2 tests]
        simd.rs              (1,018)  # NEON/AVX2 SIMD distance (L2, IP), batch-4, FlatMinHeap/FlatMaxHeap, VisitedList [11 tests]
        build.rs             (1,053)  # Parallel HNSW build (rayon), early termination, monomorphized distance [3 tests]
        search.rs              (770)  # Beam search + recompute, SIMD batch-4 distance, flat heaps [2 tests]
        csr.rs                 (150)  # CSR conversion + embedding pruning [1 test]
        io.rs                  (408)  # Binary format read/write (compact + standard) [1 test]
      embedding/
        mod.rs                  (54)  # EmbeddingProvider trait, EmbeddingMode enum
        client.rs              (161)  # ZMQ REQ client (text/distance/by-id)
        server.rs              (225)  # ZMQ REP server (3 request types)
        manager.rs             (215)  # Subprocess lifecycle, port allocation
        openai.rs              (107)  # OpenAI embedding API
        ollama.rs              (259)  # Ollama embedding API (pipelined async)
        gemini.rs              (100)  # Gemini batch embedding API
        onnx.rs                (102)  # ONNX Runtime scaffold (not yet activated)
      chunking/
        mod.rs                  (73)  # chunk_text with sentence overlap [2 tests]
        sentence.rs            (120)  # Sentence splitter [4 tests]
        ast.rs                 (495)  # Python/Rust/JS/TS code chunking [4 tests]
      document_loaders/
        mod.rs                  (67)  # extract_text dispatcher, is_binary_document
        pdf.rs                  (80)  # PDF text extraction via pdf-extract [3 tests]
    leann-cli/src/
      main.rs                (1,406)  # clap CLI: build/search/ask/react/list/remove/watch/serve
    leann-server/src/
      main.rs                  (420)  # Axum server: health/indexes/search/info endpoints
    leann-python/
      Cargo.toml                      # Standalone (excluded from workspace, built via maturin)
      pyproject.toml                  # maturin config
      src/lib.rs               (349)  # PyO3: LeannBuilder/Searcher/Chat/ReActAgent/SearchResult
      python/leann/__init__.py        # Re-exports
```

---

## Context

LEANN is a ~15K LOC Python vector database/RAG system with a custom C++ FAISS fork for HNSW graph search with ZMQ-based embedding recomputation. The goal is a full rewrite in Rust with Python bindings (PyO3), keeping Python example apps that consume the bindings.

### Current Architecture Summary

| Layer | Files | LOC | Key Dependencies |
|-------|-------|-----|-----------------|
| Core API (`api.py`) | LeannBuilder, LeannSearcher, LeannChat, BM25Scorer, PassageManager | ~1500 | numpy, zmq, msgpack, pickle |
| CLI (`cli.py`) | `leann build/search/ask/list/remove/watch` | ~2200 | typer, llama_index (chunking), tqdm |
| Embedding compute (`embedding_compute.py`) | sentence-transformers, OpenAI, Ollama, MLX, Gemini providers | ~1330 | torch, sentence-transformers, openai, tiktoken, requests |
| LLM chat (`chat.py`) | Ollama, HF, OpenAI, Anthropic, Gemini chat backends | ~1000 | torch, transformers, openai, anthropic, requests |
| Backend: HNSW (`hnsw_backend.py`) | Builder/Searcher wrapping custom FAISS C++ fork | ~290 | Custom FAISS (C++ w/ SWIG), numpy |
| Backend: DiskANN (`diskann_backend.py`) | Builder/Searcher wrapping custom DiskANN C++ | ~470 | Custom DiskANN (C++ pybind11), numpy, psutil |
| HNSW embedding server (`hnsw_embedding_server.py`) | ZMQ REP server for on-demand recompute | ~505 | zmq, msgpack, numpy |
| Graph pruning (`convert_to_csr.py`) | HNSW graph CSR conversion, embedding pruning | ~1050 | numpy, struct |
| Embedding server manager (`embedding_server_manager.py`) | Subprocess lifecycle for embedding servers | ~575 | subprocess, zmq, socket |
| Searcher base (`searcher_base.py`) | Shared searcher logic, ZMQ client | ~220 | zmq, msgpack, numpy |
| Metadata filter (`metadata_filter.py`) | Filter engine with comparison/membership/string ops | ~240 | (stdlib only) |
| ReAct agent (`react_agent.py`) | Multi-turn reasoning agent | ~290 | (uses LeannSearcher + LLMInterface) |
| HTTP server (`server.py`) | FastAPI server for HTTP search | ~200 | fastapi, uvicorn, pydantic |
| Sync (`sync.py`) | Merkle tree file change detection | ~160 | llama_index, pickle, hashlib |
| Chunking (`chunking_utils.py`) | AST-aware + traditional text chunking | ~420 | llama_index, astchunk, tiktoken |
| Settings (`settings.py`) | URL/API key resolution from env | ~100 | (stdlib only) |

### Key Protocols
- **ZMQ Embedding Recompute**: REQ/REP over msgpack. Three message types: text embedding (list of strings), distance calculation ([[ids], [query_vec]]), embedding-by-id ([[ids]])
- **Index file format**: `<name>.meta.json` + `<name>.passages.jsonl` + `<name>.passages.idx` (pickle offset map) + `<name>.index` (FAISS/DiskANN binary)
- **Backend plugin system**: Auto-discovery via `leann-backend-*` package naming + `@register_backend` decorator

---

## Phase 1: Rust Workspace Setup & Core Data Structures [COMPLETE]

### Crate structure — implemented as planned.

### Key dependencies (Cargo.toml) — all resolved:
```toml
[workspace.dependencies]
serde, serde_json, bincode, tokio, anyhow, thiserror, tracing, tracing-subscriber,
ndarray, rand, zeromq, rmp-serde, reqwest (blocking+json), clap, rayon, memmap2,
axum, tower, tower-http, tiktoken-rs, sha2, regex, uuid, indicatif, notify, libc
```

PyO3 0.25 used for leann-python (standalone crate, Python 3.14 compatible).

---

## Phase 2: HNSW Graph Engine (Pure Rust) [COMPLETE + OPTIMIZED]

### 2a. HNSW Data Structures (`hnsw/graph.rs`) [COMPLETE]
- `HnswGraph` with `GraphStorage` enum: Standard (offsets/neighbors) and Compact (level_ptr/node_offsets/neighbors)
- `VectorStorage` enum: Null (pruned) and Raw (with fourcc + data bytes)
- `HnswConfig`: M, efConstruction, efSearch, distance metric, compact/recompute flags
- FourCC constants: `FOURCC_HNSW_FLAT = "IHNf"`, `FOURCC_NULL = "null"`

### 2b. SIMD Distance Functions (`hnsw/simd.rs`) [COMPLETE]
- **Platform-specific SIMD**: NEON (aarch64) and AVX2 (x86_64) implementations with scalar fallback
- **L2 distance**: `l2_distance(a, b)` and `l2_distance_batch_4(query, db0, db1, db2, db3)` — query loaded once, reused against 4 database vectors
- **Inner product distance**: `inner_product_distance(a, b)` and `inner_product_distance_batch_4`
- **L2 normalization**: `normalize_l2` for cosine distance support
- **FlatMinHeap / FlatMaxHeap**: Cache-friendly flat-array heaps replacing `BinaryHeap` for better SIMD performance in search/build hot loops
- **VisitedList**: Generation-counter visited set for O(1) reset between searches (avoids `HashSet` allocation)
- 11 tests: L2, IP, batch-4, non-aligned, large vectors, normalization, visited list operations

### 2c. HNSW Build (`hnsw/build.rs`) [COMPLETE + OPTIMIZED]
- FAISS-style `IndexHNSW::add`: exponential level assignment, greedy insert, bidirectional neighbor connections
- **Parallel build**: `build_hnsw_with_pool` using rayon `ThreadPool` with configurable thread count
- **Monomorphized distance**: Distance functions use generics (not function pointers) for full inlining
- **Early termination**: Stops neighbor search when improvement is unlikely
- **RNG seed support**: Deterministic builds via configurable seed
- Worst-neighbor eviction when neighbor count exceeds M
- MIPS, L2, and Cosine distance metrics
- 3 tests: small graph build, parallel small graph, parallel larger graph

### 2d. HNSW Search with Recompute (`hnsw/search.rs`) [COMPLETE + OPTIMIZED]
- Two-phase beam search: top-down greedy from max level, ef-search at level 0
- **SIMD batch-4 distance**: Processes 4 neighbor candidates at once using batch distance functions
- **Flat heaps**: Uses `FlatMinHeap`/`FlatMaxHeap` for cache-friendly candidate management
- **VisitedList**: Generation-counter visited tracking (O(1) reset instead of HashSet realloc)
- `search_hnsw` (stored vectors) and `search_hnsw_recompute` (callback-based distance computation)
- `SearchParams`: ef_search, beam_size, prune_ratio, pruning_strategy, batch_size
- `PruningStrategy` enum: Global, Local, Proportional
- 2 tests: stored vector search + recompute callback search

### 2d. CSR Format (`hnsw/csr.rs`) [COMPLETE]
- `convert_to_csr`: strips -1 padding, builds level_ptr/node_offsets/neighbors arrays
- `prune_embeddings`: zeros vector storage
- 1 test: roundtrip conversion

### 2e. Index I/O (`hnsw/io.rs`) [COMPLETE]
- `read_hnsw_index`: reads both compact and standard formats (LE binary, 8-byte count vectors)
- `write_hnsw_compact` and `write_hnsw_standard`
- Compact flag probing, extra byte handling
- 1 test: compact roundtrip

---

## Phase 3: Passage Storage & Index Management [COMPLETE]

### 3a. PassageManager (`passages.rs`) [COMPLETE]
- JSONL read/write with byte-offset random access
- `write_passages`, `write_id_map`, `load_id_map`
- Multi-shard offset map support
- Python pickle parser (`parse_python_pickle_offset_map`) for protocols 2-4 backward compatibility
- Path resolution with metadata-relative fallbacks
- 3 tests

### 3b. Index Metadata (`index.rs`) [COMPLETE]
- `IndexMeta` with full serde for `.meta.json`
- `PassageSource`, `DistanceMetric` enum (Mips/L2/Cosine)
- `IndexPaths` for sibling file resolution
- 5 tests

### 3c. BM25 (`bm25.rs`) [COMPLETE]
- `BM25Scorer` with configurable k1/b parameters
- Regex-based tokenization (punctuation removal + lowercase)
- fit/score/search methods
- 4 tests

### 3d. Metadata Filter (`metadata_filter.rs`) [COMPLETE]
- 13 operators: ==, !=, <, <=, >, >=, in, not_in, contains, starts_with, ends_with, is_true, is_false
- AND logic for multiple filters
- Helper functions: value_to_f64, value_to_string, value_is_truthy
- 7 tests

---

## Phase 4: Embedding Infrastructure [COMPLETE]

### 4a. ZMQ Embedding Server (`embedding/server.rs`) [COMPLETE]
- ZMQ REP socket with 3 request types: text embedding, distance calculation, embedding-by-id
- Msgpack serialization, tokio timeout-based shutdown
- Distance computation for L2 and MIPS/Cosine

### 4b. ZMQ Client (`embedding/client.rs`) [COMPLETE]
- `EmbeddingClient` with `compute_text_embeddings`, `compute_distances`, `get_embeddings_by_id`
- zeromq async sockets with tokio runtime

### 4c. Server Manager (`embedding/manager.rs`) [COMPLETE]
- `EmbeddingServerManager`: subprocess lifecycle, port allocation, config signature-based reuse
- Graceful shutdown (SIGTERM then SIGKILL after timeout)

### 4d. ONNX Runtime Inference (`embedding/onnx.rs`) [SCAFFOLD]
- `OnnxEmbedding` struct with model path resolution
- Validates model directory (model.onnx / model_optimized.onnx + tokenizer.json)
- **Not yet activated** — returns error directing users to OpenAI/Ollama instead
- Requires `ort` crate integration to enable (tokenizer loading, session inference, mean-pooling)

### 4e. API Embedding Clients [COMPLETE]
- `openai.rs`: OpenAI embedding API via reqwest blocking, configurable base_url and dimensions
- `ollama.rs`: Ollama embedding API with pipelined async dispatch — up to 3 concurrent in-flight requests via `tokio::spawn` + `Semaphore` to keep the GPU saturated between batches (batch size 128, `reqwest::Client` async)
- `gemini.rs`: Gemini batch embedding API (batchEmbedContents endpoint)

---

## Phase 5: High-Level API (Builder / Searcher / Chat) [COMPLETE]

### 5a. LeannBuilder (`builder.rs`) [COMPLETE]
- Fluent builder: `with_m`, `with_ef_construction`, `with_distance_metric`, `with_compact`, `with_recompute`
- `build_index`: compute embeddings, build HNSW, optional CSR conversion, write passages/offset/idmap/meta
- `build_index_from_embeddings`: from pre-computed Array2<f32>
- L2 normalization for cosine distance

### 5b. LeannSearcher (`searcher.rs`) [COMPLETE]
- `open`: loads meta, passages, graph, id_map
- `search` / `search_with_params`: vector search with recompute via ZMQ
- BM25 search, grep search (regex), hybrid search (gemma weighting)
- Metadata filtering, passage enrichment
- `SearchConfig`: complexity, beam_width, prune_ratio, metadata_filters, batch_size, use_grep, gemma, zmq_port

### 5c. LeannChat (`chat.rs`) [COMPLETE]
- `LlmProvider` trait with `ask()` method
- `OllamaChat`, `OpenAiChat`, `AnthropicChat`, `SimulatedChat` implementations
- `get_llm` factory from `LlmConfig`
- `LeannChat`: wraps searcher + LLM, formats context prompt
- Note: `GeminiChat` not yet implemented (Gemini embedding works, but chat LLM provider pending)

### 5d. ReAct Agent (`react_agent.rs`) [COMPLETE]
- Multi-turn search-reason-answer loop
- Prompt construction with iteration tracking
- Response parsing: Thought/Action/Final Answer extraction
- Early stop on empty results, max iteration final synthesis
- 2 tests

---

## Phase 6: CLI (`leann-cli`) [COMPLETE]

### All commands implemented (aligned with Python CLI):
- `leann build [name] --docs <path>...` — Full pipeline: recursive document loading (30+ file extensions), sentence chunking with configurable size/overlap, embedding computation, HNSW build. Supports both files and directories via `--docs`.
- `leann search <name> <query>` — Vector search with configurable complexity, `--show-metadata`, `--embedding-prompt-template`
- `leann ask <name> [question]` — Single-shot RAG Q&A with `--thinking-budget` support
- `leann ask <name> --interactive` — Interactive REPL mode
- `leann react <name> <query>` — Multi-turn ReAct agent for complex retrieval
- `leann list` — Scans for `.meta.json` files in current project and app-format indexes
- `leann remove <name>` — Deletes index directory with confirmation (or `--force`)
- `leann watch <name>` — Reports file changes since last checkpoint using `FileSynchronizer`
- `leann serve [--host --port]` — Starts embedded axum server

### CLI alignment with Python (2026-02-19):
- Added global `-v/--verbose` and `-q/--quiet` flags (mutually exclusive)
- `--backend-name` now validates choices: `hnsw`, `diskann`
- Boolean flags use `--flag/--no-flag` pattern (matching Python's `BooleanOptionalAction`):
  - `--compact/--no-compact` (default: true)
  - `--recompute/--no-recompute` (default: true)
- Added `--ast-fallback-traditional` to build command
- Added `--thinking-budget` to ask command (`low`, `medium`, `high`)
- Fixed `--num-threads` passthrough (was parsed but not forwarded)
- `--docs` accepts both files and directories (e.g., `--docs ./src ./file.jsonl ./config.json`)

### Document loading [COMPLETE]
- Recursive directory walker with skip logic (.git, node_modules, target, __pycache__, venv)
- 30+ supported extensions: txt, md, rst, rs, py, js, jsx, ts, tsx, java, go, c, cpp, cc, cxx, h, hpp, rb, sh, bash, toml, yaml, yml, json, xml, html, htm, css, sql, r, lua, php, swift, kt, scala, ex, exs, pdf
- PDF text extraction via `pdf-extract` crate (behind `pdf` feature flag, enabled by default)
- `document_loaders` module dispatches between text-based and binary formats automatically

### Chunking [COMPLETE]
- Sentence-based chunking with configurable size and overlap
- AST-aware code chunking (heuristic-based, not tree-sitter):
  - Python: detects def/class/async def by indentation
  - Rust: detects fn/struct/enum/impl/trait by brace counting
  - JavaScript/TypeScript: detects function/class/arrow functions
  - Generic fallback for other languages

---

## Phase 7: PyO3 Python Bindings (`leann-python`) [COMPLETE]

### Exposed classes:
- `LeannBuilder(embedding_model, dimensions=None, embedding_mode, **kwargs)` — add_text, build_index
- `LeannSearcher(index_path, **kwargs)` — search with GIL release via `py.allow_threads()`
- `LeannChat(index_path, llm_config=None, **kwargs)` — ask
- `ReActAgent(index_path, max_iterations=5)` — run
- `SearchResult` — id, score, text, metadata (as Python dict)

### Build config:
- PyO3 0.25 (compatible with Python 3.14)
- Standalone crate (excluded from workspace, built via maturin)
- `pyproject.toml` with `maturin` build backend
- `python/leann/__init__.py` with re-exports

### Not yet done:
- [ ] Maturin end-to-end build test
- [ ] Python type stubs (.pyi files)
- [ ] Python example scripts (basic_demo.py, document_rag.py, code_rag.py)

---

## Phase 8: HTTP Server (`leann-server`) [COMPLETE]

### Endpoints:
- `GET /health` — status + version
- `GET /indexes` — list all with IndexInfo (name, model, dimensions, backend, total_passages)
- `GET /indexes/{name}` — single index info
- `POST /indexes/{name}/search` — vector search with optional complexity and grep mode
- Proper error responses with `ErrorResponse` type
- Configurable via `LEANN_INDEX_DIR` and `PORT` env vars
- `AppState` with `with_state()` for shared config

---

## End-to-End Test Plan

Maps the Python test suite (`tests/test_*.py`) to equivalent Rust integration tests. Tests live in `crates/leann-core/tests/` as Rust integration tests (separate from unit tests in `src/`). All tests that don't require an external embedding service use a `FakeEmbeddingProvider` that returns deterministic vectors (e.g., hash-based or sequential), avoiding any network dependency.

### Test File Layout

```
crates/leann-core/tests/         # 82 integration tests
  common/mod.rs              # ✅ Shared helpers: FakeEmbeddingProvider, temp_dir, sample docs
  test_build_search.rs       # ✅ 13 tests — Core pipeline: build → search → verify
  test_chat_pipeline.rs      # ✅ 5 tests — Build → search → chat with SimulatedChat
  test_hybrid_search.rs      # ✅ 7 tests — Vector + BM25 blending via gemma + grep search
  test_metadata_filtering.rs # ✅ 18 tests — All 13 operators + compound AND
  test_document_loading.rs   # ✅ 17 tests — Load txt/md/rs/py → chunk → AST chunking
  test_sync.rs               # ✅ 7 tests — Merkle tree + FileSynchronizer e2e
  test_index_format.rs       # ✅ 7 tests — Index file validation + format roundtrips
  test_bm25_search.rs        # ✅ 8 tests — Pure BM25 keyword search
  (planned) test_embedding_manager.rs  # Server lifecycle: start/reuse/restart
  (planned) test_react_agent.rs        # ReAct multi-turn agent with SimulatedChat

crates/leann-cli/tests/          # 7 tests
  test_cli_args.rs           # ✅ 7 tests — Help output, version, argument parsing
  (planned) test_cli_build_search.rs   # CLI subprocess: build + search e2e
  (planned) test_cli_list_remove.rs    # list + remove commands

crates/leann-server/tests/       # 4 tests
  test_server_endpoints.rs   # ✅ 4 tests — health, indexes, not_found, index info
```

### Shared Test Helpers (`common/mod.rs`)

```rust
/// Deterministic embedding provider for tests — no network, no model loading.
/// Maps each text to a unique vector using a simple hash → f32 conversion.
/// Supports configurable dimensions (default 128).
pub struct FakeEmbeddingProvider { dimensions: usize }

/// Generate N synthetic documents: "This is document {i} about topic {i % 5}"
/// with metadata: {"id": "{i}", "doc_num": i, "topic": "topic_{i%5}"}
pub fn sample_documents(n: usize) -> Vec<(String, serde_json::Value)>

/// Create a temp dir that auto-cleans on drop
pub fn temp_index_dir() -> tempfile::TempDir

/// Build a test index from sample docs and return (index_path, texts)
pub fn build_test_index(n_docs: usize, dir: &Path) -> (PathBuf, Vec<String>)
```

---

### E2E-1: Core Build & Search Pipeline (`test_build_search.rs`)
**Python source:** `test_basic.py::test_backend_basic`, `test_basic.py::test_large_index`

| Test | What it verifies |
|------|-----------------|
| `test_build_and_search_100_docs` | Build HNSW index from 100 synthetic docs with FakeEmbeddingProvider, search with top_k=5, verify: result count == 5, each result has non-empty text, scores are descending, results are SearchResult type |
| `test_build_and_search_1000_docs` | Same with 1000 docs and top_k=10 — verifies scaling doesn't break |
| `test_build_creates_expected_files` | After build, verify `.meta.json`, `.passages.jsonl`, `.passages.idx`, `.index` all exist and are non-empty |
| `test_meta_json_content` | Parse `.meta.json`, verify: backend_name == "hnsw", embedding_model matches, dimensions correct, has passage_sources |
| `test_search_empty_index` | Build from 0 docs, search → empty results (no crash) |
| `test_search_returns_relevant_results` | Build from docs with distinct topics, search for topic-specific query, verify top result matches the right topic (using FakeEmbeddingProvider with topic-clustered vectors) |
| `test_build_with_compact_csr` | Build with compact=true, recompute=true, verify .index is in compact format (smaller than standard) |
| `test_build_with_distance_metrics` | Build separate indexes with L2, Cosine, MIPS — all three succeed and return results on search |
| `test_search_top_k_bounds` | top_k=1 returns 1, top_k > n_docs returns n_docs |
| `test_build_from_precomputed_embeddings` | Use `build_index_from_embeddings` with an Array2<f32>, verify search works |

---

### E2E-2: Full RAG Pipeline — Build → Search → Chat (`test_readme_pipeline.rs`)
**Python source:** `test_readme_examples.py::test_readme_basic_example`, `test_readme_examples.py::test_llm_config_simulated`

| Test | What it verifies |
|------|-----------------|
| `test_build_search_chat_pipeline` | Build index, search for query, create LeannChat with SimulatedChat LLM, call `ask()`, verify non-empty response string |
| `test_chat_with_simulated_llm` | LeannChat with `LlmConfig { type: "simulated" }`, ask a question, verify response contains simulated text |
| `test_chat_context_includes_passages` | Verify the LLM receives a prompt that includes retrieved passage text (mock/intercept the LLM call) |

---

### E2E-3: Hybrid Search (`test_hybrid_search.rs`)
**Python source:** `test_hybrid_search.py`

Uses 10 diverse documents (animals, programming, weather, databases, cooking) with metadata `{"id": "{i}", "doc_num": i}`.

| Test | What it verifies |
|------|-----------------|
| `test_pure_vector_search` | `gemma=1.0` — results come from vector search only |
| `test_pure_keyword_search` | `gemma=0.0` — results come from BM25 only, matching keyword query |
| `test_hybrid_balanced` | `gemma=0.5` — returns results (non-empty) |
| `test_hybrid_vector_heavy` | `gemma=0.8` — results skew toward vector |
| `test_hybrid_keyword_heavy` | `gemma=0.2` — results skew toward BM25 |
| `test_hybrid_with_metadata_filter` | `gemma=0.5` + metadata filter `{"doc_num": {"<": 8}}` — all results have doc_num < 8 |
| `test_bm25_scores_keyword_match` | Pure BM25 search for "python programming", verify top results contain those terms |

---

### E2E-4: Metadata Filtering (`test_metadata_filtering.rs`)
**Python source:** `test_metadata_filtering.py`

Tests `MetadataFilterEngine` at the end-to-end level (applied to real SearchResults from a built index), complementing the existing 7 unit tests.

| Test | What it verifies |
|------|-----------------|
| `test_filter_equals` | `{"topic": {"==": "topic_0"}}` — only matching docs returned |
| `test_filter_not_equals` | `{"topic": {"!=": "topic_0"}}` — topic_0 excluded |
| `test_filter_less_than` | `{"doc_num": {"<": 5}}` — all results have doc_num < 5 |
| `test_filter_greater_equal` | `{"doc_num": {">=": 50}}` — all results have doc_num >= 50 |
| `test_filter_in` | `{"topic": {"in": ["topic_0", "topic_1"]}}` — only those two topics |
| `test_filter_not_in` | `{"topic": {"not_in": ["topic_0"]}}` — topic_0 excluded |
| `test_filter_contains` | `{"text": {"contains": "document 1"}}` — substring match |
| `test_filter_starts_with` | `{"topic": {"starts_with": "topic_"}}` — all pass |
| `test_filter_ends_with` | `{"topic": {"ends_with": "_0"}}` — only topic_0 |
| `test_filter_compound_and` | Multiple filters on different fields — AND logic, intersection |
| `test_filter_range` | `{"doc_num": {">=": 10, "<": 20}}` — range query |
| `test_filter_no_matches` | Filter that matches nothing → empty results |
| `test_filter_none_passthrough` | No filters → all results returned unmodified |

---

### E2E-5: Document Loading & Chunking (`test_document_loading.rs`)
**Python source:** `test_document_rag.py`, `test_astchunk_integration.py`

Uses real small test files created in a temp dir.

| Test | What it verifies |
|------|-----------------|
| `test_load_txt_file` | Create a .txt file, extract_text, verify content matches |
| `test_load_md_file` | Create a .md file, extract_text, verify content |
| `test_load_rs_file` | Create a .rs file, extract_text, verify content |
| `test_load_py_file` | Create a .py file, extract_text, verify content |
| `test_load_pdf_file` | Create a minimal PDF or use a small test PDF, extract_text, verify non-empty |
| `test_load_unsupported_extension` | .xyz file → error or empty |
| `test_build_from_directory` | Create temp dir with mixed .txt/.md/.rs files, build index from directory, search, verify results come from all file types |
| `test_chunk_text_sizes` | Chunk a long document, verify all chunks ≤ max_size, chunk count > 1 |
| `test_chunk_overlap` | Verify sentence overlap between consecutive chunks |
| `test_ast_chunk_python` | Python source file → AST chunking → chunks align with function/class boundaries |
| `test_ast_chunk_rust` | Rust source file → AST chunking → chunks align with fn/impl/struct boundaries |
| `test_ast_chunk_javascript` | JS source file → AST chunking → chunks align with function/class boundaries |
| `test_ast_fallback_to_sentence` | Non-code file with AST chunking enabled → falls back to sentence chunking |

---

### E2E-6: File Synchronization (`test_sync.rs`)
**Python source:** `test_sync.py`

| Test | What it verifies |
|------|-----------------|
| `test_merkle_tree_no_changes` | Build tree, compare to self → no changes detected |
| `test_merkle_tree_detects_added_file` | Add a file → changes detected, lists the added file |
| `test_merkle_tree_detects_removed_file` | Remove a file → changes detected, lists the removed file |
| `test_merkle_tree_detects_modified_file` | Modify file contents → changes detected |
| `test_file_synchronizer_generate_hashes` | `generate_file_hashes()` returns hashes for all files in dir |
| `test_file_synchronizer_check_for_changes` | Full lifecycle: initial scan → modify → check_for_changes → detects delta |

---

### E2E-7: Embedding Server Lifecycle (`test_embedding_manager.rs`)
**Python source:** `test_embedding_server_manager.py`

| Test | What it verifies |
|------|-----------------|
| `test_server_reuse_same_config` | Start server with config A, request again with same config → reuses (no restart) |
| `test_server_restart_on_config_change` | Start server, change metadata (passage sources change), request → restarts |
| `test_server_port_allocation` | Manager allocates a free port (no collision) |

---

### E2E-8: Index File Format Validation (`test_index_format.rs`)
**Python source:** `test_diskann_partition.py` (file format parts)

| Test | What it verifies |
|------|-----------------|
| `test_index_files_exist_after_build` | `.meta.json`, `.passages.jsonl`, `.passages.idx`, `.index` — all created |
| `test_passages_jsonl_format` | Each line is valid JSON with `text`, `metadata`, and `id` fields |
| `test_id_map_roundtrip` | Write id_map, read it back → identical |
| `test_passages_offset_random_access` | Load offset map, access passage by ID → correct text returned |
| `test_hnsw_index_binary_roundtrip` | Write compact HNSW index, read back → graph structure matches |
| `test_read_python_pickle_offset_map` | Parse a Python pickle protocol 2 offset map → correct offsets (backward compat) |
| `test_meta_json_schema` | Verify all required fields present: backend_name, embedding_model, dimensions, passage_sources |

---

### E2E-9: CLI Integration Tests (`test_cli_build_search.rs`, `test_cli_args.rs`, `test_cli_list_remove.rs`)
**Python source:** `test_ci_minimal.py`, `test_cli_ask.py`, `test_cli_verbosity.py`, `test_document_rag.py`

Tests run the `leann` binary as a subprocess via `std::process::Command`.

| Test | File | What it verifies |
|------|------|-----------------|
| `test_cli_help` | args | `leann --help` exits 0, output contains "build", "search", "ask" |
| `test_cli_build_help` | args | `leann build --help` exits 0, output contains "--docs", "--embedding-model" |
| `test_cli_search_help` | args | `leann search --help` exits 0, output contains "--complexity" |
| `test_cli_ask_positional_query` | args | `leann ask my-docs "some query"` parses without error |
| `test_cli_verbose_quiet_exclusive` | args | `leann -v -q list` fails (mutually exclusive) |
| `test_cli_list_empty` | list_remove | `leann list` in empty dir → no indexes found message |
| `test_cli_build_then_search` | build_search | Build from test corpus via CLI, then search via CLI, verify output contains results |
| `test_cli_build_then_list` | list_remove | Build an index, `leann list` → shows the index name |
| `test_cli_remove_force` | list_remove | Build index, `leann remove --force <name>` → index gone, `leann list` → empty |

---

### E2E-10: HTTP Server Tests (`test_server_endpoints.rs`)
**Python source:** None (new for Rust, but validates `leann-server`)

Uses `axum::test` helpers or spawns server on a random port.

| Test | What it verifies |
|------|-----------------|
| `test_health_endpoint` | `GET /health` → 200, body has "status": "ok" |
| `test_indexes_empty` | `GET /indexes` with no indexes → 200, empty list |
| `test_indexes_after_build` | Build an index, `GET /indexes` → lists it with correct metadata |
| `test_index_info` | `GET /indexes/{name}` → 200, correct name/model/dimensions |
| `test_index_not_found` | `GET /indexes/nonexistent` → 404 |
| `test_search_endpoint` | `POST /indexes/{name}/search` with query → 200, results array |

---

### Test Markers / Categories

```rust
// In Cargo.toml or via cfg attributes:
// - Default: all tests that use FakeEmbeddingProvider (no network, fast)
// - #[ignore] tests that require: external embedding service, live LLM, OpenAI API key
// - Feature-gated: #[cfg(feature = "pdf")] for PDF tests
```

| Category | Runs in CI | Needs network | Actual count |
|----------|-----------|---------------|--------------|
| Unit tests (leann-core src/) | Yes | No | 64 |
| Integration tests (leann-core tests/) | Yes | No | 82 |
| CLI subprocess tests (leann-cli) | Yes | No | 7 |
| HTTP server tests (leann-server) | Yes | No | 4 |
| OpenAI embedding | No (`#[ignore]`) | Yes | 0 (planned) |
| Ollama embedding | No (`#[ignore]`) | Yes | 0 (planned) |
| Format compat (Python indexes) | Yes | No | 0 (planned) |
| **Total** | | | **157** |

---

## Remaining Work

### High Priority
1. **ONNX Runtime activation** — Wire up `ort` crate for local sentence-transformer inference (currently scaffold only)
2. ~~**End-to-end tests**~~ — DONE: 82 integration tests + 11 CLI/server tests (157 total). Remaining gap: CLI build+search subprocess tests, embedding server lifecycle tests, Python format compat tests
3. **GeminiChat LLM provider** — Chat backend for Gemini (embedding provider is done)
4. ~~**PDF document loading**~~ — DONE: `pdf-extract` crate via `document_loaders` module with `pdf` feature flag
5. **Maturin build test** — Verify PyO3 bindings produce a working Python wheel

### Medium Priority
6. **Python example scripts** — Port basic_demo.py, document_rag.py, code_rag.py to use Rust bindings
7. **Python type stubs** — Generate .pyi files for IDE support
8. **CI setup** — GitHub Actions for cargo test, clippy, maturin build (Linux x86_64, macOS ARM64)
9. **Incremental rebuild in watch** — Currently watch detects changes but doesn't rebuild
10. **Tree-sitter integration** — Replace heuristic AST chunking with real tree-sitter parsing

### Low Priority / Deferred
11. **DiskANN backend** — Deferred per plan; HNSW-only for now
12. **Format compatibility tests** — Reading indexes built by the Python version
13. ~~**Recall benchmarks**~~ — DONE: Criterion benchmark suite + Rust vs Python comparison at `benchmarks/` (distance, build, search, recompute, full pipeline, index size). HNSW performance optimization complete: SIMD (NEON/AVX2), batch-4 distance, parallel build, flat heaps, visited list, early termination.
14. **MLX embedding provider** — Apple Silicon specific; defer
15. **HuggingFace chat provider** — Local model inference; defer to ONNX/Ollama

---

## Risk Areas & Mitigations

| Risk | Mitigation |
|------|-----------|
| HNSW correctness (search quality regression) | Port existing Python tests, add recall benchmarks against FAISS baseline |
| FAISS binary format compatibility (reading existing indexes) | Write a dedicated FAISS format reader; alternatively provide a migration CLI tool |
| ONNX model compatibility with sentence-transformers | Test with top 5 models (contriever, all-mpnet, nomic-embed, bge, e5). Export pipeline: `optimum-cli export onnx` |
| ZMQ protocol compatibility during migration | Keep exact same msgpack wire format; test interop (Rust client <-> Python server and vice versa) |
| DiskANN backend | Defer to Phase 2 of the project. Initially support HNSW only. DiskANN can be added later via FFI to existing C++ or a pure Rust implementation |
| PyO3 GIL management for long-running search | Release GIL during search operations with `py.allow_threads()` — already implemented |

---

## Verification

1. **Unit tests**: 64 passing across leann-core (search_result, settings, index, metadata_filter, passages, bm25, hnsw/{build, search, graph, csr, io, simd}, chunking/*, document_loaders/pdf, react_agent, sync). 0 errors, 0 warnings.
2. **CLI conformance**: Rust CLI options aligned with Python CLI (2026-02-19) — verified via `--help` output comparison
3. **Integration tests**: 82 passing across 8 test files in leann-core. Coverage: core build/search pipeline (13 tests), BM25 keyword search (8 tests), hybrid/grep search via LeannSearcher (7 tests), metadata filtering with all 13 operators (18 tests), document loading + AST chunking (17 tests), file sync/Merkle tree (7 tests), index file format validation (7 tests), chat/LLM pipeline (5 tests). All use FakeEmbeddingProvider for deterministic, network-free execution.
4. **CLI/server tests**: 11 passing — CLI subprocess/help tests (7 in leann-cli), HTTP server endpoints (4 in leann-server).
5. **Python binding tests**: Not yet written — port tests/test_basic.py, tests/test_metadata_filtering.py, tests/test_hybrid_search.py
6. **Benchmark**: Criterion benchmark suite implemented (`cargo bench --package leann-core`) with 5 groups: distance computation (SIMD at 128/384/768 dims), HNSW build (100/1K/10K/50K), HNSW search (ef 16-256), HNSW search recompute (ef 16-256), full pipeline (build+write+read+search). JSON output binary with quantile tracking and RNG seed support for scripted comparison. Python FAISS comparison suite at `benchmarks/` with orchestration script (`benchmarks/compare_rust_python.sh`). See Benchmark Results below for detailed numbers.
7. **Format compatibility**: Not yet tested — reading indexes built by Python version
8. **Cross-platform**: Not yet set up — CI for Linux (x86_64), macOS (ARM64), Windows

---

## Benchmark Results

**Geometric mean speedup: 3.41x** vs FAISS C++. Rust faster in 18/23 benchmarks, 5 ties, 0 Python wins. Distance 10-204x faster (SIMD); build 1.4-4.8x faster; search on par; index files 85% smaller.

See **[RUST_PERFORMANCE.md](RUST_PERFORMANCE.md)** for full tables, methodology, and reproduction instructions.
