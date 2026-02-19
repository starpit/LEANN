# LEANN Rust Port Plan

## Current Status (2026-02-19)

**8,415 lines of Rust across 4 crates. 47 unit tests passing. 0 errors, 0 warnings.**

All 8 phases of the initial implementation are complete. CLI has been aligned with Python CLI options and semantics. What remains is hardening: integration tests, ONNX Runtime activation, Python example porting, and CI setup.

### Crate Status

| Crate | LOC | Status |
|-------|-----|--------|
| `leann-core` | 6,561 | Complete - all modules implemented with 47 unit tests |
| `leann-cli` | 1,283 | Complete - all 8 commands wired up, aligned with Python CLI |
| `leann-server` | 222 | Complete - all endpoints functional with state management |
| `leann-python` | 349 | Complete - compiles with PyO3 0.25 (needs maturin build test) |

### File Inventory

```
leann-rs/
  Cargo.toml                         # workspace root
  crates/
    leann-core/src/
      lib.rs                    (22)  # Module declarations + re-exports
      search_result.rs          (72)  # SearchResult struct [3 tests]
      settings.rs              (153)  # Env var resolution [4 tests]
      index.rs                 (258)  # IndexMeta, IndexPaths, DistanceMetric [5 tests]
      metadata_filter.rs       (393)  # 13 operators, AND logic [7 tests]
      passages.rs              (633)  # PassageManager, JSONL I/O, pickle parser [3 tests]
      bm25.rs                  (202)  # BM25Scorer [4 tests]
      builder.rs               (332)  # LeannBuilder (build + from_embeddings)
      searcher.rs              (337)  # LeannSearcher (vector/BM25/grep/hybrid search)
      chat.rs                  (268)  # LlmProvider trait + Ollama/OpenAI/Anthropic/Simulated
      react_agent.rs           (240)  # ReAct agent [2 tests]
      sync.rs                  (283)  # MerkleTree + FileSynchronizer [2 tests]
      hnsw/
        mod.rs                   (7)
        graph.rs               (184)  # HnswGraph, GraphStorage, VectorStorage [2 tests]
        build.rs               (410)  # FAISS-style HNSW insert algorithm [1 test]
        search.rs              (490)  # Beam search + recompute callback [2 tests]
        csr.rs                 (153)  # CSR conversion + embedding pruning [1 test]
        io.rs                  (402)  # Binary format read/write (compact + standard) [1 test]
      embedding/
        mod.rs                  (54)  # EmbeddingProvider trait, EmbeddingMode enum
        client.rs              (150)  # ZMQ REQ client (text/distance/by-id)
        server.rs              (237)  # ZMQ REP server (3 request types)
        manager.rs             (215)  # Subprocess lifecycle, port allocation
        openai.rs              (107)  # OpenAI embedding API
        ollama.rs               (83)  # Ollama embedding API
        gemini.rs              (100)  # Gemini batch embedding API
        onnx.rs                (102)  # ONNX Runtime scaffold (not yet activated)
      chunking/
        mod.rs                  (73)  # chunk_text with sentence overlap [2 tests]
        sentence.rs            (120)  # Sentence splitter [4 tests]
        ast.rs                 (481)  # Python/Rust/JS/TS code chunking [4 tests]
    leann-cli/src/
      main.rs                (1,283)  # clap CLI: build/search/ask/react/list/remove/watch/serve
    leann-server/src/
      main.rs                  (222)  # Axum server: health/indexes/search/info endpoints
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

## Phase 2: HNSW Graph Engine (Pure Rust) [COMPLETE]

### 2a. HNSW Data Structures (`hnsw/graph.rs`) [COMPLETE]
- `HnswGraph` with `GraphStorage` enum: Standard (offsets/neighbors) and Compact (level_ptr/node_offsets/neighbors)
- `VectorStorage` enum: Null (pruned) and Raw (with fourcc + data bytes)
- `HnswConfig`: M, efConstruction, efSearch, distance metric, compact/recompute flags
- FourCC constants: `FOURCC_HNSW_FLAT = "IHNf"`, `FOURCC_NULL = "null"`

### 2b. HNSW Build (`hnsw/build.rs`) [COMPLETE]
- FAISS-style `IndexHNSW::add`: exponential level assignment, greedy insert, bidirectional neighbor connections
- Worst-neighbor eviction when neighbor count exceeds M
- MIPS, L2, and Cosine distance metrics
- 1 test: small graph build + verify structure

### 2c. HNSW Search with Recompute (`hnsw/search.rs`) [COMPLETE]
- Two-phase beam search: top-down greedy from max level, ef-search at level 0
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
- `ollama.rs`: Ollama embedding API via reqwest blocking
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
- 30+ supported extensions: txt, md, rst, rs, py, js, jsx, ts, tsx, java, go, c, cpp, cc, cxx, h, hpp, rb, sh, bash, toml, yaml, yml, json, xml, html, htm, css, sql, r, lua, php, swift, kt, scala, ex, exs
- Note: Binary format support (PDF, PPTX, DOCX) not yet implemented — text-based files only

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

## Remaining Work

### High Priority
1. **ONNX Runtime activation** — Wire up `ort` crate for local sentence-transformer inference (currently scaffold only)
2. **Integration tests** — End-to-end: build index from test corpus, search, verify recall
3. **GeminiChat LLM provider** — Chat backend for Gemini (embedding provider is done)
4. **PDF document loading** — Add `pdf-extract` or similar crate for binary document support
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
13. **Recall benchmarks** — Compare against FAISS baseline
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

1. **Unit tests**: 47 passing across leann-core (search_result, settings, index, metadata_filter, passages, bm25, hnsw/*, chunking/*, react_agent, sync). 0 errors, 0 warnings.
2. **CLI conformance**: Rust CLI options aligned with Python CLI (2026-02-19) — verified via `--help` output comparison
3. **Integration tests**: Not yet written — build index from test corpus, search, verify recall matches Python version
4. **Python binding tests**: Not yet written — port tests/test_basic.py, tests/test_metadata_filtering.py, tests/test_hybrid_search.py
5. **Benchmark**: Not yet run — compare search latency and index build time against current Python+FAISS
6. **Format compatibility**: Not yet tested — reading indexes built by Python version
7. **Cross-platform**: Not yet set up — CI for Linux (x86_64), macOS (ARM64), Windows
