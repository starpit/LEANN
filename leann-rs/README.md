# LEANN-rs

Rust implementation of [LEANN](https://github.com/your-org/leann) -- a lightweight vector database and RAG system that achieves 97% storage reduction through graph-based selective embedding recomputation.

## Overview

LEANN-rs is a full rewrite of the Python LEANN system in Rust, providing:

- **Pure Rust HNSW engine** -- build and search without FAISS or any C++ dependencies
- **Embedding recomputation** -- prune stored embeddings and recompute on-the-fly via in-process providers, reducing index size by ~97%
- **Multiple embedding backends** -- OpenAI, Ollama (pipelined async), Gemini APIs wired directly into the searcher (ONNX local inference planned)
- **RAG pipeline** -- search + LLM chat with Ollama, OpenAI, and Anthropic providers
- **Python bindings** -- PyO3-based native module, drop-in replacement for the Python version
- **HTTP server** -- Axum-based REST API for search

## Crates

| Crate | Description |
|-------|-------------|
| `leann-core` | Core library: HNSW graph (SIMD-optimized), embeddings, search, builder, passages, BM25, metadata filtering (~9K LOC) |
| `leann-cli` | CLI binary (`leann build`, `search`, `ask`, `react`, `list`, `remove`, `watch`, `serve`) |
| `leann-server` | Standalone HTTP server with index management and search endpoints |
| `leann-python` | PyO3 bindings exposing `LeannBuilder`, `LeannSearcher`, `LeannChat`, `ReActAgent` |

## Quick Start

### Build

```bash
cargo build --release
```

### Build an index

```bash
# From a directory of text/code files
leann build my-index --docs ./documents/ --embedding-model text-embedding-3-small --embedding-mode openai

# With Ollama (local, no API key needed)
leann build my-index --docs ./documents/ --embedding-model nomic-embed-text --embedding-mode ollama

# From multiple directories and individual files
leann build my-index --docs ./src ./tests ./config.json

# Build with AST-aware code chunking
leann build my-code --docs ./src --use-ast-chunking

# Force rebuild with custom file types
leann build my-docs --docs ./ --file-types .txt,.pdf,.pptx --force

# Disable compact storage / recomputation
leann build my-index --docs ./data --no-compact --no-recompute
```

### Search

```bash
leann search my-index "how does HNSW search work"
leann search my-index "query" --top-k 10 --show-metadata
```

### RAG Q&A

```bash
# Single question
leann ask my-index "What is embedding recomputation?"

# Interactive mode
leann ask my-index --interactive

# With a specific LLM provider
leann ask my-index "question" --llm openai --model gpt-4o --api-key $OPENAI_API_KEY

# With thinking budget for reasoning models
leann ask my-index "complex question" --thinking-budget high
```

### ReAct Agent

```bash
# Multi-turn retrieval and reasoning
leann react my-index "complex question requiring multiple searches"
```

### Other commands

```bash
leann list                         # List all indexes in current directory
leann remove my-index              # Delete an index and all its files
leann remove my-index --force      # Delete without confirmation
leann warmup my-index              # Verify embedding provider connectivity
leann watch my-index               # Check for file changes since last build
leann serve --port 8080            # Start HTTP server
```

### Global options

```bash
leann -v build my-index --docs ./src  # Verbose output (including backend logs)
leann -q search my-index "query"      # Quiet mode (suppress non-essential output)
```

## HTTP Server

Start the standalone server:

```bash
# Via the server binary
LEANN_INDEX_DIR=./indexes leann-server

# Or via the CLI
leann serve --port 8080
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/indexes` | List all indexes |
| `GET` | `/indexes/{name}` | Get index info |
| `POST` | `/indexes/{name}/search` | Search an index |

Search request body:

```json
{
  "query": "your search query",
  "top_k": 5,
  "complexity": 64,
  "use_grep": false
}
```

## Python Bindings

The `leann-python` crate provides native Python bindings via PyO3, with `.pyi` type stubs for IDE autocomplete/typecheck. Build with [maturin](https://www.maturin.rs/):

```bash
cd crates/leann-python
maturin develop --release
```

Run binding tests:

```bash
cd crates/leann-python && cargo test --test test_maturin
```

Usage:

```python
from leann import LeannBuilder, LeannSearcher, LeannChat

# Build an index
builder = LeannBuilder("nomic-embed-text", embedding_mode="ollama")
builder.add_text("HNSW is a graph-based ANN algorithm.", {"source": "docs"})
builder.add_text("Embedding recomputation saves storage.", {"source": "docs"})
builder.build_index("./my-index")

# Search
searcher = LeannSearcher("./my-index")
results = searcher.search("graph algorithms", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.text[:80]}")

# RAG Q&A
chat = LeannChat("./my-index", llm_config={"type": "ollama", "model": "llama3:8b"})
answer = chat.ask("What is HNSW?")
print(answer)
```

## Feature Flags

`leann-core` uses Cargo feature flags to control compilation scope. All features are enabled by default.

| Feature | Dependencies | What it enables |
|---------|-------------|-----------------|
| `chat` | `reqwest` | LLM chat backends (OpenAI, Anthropic, Gemini, Ollama) + ReAct agent |
| `embedding-remote` | `reqwest`, `tokio` | Remote embedding providers (OpenAI, Ollama, Gemini) wired into `LeannSearcher` |
| `parallel` | `rayon` | Parallel HNSW build via rayon thread pool |
| `bm25` | -- | BM25 keyword search + hybrid search |
| `watch` | `sha2` | Merkle-tree file change detection |
| `pdf` | `pdf-extract` | PDF document loading |
| `full` | *all of the above* | Everything |

### Minimal builds

```toml
# HNSW-only (build, search, I/O, SIMD) — no network, no async, no rayon
leann-core = { version = "0.1", default-features = false }

# Add parallel build
leann-core = { version = "0.1", default-features = false, features = ["parallel"] }

# Search with BM25 but no LLM/embedding network calls
leann-core = { version = "0.1", default-features = false, features = ["parallel", "bm25"] }

# Full RAG application
leann-core = { version = "0.1" }
```

With `--no-default-features`, the only required dependencies are `serde`, `ndarray`, `rand`, `regex`, and `tracing`.

## Architecture

### HNSW Engine

The core HNSW implementation in `leann-core/src/hnsw/` includes:

- **simd.rs** -- NEON (aarch64) and AVX2 (x86_64) SIMD-optimized L2 and inner product distance with batch-4 processing, plus `FlatMinHeap`/`FlatMaxHeap` and `VisitedList` data structures
- **build.rs** -- FAISS-style insert algorithm with parallel construction (rayon), monomorphized distance functions, early termination, and deterministic RNG seed support
- **search.rs** -- Two-phase beam search with SIMD batch-4 distance, flat heaps, generation-counter visited tracking, and support for both stored-vector and recompute modes
- **csr.rs** -- Compact CSR format conversion for pruned indexes
- **io.rs** -- Binary serialization for both standard and compact graph formats

### Embedding Recomputation

Instead of storing all embedding vectors (which dominate index size), LEANN prunes them and recomputes distances on-the-fly during search using in-process embedding providers:

1. The search algorithm encounters a pruned node
2. The recompute callback looks up passage texts from the `PassageManager`
3. The embedding provider (Ollama, OpenAI, Gemini) computes fresh embeddings in-process
4. Distances are computed locally via SIMD (L2/inner product)
5. Search continues with fresh distances

This achieves ~97% storage reduction with minimal latency impact. Unlike the Python version (which uses a ZMQ subprocess to bridge C++/FAISS), the Rust HNSW engine is pure Rust, so providers are called directly -- no IPC overhead.

### Index File Format

A LEANN index consists of:

| File | Contents |
|------|----------|
| `<name>.meta.json` | Index metadata (model, dimensions, backend config) |
| `<name>.passages.jsonl` | Raw text chunks with metadata |
| `<name>.passages.idx` | Byte offsets for random passage access (one offset per line) |
| `<name>.index` | HNSW graph (standard or compact CSR) |
| `<name>.ids.txt` | Node ID to passage ID mapping |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | -- | OpenAI API key for embeddings and chat |
| `ANTHROPIC_API_KEY` | -- | Anthropic API key for chat |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | -- | Gemini API key for embeddings |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `PORT` | `8080` | HTTP server port |
| `LEANN_INDEX_DIR` | `.` | Directory for index storage (server) |

## Benchmarks

The pure-Rust HNSW engine matches or exceeds FAISS C++ performance: **3.3x geometric mean speedup** across 25 benchmarks (distance, build, search, recompute, full pipeline, passage lookup, index size). Distance computations are 9-111x faster via SIMD; search is on par; passage lookups are 2.5-2.8x faster; index files are 85% smaller.

See **[RUST_PERFORMANCE.md](RUST_PERFORMANCE.md)** for full results, methodology, and reproduction instructions.

Quick start:

```bash
# All-in-one comparison (Rust vs Python/FAISS)
bash benchmarks/compare_rust_python.sh

# Criterion benchmarks only (HTML reports in target/criterion/)
cargo bench --package leann-core

# Benchmarks only need the `parallel` feature (for build_hnsw_with_pool);
# no-default-features + parallel is sufficient:
cargo bench --package leann-core --no-default-features --features parallel
```

## Development

```bash
# Run tests
cargo test

# Check all crates
cargo check

# Build release
cargo build --release

# Build Python bindings
cd crates/leann-python && maturin develop

# Test Python bindings (maturin develop + pytest)
cd crates/leann-python && cargo test --test test_maturin
```

## License

MIT
