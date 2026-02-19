# LEANN-rs

Rust implementation of [LEANN](https://github.com/your-org/leann) -- a lightweight vector database and RAG system that achieves 97% storage reduction through graph-based selective embedding recomputation.

## Overview

LEANN-rs is a full rewrite of the Python LEANN system in Rust, providing:

- **Pure Rust HNSW engine** -- build and search without FAISS or any C++ dependencies
- **Embedding recomputation** -- prune stored embeddings and recompute on-the-fly via ZMQ, reducing index size by ~97%
- **Multiple embedding backends** -- OpenAI, Ollama, Gemini APIs (ONNX local inference planned)
- **RAG pipeline** -- search + LLM chat with Ollama, OpenAI, and Anthropic providers
- **Python bindings** -- PyO3-based native module, drop-in replacement for the Python version
- **HTTP server** -- Axum-based REST API for search

## Crates

| Crate | Description |
|-------|-------------|
| `leann-core` | Core library: HNSW graph, embeddings, search, builder, passages, BM25, metadata filtering |
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

The `leann-python` crate provides native Python bindings via PyO3. Build with [maturin](https://www.maturin.rs/):

```bash
cd crates/leann-python
maturin develop --release
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

## Architecture

### HNSW Engine

The core HNSW implementation in `leann-core/src/hnsw/` includes:

- **build.rs** -- FAISS-style insert algorithm with random level assignment, greedy neighbor selection, and bidirectional connections
- **search.rs** -- Two-phase beam search with support for both stored-vector and recompute modes
- **csr.rs** -- Compact CSR format conversion for pruned indexes
- **io.rs** -- Binary serialization for both standard and compact graph formats

### Embedding Recomputation

Instead of storing all embedding vectors (which dominate index size), LEANN prunes them and recomputes distances on-the-fly during search via a ZMQ REQ/REP protocol:

1. The search algorithm encounters a pruned node
2. It sends node IDs + query vector to the embedding server via ZMQ
3. The server recomputes embeddings and returns distances
4. Search continues with fresh distances

This achieves ~97% storage reduction with minimal latency impact.

### Index File Format

A LEANN index consists of:

| File | Contents |
|------|----------|
| `<name>.meta.json` | Index metadata (model, dimensions, backend config) |
| `<name>.passages.jsonl` | Raw text chunks with metadata |
| `<name>.passages.idx` | Byte-offset map for random passage access |
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
```

## License

MIT
