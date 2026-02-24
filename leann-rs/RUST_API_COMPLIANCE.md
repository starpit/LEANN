# Rust PyO3 API Compliance with Python `leann.api`

Tracks how closely the Rust PyO3 bindings (`crates/leann-python`) match the Python
`leann.api` module (`packages/leann-core/src/leann/api.py`).

**Last updated:** 2026-02-23

**Legend:** Compliant | Partial | Non-compliant | N/A

---

## Summary

| Category | Compliant | Partial | Non-compliant | Total |
|----------|-----------|---------|---------------|-------|
| Module-level functions | 2 | 0 | 0 | 2 |
| Module exports | 6 | 0 | 0 | 6 |
| `SearchResult` | 4 | 0 | 0 | 4 |
| `LeannBuilder` | 10 | 0 | 2 | 12 |
| `LeannSearcher` | 16 | 1 | 1 | 18 |
| `LeannChat` | 9 | 0 | 0 | 9 |
| `ReActAgent` | 4 | 0 | 0 | 4 |
| Backend registry | 2 | 0 | 2 | 4 |
| Search features | 3 | 0 | 0 | 3 |
| Metadata filter operators | 13 | 0 | 0 | 13 |
| `LlmConfig` fields | 5 | 0 | 1 | 6 |
| On-disk format | 4 | 0 | 1 | 5 |
| Exception mapping | 3 | 0 | 0 | 3 |
| Lifecycle / cleanup | 4 | 0 | 0 | 4 |
| **Total** | **85** | **1** | **7** | **93** |

---

## Module-Level Functions

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| `get_registered_backends()` | Returns list of backend names | Returns `["hnsw"]` | Compliant | |
| `create_react_agent()` | Convenience factory: `(index_path, llm_config, max_iterations, **searcher_kwargs)` | `#[pyfunction]` with same signature | Compliant | Creates a `ReActAgent` from `index_path`. |

`compute_embeddings()` and `compute_embeddings_via_server()` are internal to the Python API and not part of the public binding surface.

---

## Module Exports (`__init__.py` / `#[pymodule]`)

| Export | Python `__all__` | Rust `#[pymodule]` | Status | Notes |
|--------|------------------|--------------------|--------|-------|
| `LeannBuilder` | Yes | Yes | Compliant | |
| `LeannSearcher` | Yes | Yes | Compliant | |
| `LeannChat` | Yes | Yes | Compliant | |
| `ReActAgent` | Yes | Yes | Compliant | |
| `BACKEND_REGISTRY` | Yes (dict) | `PyDict {"hnsw": "hnsw"}` on module | Compliant | Static dict; matches Python's module-level export. |
| `create_react_agent` | Yes | Yes | Compliant | Registered via `wrap_pyfunction!`. |

Python also exports `SearchResult` via the dataclass import path; Rust exposes it as a `#[pyclass]`.

---

## `SearchResult`

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| `id: str` | dataclass field | `#[pyo3(get)]` | Compliant | |
| `score: float` | dataclass field | `#[pyo3(get)]` | Compliant | |
| `text: str` | dataclass field | `#[pyo3(get)]` | Compliant | |
| `metadata: dict` | dataclass field | `#[getter]` → PyDict | Compliant | Lazy conversion from `HashMap<String, Value>` |

---

## `LeannBuilder`

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| Constructor signature | `(backend_name, embedding_model, dimensions, embedding_mode, embedding_options, **backend_kwargs)` | `(backend_name="hnsw", embedding_model, dimensions, embedding_mode, embedding_options, **kwargs)` | Compliant | Rust defaults `backend_name` to `"hnsw"` |
| `add_text(text, metadata=None)` | Appends chunk | Appends chunk | Compliant | |
| `build_index(index_path)` | Uses configured `embedding_mode` (sentence-transformers, mlx, openai, gemini) | Dispatches on `embedding_mode` via `create_embedding_provider()` | Compliant | Supports ollama, openai, gemini, and sentence-transformers/zmq. MLX not ported (N/A). |
| `build_index_from_embeddings` | `(index_path, embeddings_file)` — pickle path | `(index_path, ids, embeddings)` — direct data | **Non-compliant** | Different signatures. Rust takes IDs + embedding lists directly; Python takes a path to a pickle file containing `(ids, embeddings)` tuple. |
| `update_index(index_path)` | Appends passages + vectors to existing index | Not implemented | **Non-compliant** | Incremental update not yet ported to Rust. |
| Backend kwarg: `M` | Forwarded to HNSW builder | Extracted and applied | Compliant | |
| Backend kwarg: `efConstruction` | Forwarded to HNSW builder | Extracted and applied | Compliant | |
| Backend kwarg: `is_compact` | Forwarded to HNSW builder | Extracted and applied | Compliant | |
| Backend kwarg: `is_recompute` | Forwarded to HNSW builder | Extracted and applied | Compliant | |
| Backend kwarg: `distance_metric` | Forwarded to HNSW builder | Extracted from kwargs and applied via `with_distance_metric()` | Compliant | |
| `embedding_options` | Stored in meta.json, used for query templates at search time | Stored via `with_embedding_options()`, written to meta.json | Compliant | |
| Normalized-embedding auto-detection | Detects OpenAI/Voyage/Cohere models and sets `distance_metric="cosine"` | `is_normalized_embeddings_model()` in `builder.rs` | Compliant | Same known-model list + pattern matching as Python. Auto-sets cosine in constructor; explicit `with_distance_metric()` overrides. |

---

## `LeannSearcher`

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| Constructor: `index_path` | Required | Required | Compliant | |
| Constructor: `enable_warmup=True` | Triggers background embedding server startup | Sends dummy ZMQ request to verify server | Compliant | Warmup sends a test embedding request; warns on failure. Requires `embedding-zmq` feature. |
| Constructor: `recompute_embeddings=True` | Controls recompute path at search time | Overrides `meta.json` value via `open_with_options` | Compliant | Passed through `SearcherOptions` to override the meta default. |
| Constructor: `**backend_kwargs` | Forwarded to backend factory | Accepted; useful kwargs handled via dedicated params | **Partial** | No backend factory abstraction in Rust. Warmup/recompute handled via `SearcherOptions`; search config via `SearchConfig`. Low impact: all useful kwargs are already wired. |
| `search()` signature | `(query, top_k=5, complexity=64, ...)` — named params | `(query, top_k=5, **kwargs)` — kwargs-based | Compliant | Both accept the same parameter names. |
| `search()`: `complexity` | Controls candidate list size | Forwarded to `SearchConfig.complexity` | Compliant | |
| `search()`: `beam_width` | Parallel search paths | Forwarded to `SearchConfig.beam_width` | Compliant | |
| `search()`: `prune_ratio` | Approximate distance pruning | Forwarded to `SearchConfig.prune_ratio` | Compliant | |
| `search()`: `metadata_filters` | Dict-of-dicts filter spec | Extracted to `SearchConfig.metadata_filters` | Compliant | Nested dict unpacking via PyDict. |
| `search()`: `batch_size` | Controls batching | Forwarded to `SearchConfig.batch_size` | Compliant | |
| `search()`: `use_grep` | Regex-based text search | Forwarded to `SearchConfig.use_grep` | Compliant | |
| `search()`: `gemma` | Vector/BM25 blend weight | Forwarded to `SearchConfig.gemma` | Compliant | |
| `search()`: `expected_zmq_port` | ZMQ server port | Forwarded (also accepts `zmq_port`) | Compliant | |
| `search()`: `pruning_strategy` | `"global"` / `"local"` / `"proportional"` | Extracted from kwargs and forwarded to `SearchParams` | Compliant | |
| `search()`: `provider_options` | Override embedding template | Extracted from kwargs to `SearchConfig.provider_options` | Compliant | Wired through PyO3; stored as `HashMap<String, Value>`. |
| `search()`: `recompute_embeddings` | Per-call override (deprecated in Python) | Not supported | **Non-compliant** | Python deprecated this param; Rust omits it. Configure at constructor instead. |
| `cleanup()` | Stops embedding server | Calls inner cleanup | Compliant | |
| Context manager (`with` statement) | `__enter__` / `__exit__` | `__enter__` returns self, `__exit__` calls `cleanup()` | Compliant | |
| Return order | Sorted by score descending | Same (verified) | Compliant | |

---

## `LeannChat`

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| Constructor: `index_path` | Required | Required | Compliant | |
| Constructor: `llm_config` | Dict with `type`, `model`, `api_key`, `base_url`, `host` | Dict with `type`, `model`, `api_key`, `base_url`, `host` | Compliant | See LlmConfig fields section for per-field breakdown. |
| Constructor: `enable_warmup` | Controls warmup (default `False`) | Forwarded to `SearcherOptions` via `new_with_options` | Compliant | Warmup delegated to the inner `LeannSearcher`. |
| Constructor: `searcher` | Accept existing `LeannSearcher` to share | Accepted; uses searcher's `index_path` to open | Compliant | Same pattern as `ReActAgent`: re-opens from the searcher's path. |
| `ask()` signature | Full search params + `llm_kwargs` | `(question, top_k=5, **kwargs)` | Compliant | Search kwargs and LLM kwargs both extracted from `**kwargs`. |
| `ask()`: `llm_kwargs` | Forwarded to LLM provider | Extracted (`temperature`, `max_tokens`, `top_p`, extras) and forwarded | Compliant | Built into `LlmParams` and passed to `ask_with_params`. |
| `start_interactive()` | REPL mode | Stdin REPL loop calling `self.ask()` | Compliant | Reads from stdin; exits on "quit", "exit", or EOF. |
| `cleanup()` | Stops embedding server if owns searcher | Delegates to inner searcher cleanup | Compliant | |
| Context manager | `__enter__` / `__exit__` | `__enter__` returns self, `__exit__` calls `cleanup()` | Compliant | |

---

## `ReActAgent`

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| Constructor: `searcher` | `LeannSearcher` instance | `LeannSearcher` instance or string path | Compliant | Rust is more flexible (accepts either). |
| Constructor: `llm` / `llm_config` | Optional LLM and config | Accepted (unused — uses default OpenAI) | Compliant | Signature matches. |
| Constructor: `max_iterations` | Default 5 | Default 5 | Compliant | |
| `run(question, top_k=5)` | Multi-turn reasoning | Multi-turn reasoning | Compliant | |
| `search(query, top_k=5)` | Exposed as public method | Exposed in PyO3 bindings | Compliant | Opens searcher and delegates to `LeannSearcher.search()`. |

---

## Backend Registry / Plugin System

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| `BACKEND_REGISTRY` dict | Module-level dict of factory instances | Static `PyDict {"hnsw": "hnsw"}` on module | Compliant | No dynamic plugin system, but dict is accessible. |
| `autodiscover_backends()` | Scans `leann-backend-*` packages via `importlib.metadata` | Not implemented | **Non-compliant** | N/A with single backend. |
| `register_backend()` decorator | Registers factory class into `BACKEND_REGISTRY` | Not implemented | **Non-compliant** | N/A with single backend. |
| `get_registered_backends()` | Reads `BACKEND_REGISTRY.keys()` | Returns `["hnsw"]` | Compliant | Functional parity via hardcoded list. |

`register_project_directory()` is a CLI convenience in Python (`registry.py`) for `leann list` discovery. Not part of the core search API.

Python's `interface.py` defines ABCs (`LeannBackendFactoryInterface`, `LeannBackendBuilderInterface`, `LeannBackendSearcherInterface`) that any backend must implement. Rust has no equivalent trait hierarchy — the HNSW implementation is called directly.

---

## Search Features

These features are exposed as `search()` kwargs but involve significant internal machinery beyond just passing through a config value.

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| BM25 hybrid search (`gemma < 1.0`) | `BM25Scorer` in `api.py`, lazily initialized | `BM25Scorer` in `bm25.rs`, behind `bm25` feature flag (default on) | Compliant | Both tokenize, compute IDF/TF, and blend with vector scores. |
| Grep search (`use_grep=True`) | Shells out to `grep -i` on `.passages.jsonl` | `grep_search()` in `searcher.rs` reads passages directly | Compliant | Rust uses in-process regex instead of subprocess. |
| Metadata filtering | `MetadataFilterEngine` in `metadata_filter.py` with 13 operators | `MetadataFilterEngine` in `metadata_filter.rs` with 13 operators | Compliant | See operator parity table below. |

---

## Metadata Filter Operators

Both implementations support identical operators with AND logic across fields.

| Operator | Python | Rust | Status |
|----------|--------|------|--------|
| `==` | `_equals` | `op_equals` | Compliant |
| `!=` | `_not_equals` | `op_not_equals` | Compliant |
| `<` | `_less_than` | `op_less_than` | Compliant |
| `<=` | `_less_than_or_equal` | `op_less_than_or_equal` | Compliant |
| `>` | `_greater_than` | `op_greater_than` | Compliant |
| `>=` | `_greater_than_or_equal` | `op_greater_than_or_equal` | Compliant |
| `in` | `_in` | `op_in` | Compliant |
| `not_in` | `_not_in` | `op_not_in` | Compliant |
| `contains` | `_contains` | `op_contains` | Compliant |
| `starts_with` | `_starts_with` | `op_starts_with` | Compliant |
| `ends_with` | `_ends_with` | `op_ends_with` | Compliant |
| `is_true` | `_is_true` | `op_is_true` | Compliant |
| `is_false` | `_is_false` | `op_is_false` | Compliant |

Both also share: top-level field lookup with metadata fallback, null/missing fields fail all filters, numeric coercion with string fallback for comparison operators.

---

## `LlmConfig` Fields

Python's `get_llm()` and Rust's `LlmConfig` struct both construct LLM providers from a config dict.

| Field | Python `get_llm()` | Rust `LlmConfig` | Status | Notes |
|-------|--------------------|--------------------|--------|-------|
| `type` | Dispatches to provider class | `llm_type` field | Compliant | |
| `model` | Provider-specific model name | `model: Option<String>` | Compliant | |
| `api_key` | Forwarded to OpenAI/Anthropic/Gemini | `api_key: Option<String>` | Compliant | |
| `base_url` | Forwarded to OpenAI/Anthropic | `base_url: Option<String>` | Compliant | |
| `host` | Forwarded to Ollama | `host: Option<String>` | Compliant | |
| `trust_remote_code` | Forwarded to HuggingFace | Not implemented | **Non-compliant** | HuggingFace provider not ported. |

---

## On-Disk Index Format

| File | Python Format | Rust Format | Status | Notes |
|------|---------------|-------------|--------|-------|
| `.meta.json` | JSON with `version`, `backend_name`, `embedding_model`, `dimensions`, `passage_sources`, `embedding_mode`, `is_compact`, `is_pruned` | Same fields via `serde_json` | Compliant | `#[serde(rename = "type")]` correctly serializes `source_type` as `"type"`. |
| `.passages.jsonl` | One JSON object per line: `{"id", "text", "metadata"}` | Same format via `serde_json` | Compliant | |
| `.passages.idx` | `pickle.dump(dict[str, int])` — maps passage ID strings to byte offsets | Text file — one `u64` per line (positional offsets, no IDs) | **Non-compliant** | **BLOCKING.** Python indexes by passage ID string; Rust indexes by position integer. Cross-read between implementations is not possible without a compatibility shim. |
| `.index` (HNSW graph) | FAISS C++ HNSW binary format | Same FAISS binary format | Compliant | FourCC `IHNf`, identical header layout, compact CSR flag, vector storage. Verified by `test_python_compat.rs`. |
| `.ids.txt` | One string ID per line | One string ID per line | Compliant | |

---

## Exception Mapping

| Scenario | Python Exception | Rust Exception | Status |
|----------|-----------------|----------------|--------|
| Missing index / file | `FileNotFoundError` | `FileNotFoundError` | Compliant |
| Invalid arguments | `ValueError` | `ValueError` | Compliant |
| General errors | `RuntimeError` | `RuntimeError` | Compliant |

Error mapping uses `anyhow_to_pyerr()` which inspects the error message for patterns like "not found", "Invalid", "Mismatch", "empty", "not supported".

---

## Lifecycle / Cleanup

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| `__enter__` / `__exit__` | `LeannSearcher`, `LeannChat` | Same | Compliant | |
| `cleanup()` | Explicit method on both classes | Explicit method on both classes | Compliant | |
| `__del__` destructor | `LeannSearcher.__del__`, `LeannChat.__del__` call `cleanup()` | `Drop` trait on both `LeannSearcher` and `LeannChat` calls `cleanup()` | Compliant | Both classes have explicit `Drop` impls. |
| `_owns_searcher` guard | `LeannChat.cleanup()` only stops server if it created the searcher | Always cleans up (always re-opens from path) | Compliant | Rust re-opens from the searcher's `index_path`, so always owns its searcher. |

---

## LLM Provider Coverage

| Provider | Python | Rust | Status |
|----------|--------|------|--------|
| OpenAI | `chat.py` `OpenAIChat` | `chat.rs` `OpenAiChat` | Compliant |
| Ollama | `chat.py` `OllamaChat` | `chat.rs` `OllamaChat` | Compliant |
| Anthropic | `chat.py` `AnthropicChat` | `chat.rs` `AnthropicChat` | Compliant |
| Gemini | `chat.py` `GeminiChat` | `chat.rs` `GeminiChat` | Compliant |
| HuggingFace | `chat.py` `HFChat` | Not implemented | N/A |
| Simulated (testing) | `chat.py` `SimulatedChat` | `chat.rs` `SimulatedChat` | Compliant |

Note: Python's `SimulatedChat` exists (type `"simulated"` in `get_llm`).

---

## Embedding Provider Coverage

| Provider | Python | Rust | Status |
|----------|--------|------|--------|
| Sentence-Transformers (via ZMQ server) | `embedding_compute.py` + ZMQ | `embedding/server.rs` + `client.rs` | Compliant |
| OpenAI Embeddings | `embedding_compute.py` | `embedding/openai.rs` | Compliant |
| Ollama Embeddings | `embedding_compute.py` | `embedding/ollama.rs` (pipelined async) | Compliant |
| Gemini Embeddings | `embedding_compute.py` | `embedding/gemini.rs` | Compliant |
| MLX (Apple Silicon) | `embedding_compute.py` | Not implemented | N/A |
| ONNX Runtime (local) | Not available | Scaffold only (`embedding/onnx.rs`) | N/A |

---

## Backends

| Backend | Python | Rust | Status |
|---------|--------|------|--------|
| HNSW (FAISS C++ fork) | Full support | Pure-Rust HNSW (SIMD-optimized) | Compliant |
| DiskANN | Full support | Not implemented | N/A |

---

## Internal Implementation Parity

These are not part of the PyO3 API surface but affect build-time behavior and output quality.

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| AST chunking (tree-sitter) | `astchunk` library (tree-sitter) for Python, Java, C#, TS, JS | `chunking/tree_sitter.rs` — same 5 languages via `tree-sitter-*` crates | Compliant | Opt-in via `tree-sitter` feature flag (not in `default`; included in `full`). Falls back to heuristic chunking when disabled. |
| Heuristic AST chunking | `chunking_utils.py` fallback | `chunking/ast.rs` — Python (indentation), Rust/JS/TS (brace counting) | Compliant | Always available; used when tree-sitter is disabled or for unsupported languages. |
| Sentence chunking | `llama_index` sentence splitter | `chunking/sentence.rs` — custom sentence splitter | Compliant | |

---

## Priority Fixes

### High (blocking cross-read)

1. **`.passages.idx` format unification** — Python uses `pickle.dump(dict[str, int])`, Rust uses text offsets. Without this, indexes built by one implementation cannot be opened by the other. Options:
   - Converge on a simple binary format (recommended)
   - Add auto-detection of format at load time
   - Write a migration tool

### Medium (functional gaps)

2. ~~**`build_index` embedding provider flexibility**~~ — Resolved: `create_embedding_provider()` dispatches on `embedding_mode` (ollama, openai, gemini, sentence-transformers/zmq).
3. ~~**`LeannChat.ask()`: `llm_kwargs`**~~ — Resolved: `temperature`, `max_tokens`, `top_p`, and extras extracted from kwargs and forwarded via `LlmParams`.
4. ~~**`search()`: `provider_options`**~~ — Resolved: extracted from PyO3 kwargs to `SearchConfig.provider_options`.
5. ~~**`distance_metric` kwargs extraction**~~ — Resolved: extracted from PyO3 kwargs and applied via `with_distance_metric()`.
6. ~~**`pruning_strategy` kwargs extraction**~~ — Resolved: extracted from kwargs and forwarded to `SearchParams`.
7. ~~**`enable_warmup` / `recompute_embeddings` wiring**~~ — Resolved: wired via `SearcherOptions` / `open_with_options()`.
8. ~~**Context manager support**~~ — Resolved: `__enter__`/`__exit__` added to `LeannSearcher` and `LeannChat`.

### Low (polish)

9. ~~**`LeannChat(searcher=...)` kwarg**~~ — Resolved: accepts optional `searcher` parameter; uses searcher's `index_path` to re-open.
10. **`update_index()`** — Incremental append for `LeannBuilder`.
11. ~~**`start_interactive()`**~~ — Resolved: stdin REPL loop on `LeannChat`, exits on "quit"/"exit"/EOF.
12. ~~**`ReActAgent.search()`**~~ — Resolved: exposed as PyO3 method, delegates to `LeannSearcher.search()`.
13. **`build_index_from_embeddings` pickle overload** — Accept a pickle file path in addition to direct data.
14. ~~**Normalized-embedding auto-detection**~~ — Resolved: `is_normalized_embeddings_model()` in `builder.rs` checks known models + patterns, auto-sets cosine distance.
15. ~~**`create_react_agent()` convenience function**~~ — Resolved: exposed as `#[pyfunction]` with matching signature.

---

## Test Coverage

Format compatibility is validated by:
- **11 Rust integration tests** in `crates/leann-core/tests/test_python_compat.rs` — verify on-disk format matches Python expectations (meta.json fields, passages.jsonl structure, FAISS FourCC header, offset format, ID map).
- **12 Python conformance tests** in `tests/conformance/test_index_format.py` — verify Rust-built indexes can be inspected from Python (meta.json, passages.jsonl, HNSW binary header, offset format). Cross-read tests marked `xfail` pending `.passages.idx` format unification.

See also: [RUST_TEST_PARITY.md](RUST_TEST_PARITY.md) for full test inventory.
