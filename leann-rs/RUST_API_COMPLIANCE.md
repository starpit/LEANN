# Rust PyO3 API Compliance with Python `leann.api`

Tracks how closely the Rust PyO3 bindings (`crates/leann-python`) match the Python
`leann.api` module (`packages/leann-core/src/leann/api.py`).

**Last updated:** 2026-02-22

**Legend:** Compliant | Partial | Non-compliant | N/A

---

## Summary

| Category | Compliant | Partial | Non-compliant | Total |
|----------|-----------|---------|---------------|-------|
| Module-level functions | 1 | 0 | 0 | 1 |
| `SearchResult` | 4 | 0 | 0 | 4 |
| `LeannBuilder` | 9 | 0 | 2 | 11 |
| `LeannSearcher` | 15 | 1 | 2 | 18 |
| `LeannChat` | 7 | 0 | 2 | 9 |
| `ReActAgent` | 4 | 0 | 0 | 4 |
| On-disk format | 4 | 0 | 1 | 5 |
| Exception mapping | 3 | 0 | 0 | 3 |
| **Total** | **47** | **1** | **7** | **55** |

---

## Module-Level Functions

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| `get_registered_backends()` | Returns list of backend names | Returns `["hnsw"]` | Compliant | |

`compute_embeddings()` and `compute_embeddings_via_server()` are internal to the Python API and not part of the public binding surface.

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
| `build_index(index_path)` | Compute embeddings + build | Uses Ollama provider | Compliant | |
| `build_index_from_embeddings` | `(index_path, embeddings_file)` — pickle path | `(index_path, ids, embeddings)` — direct data | **Non-compliant** | Different signatures. Rust takes IDs + embedding lists directly; Python takes a path to a pickle file containing `(ids, embeddings)` tuple. |
| `update_index(index_path)` | Appends passages + vectors to existing index | Not implemented | **Non-compliant** | Incremental update not yet ported to Rust. |
| Backend kwarg: `M` | Forwarded to HNSW builder | Extracted and applied | Compliant | |
| Backend kwarg: `efConstruction` | Forwarded to HNSW builder | Extracted and applied | Compliant | |
| Backend kwarg: `is_compact` | Forwarded to HNSW builder | Extracted and applied | Compliant | |
| Backend kwarg: `is_recompute` | Forwarded to HNSW builder | Extracted and applied | Compliant | |
| Backend kwarg: `distance_metric` | Forwarded to HNSW builder | Extracted from kwargs and applied via `with_distance_metric()` | Compliant | |
| Normalized-embedding auto-detection | Detects OpenAI/Voyage/Cohere models and sets `distance_metric="cosine"` | `is_normalized_embeddings_model()` in `builder.rs` | Compliant | Same known-model list + pattern matching as Python. Auto-sets cosine in constructor; explicit `with_distance_metric()` overrides. |

---

## `LeannSearcher`

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| Constructor: `index_path` | Required | Required | Compliant | |
| Constructor: `enable_warmup=True` | Triggers background embedding server startup | Sends dummy ZMQ request to verify server | Compliant | Warmup sends a test embedding request; warns on failure. Requires `embedding-zmq` feature. |
| Constructor: `recompute_embeddings=True` | Controls recompute path at search time | Overrides `meta.json` value via `open_with_options` | Compliant | Passed through `SearcherOptions` to override the meta default. |
| Constructor: `**backend_kwargs` | Forwarded to backend factory | Accepted; useful kwargs handled via dedicated params | **Partial** | No backend factory abstraction in Rust. Warmup/recompute handled via `SearcherOptions`; search config via `SearchConfig`. |
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
| `search()`: `provider_options` | Override embedding template | Not wired through PyO3 | **Non-compliant** | |
| `search()`: `recompute_embeddings` | Per-call override (deprecated in Python) | Not supported | **Non-compliant** | Python deprecated this; reasonable to omit. |
| `cleanup()` | Stops embedding server | Calls inner cleanup | Compliant | |
| Context manager (`with` statement) | `__enter__` / `__exit__` | `__enter__` returns self, `__exit__` calls `cleanup()` | Compliant | |
| Return order | Sorted by score descending | Same (verified) | Compliant | |

---

## `LeannChat`

| Feature | Python | Rust | Status | Notes |
|---------|--------|------|--------|-------|
| Constructor: `index_path` | Required | Required | Compliant | |
| Constructor: `llm_config` | Dict with `type`, `model`, `api_key` | Dict with `type`, `model`, `api_key` | Compliant | |
| Constructor: `enable_warmup` | Controls warmup (default `False`) | Forwarded to `SearcherOptions` via `new_with_options` | Compliant | Warmup delegated to the inner `LeannSearcher`. |
| Constructor: `searcher` | Accept existing `LeannSearcher` to share | Not supported | **Non-compliant** | Rust always creates its own searcher from `index_path`. |
| `ask()` signature | Full search params + `llm_kwargs` | `(question, top_k=5, **kwargs)` | Compliant | Search kwargs forwarded to `ask_with_params`. |
| `ask()`: `llm_kwargs` | Forwarded to LLM provider | Not wired | **Non-compliant** | LLM always uses default params. |
| `start_interactive()` | REPL mode | Not implemented | **Non-compliant** | CLI handles interactive mode separately. |
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

## LLM Provider Coverage

| Provider | Python | Rust | Status |
|----------|--------|------|--------|
| OpenAI | `chat.py` `OpenAIInterface` | `chat.rs` `OpenAiChat` | Compliant |
| Ollama | `chat.py` `OllamaLLM` | `chat.rs` `OllamaChat` | Compliant |
| Anthropic | `chat.py` `AnthropicLLM` | `chat.rs` `AnthropicChat` | Compliant |
| Gemini | `chat.py` `GeminiLLM` | `chat.rs` `GeminiChat` | Compliant |
| HuggingFace | `chat.py` `HuggingFaceLLM` | Not implemented | N/A |
| Simulated (testing) | Not available | `chat.rs` `SimulatedChat` | N/A |

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

## Priority Fixes

### High (blocking cross-read)

1. **`.passages.idx` format unification** — Python uses `pickle.dump(dict[str, int])`, Rust uses text offsets. Without this, indexes built by one implementation cannot be opened by the other. Options:
   - Converge on a simple binary format (recommended)
   - Add auto-detection of format at load time
   - Write a migration tool

### Medium (functional gaps)

2. ~~**`distance_metric` kwargs extraction**~~ — Resolved: extracted from PyO3 kwargs and applied via `with_distance_metric()`.
3. ~~**`pruning_strategy` kwargs extraction**~~ — Resolved: extracted from kwargs and forwarded to `SearchParams`.
4. ~~**`enable_warmup` / `recompute_embeddings` wiring**~~ — Resolved: wired via `SearcherOptions` / `open_with_options()`.
5. ~~**Context manager support**~~ — Resolved: `__enter__`/`__exit__` added to `LeannSearcher` and `LeannChat`.

### Low (polish)

6. **`LeannChat(searcher=...)` kwarg** — Accept an existing searcher to avoid re-opening.
7. **`update_index()`** — Incremental append for `LeannBuilder`.
8. **`start_interactive()`** — REPL mode for `LeannChat` (CLI handles this separately).
9. ~~**`ReActAgent.search()`**~~ — Resolved: exposed as PyO3 method, delegates to `LeannSearcher.search()`.
10. **`build_index_from_embeddings` pickle overload** — Accept a pickle file path in addition to direct data.
11. ~~**Normalized-embedding auto-detection**~~ — Resolved: `is_normalized_embeddings_model()` in `builder.rs` checks known models + patterns, auto-sets cosine distance.

---

## Test Coverage

Format compatibility is validated by:
- **11 Rust integration tests** in `crates/leann-core/tests/test_python_compat.rs` — verify on-disk format matches Python expectations (meta.json fields, passages.jsonl structure, FAISS FourCC header, offset format, ID map).
- **12 Python conformance tests** in `tests/conformance/test_index_format.py` — verify Rust-built indexes can be inspected from Python (meta.json, passages.jsonl, HNSW binary header, offset format). Cross-read tests marked `xfail` pending `.passages.idx` format unification.

See also: [RUST_TEST_PARITY.md](RUST_TEST_PARITY.md) for full test inventory.
