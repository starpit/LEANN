# Rust Test Parity with Python CI

Comparison of Python test coverage (`tests/`) with Rust test coverage (`crates/leann-core/`, `crates/leann-cli/`, `crates/leann-server/`).

**Rust totals: 231 tests (110 unit + 20 tree-sitter + 85 leann-core integration + 12 leann-cli + 4 leann-server), 0 failures.**

## Core Feature Tests

These Python test files exercise core LEANN functionality that the Rust crate also implements.

### test_basic.py — Build & Search

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_imports` | `test_simulated_chat_response` | [test_chat_pipeline.rs](crates/leann-core/tests/test_chat_pipeline.rs) |
| `test_backend_basic` (build + search) | `test_build_and_search_100_docs` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| `test_backend_basic` (build + search) | `test_build_and_search_1000_docs` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| `test_large_index` | `test_build_and_search_1000_docs` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_build_creates_expected_files` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_meta_json_content` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_build_with_compact_csr` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_build_standard_with_vectors` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_build_with_distance_metrics` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_search_top_k_bounds` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_build_from_precomputed_embeddings` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_hnsw_index_compact_roundtrip` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |
| — | `test_hnsw_index_standard_roundtrip` | [test_build_search.rs](crates/leann-core/tests/test_build_search.rs) |

**Status: Fully covered.** 11 integration tests vs Python's 3 — deeper coverage of build/search paths including compact CSR, distance metrics, and index serialization.

---

### test_metadata_filtering.py — Metadata Filters

| Python Test | Rust Equivalent | Location |
|---|---|---|
| `test_no_filters_returns_all_results` | `test_filter_none_passthrough` | integration + unit |
| `test_equals_filter` | `test_filter_equals` | integration + unit |
| `test_not_equals_filter` | `test_filter_not_equals` | integration + unit |
| `test_less_than_filter` | `test_filter_less_than` | integration + unit |
| `test_less_than_or_equal_filter` | `test_filter_less_than_or_equal` | integration + unit |
| `test_greater_than_filter` | `test_filter_greater_than` | integration + unit |
| `test_greater_than_or_equal_filter` | `test_filter_greater_equal` | integration + unit |
| `test_in_filter` | `test_filter_in` | integration + unit |
| `test_not_in_filter` | `test_filter_not_in` | integration + unit |
| `test_contains_filter` | `test_filter_contains` | integration + unit |
| `test_starts_with_filter` | `test_filter_starts_with` | integration + unit |
| `test_ends_with_filter` | `test_filter_ends_with` | integration + unit |
| `test_is_true_filter` | `test_filter_is_true` | unit |
| `test_is_false_filter` | `test_filter_is_false` | unit |
| `test_compound_filters` | `test_filter_compound_and` | integration + unit |
| `test_multiple_operators_same_field` | `test_filter_range` | integration + unit |
| `test_missing_field_fails_filter` | `test_filter_missing_field` | unit |
| `test_invalid_operator` | `test_filter_invalid_operator` | unit |
| `test_type_coercion_numeric` | `test_filter_type_coercion` | unit |
| `test_empty_results_list` | `test_filter_empty_results_list` | unit |
| `test_search_result_filtering` | `test_bm25_search_with_metadata_filters` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |
| `test_filter_maintains_search_result_type` | (Rust types enforce this at compile time) | — |

**Status: Fully covered.** 16 integration tests (test_metadata_filtering.rs) + 22 unit tests (metadata_filter.rs). All 13 operators tested end-to-end via BM25 search + filter pipeline.

---

### test_hybrid_search.py — Hybrid Search (Vector + BM25)

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_pure_vector_search` | (tested at HNSW level via test_build_search.rs) | — |
| `test_pure_keyword_search` | `test_pure_bm25_search` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |
| `test_hybrid_search_balanced` | (requires ZMQ server) | — |
| `test_hybrid_search_vector_heavy` | (requires ZMQ server) | — |
| `test_hybrid_search_keyword_heavy` | (requires ZMQ server) | — |
| `test_hybrid_search_score_combination` | (requires ZMQ server) | — |
| `test_hybrid_search_with_metadata_filters` | `test_bm25_search_with_metadata_filters` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |
| — | `test_bm25_search_cooking` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |
| — | `test_bm25_scores_descending` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |
| — | `test_grep_search` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |
| — | `test_grep_search_no_match` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |
| — | `test_bm25_nonexistent_terms` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |
| — | `test_bm25_with_sample_documents` | [test_hybrid_search.rs](crates/leann-core/tests/test_hybrid_search.rs) |

**Status: Partially covered.** BM25 and grep paths fully tested (8 tests). Hybrid blend tests (`gemma` between 0 and 1) require a ZMQ embedding server and are not portable without mocking.

---

### test_sync.py — File Synchronization

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_no_changes_if_root_hash_same` | `test_merkle_tree_no_changes` | [test_sync.rs](crates/leann-core/tests/test_sync.rs) |
| `test_added_removed_modified` | 3 separate tests | [test_sync.rs](crates/leann-core/tests/test_sync.rs) |
| `test_generate_file_hashes` | `test_hash_data_consistent` | [test_sync.rs](crates/leann-core/tests/test_sync.rs) |
| `test_check_for_changes_detected` | 2 FileSynchronizer tests | [test_sync.rs](crates/leann-core/tests/test_sync.rs) |

**Status: Fully covered.** 9 Rust tests cover all Python scenarios plus additional edge cases.

---

### test_astchunk_integration.py — AST Chunking & Document Loading

**Status: Fully covered.** 17 integration tests in test_document_loading.rs cover language detection, traditional chunking, AST chunking (Python/Rust/JS), fallback, metadata preservation, and file loading. Additionally, 20 unit tests in `chunking/tree_sitter.rs` (feature-gated behind `tree-sitter-*`) validate grammar-based AST chunking for Python, Java, C#, TypeScript/TSX, and JavaScript — bringing the Rust chunker to parity with Python's `astchunk` (tree-sitter) integration.

---

### test_ci_minimal.py / test_cli_ask.py / test_cli_verbosity.py — CLI

**Status: Covered where applicable.** 7 CLI args tests + 5 CLI list/remove lifecycle tests = 12 total in leann-cli. C++ output suppression tests are Python/FAISS-specific (not applicable).

---

### test_embedding_server_manager.py — Embedding Server Lifecycle

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_server_start` | (requires leann binary; tested at unit level) | — |
| `test_server_reuse_same_config` | (requires leann binary) | — |
| `test_server_port_allocation` | `test_manager_new_state` + Default test | [test_embedding_manager.rs](crates/leann-core/tests/test_embedding_manager.rs) |

**Status: Partially covered.** 5 tests in test_embedding_manager.rs verify manager construction, state, and graceful lifecycle (new, is_alive, stop, Default). Full server start/reuse tests require a running binary.

---

## Test File Inventory

### Integration Tests (leann-core)

| Test File | Tests | Description |
|---|---|---|
| `test_build_search.rs` | 11 | Core pipeline: build → search → verify |
| `test_metadata_filtering.rs` | 16 | All 13 filter operators via BM25 search + filter e2e |
| `test_hybrid_search.rs` | 8 | BM25 + grep search, metadata filters |
| `test_document_loading.rs` | 17 | File loading, chunking, AST chunking |
| `test_python_compat.rs` | 11 | Cross-implementation format compatibility (meta.json, passages, HNSW binary, offsets) |
| `test_sync.rs` | 9 | Merkle tree + FileSynchronizer |
| `test_chat_pipeline.rs` | 5 | SimulatedChat LLM, LlmConfig |
| `test_embedding_manager.rs` | 5 | EmbeddingServerManager lifecycle |
| `test_index_format.rs` | 3 | Index meta schema validation |
| **Subtotal** | **85** | |

### CLI Tests (leann-cli)

| Test File | Tests | Description |
|---|---|---|
| `test_cli_args.rs` | 7 | Subcommand help, version, argument parsing |
| `test_cli_list_remove.rs` | 5 | List + remove lifecycle via subprocess |
| **Subtotal** | **12** | |

### Server Tests (leann-server)

| Test File | Tests | Description |
|---|---|---|
| `test_server_endpoints.rs` | 4 | Health, indexes, not_found |
| **Subtotal** | **4** | |

### Unit Tests (leann-core src/)

| Module | Tests | Description |
|---|---|---|
| `bm25.rs` | 15 | BM25 scorer: fit, search, tokenize, edge cases, large corpus |
| `metadata_filter.rs` | 22 | All operators, compound logic, edge cases, sample results |
| `hnsw/simd.rs` | 12 | L2/IP distance, batch_4, VisitedList, normalize |
| `react_agent.rs` | 11 | LLM response parsing, format_search_results |
| `hnsw/build.rs` | 3 | Graph construction (serial + parallel) |
| `hnsw/search.rs` | 2 | Search with stored vectors and recompute |
| `hnsw/io.rs` | 1 | Compact index roundtrip |
| `hnsw/graph.rs` | 2 | Config defaults, FourCC constants |
| `hnsw/csr.rs` | 1 | CSR conversion |
| `chunking/mod.rs` | 2 | Basic chunking |
| `chunking/ast.rs` | 4 | Heuristic AST chunking (Python, Rust, generic, detection) |
| `chunking/tree_sitter.rs` | 20 | Tree-sitter grammar AST chunking: per-language smoke (Python/Java/C#/TS/TSX/JS), decorators, large function splits, nested classes, line number accuracy, recursive descent, dispatch integration, fallback paths, empty/unsupported input |
| `chunking/sentence.rs` | 4 | Sentence splitting |
| `index.rs` | 4 | IndexMeta roundtrip, distance metric, paths |
| `passages.rs` | 3 | Passage I/O, ID map, not-found error |
| `search_result.rs` | 3 | SearchResult creation, metadata, serialization |
| `settings.rs` | 4 | Ollama host, OpenAI key, URL cleaning |
| `sync.rs` | 2 | Hash data, Merkle tree |
| `document_loaders/pdf.rs` | 3 | PDF extraction |
| **Subtotal** | **130** (110 default + 20 tree-sitter) | |

---

## Python-Only Tests (Not Applicable to Rust)

| Python Test File | Reason |
|---|---|
| `test_token_truncation.py` | Python-specific tiktoken/tokenizer integration |
| `test_cli_prompt_template.py` | Python CLI prompt templates |
| `test_cpu_only_install.py` | Python packaging (pyproject.toml) |
| `test_diskann_partition.py` | DiskANN backend (not implemented in Rust) |
| `test_document_rag.py` | End-to-end RAG app (Python app layer) |
| `test_embedding_prompt_template.py` | OpenAI embedding API (Python SDK) |
| `test_lmstudio_bridge.py` | LM Studio integration (Python-specific) |
| `test_mcp_integration.py` | MCP server (separate Python package) |
| `test_mcp_standalone.py` | MCP server (separate Python package) |
| `test_prompt_template_e2e.py` | Python prompt template system |
| `test_prompt_template_persistence.py` | Python prompt template persistence |
| `test_readme_examples.py` | Python API examples |

---

## Summary

| Category | Python Tests | Rust Tests | Parity |
|---|---|---|---|
| Build & Search | 3 | 11 | Rust exceeds |
| Metadata Filtering | 26 | 16 integration + 22 unit | Fully covered |
| Hybrid Search (BM25) | 7 | 8 | BM25/grep covered; blend requires ZMQ |
| Sync | 5 | 9 | Rust exceeds |
| AST Chunking / Doc Loading | 20+ | 17 integration + 20 tree-sitter unit | Fully covered (tree-sitter parity) |
| CI Smoke Tests | 4 | 5 | Fully covered |
| CLI Args & Lifecycle | 12 | 12 | Fully covered |
| Embedding Manager | 5 | 5 | Partially covered (no server start) |
| BM25 Scorer | — | 15 unit | Rust-only |
| Index Format | — | 3 | Rust-only |
| Python Format Compat | — | 11 | Rust-only (validates cross-impl format) |
| Chat Pipeline | — | 5 | Rust-only |
| HNSW Internals | — | 21 unit | Rust-only |
| ReAct Agent | — | 11 unit | Rust-only |
| HTTP Server | — | 4 | Rust-only |
| Python-specific | 55+ | — | Not applicable |
| **Total** | | **231** | |
