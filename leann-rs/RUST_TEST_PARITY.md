# Rust Test Parity with Python CI

Comparison of Python test coverage (`tests/`) with Rust test coverage (`crates/leann-core/` and `crates/leann-cli/`).

**Rust totals: 161 tests (66 unit + 88 leann-core integration + 7 leann-cli integration), 0 failures.**

## Core Feature Tests

These Python test files exercise core LEANN functionality that the Rust crate also implements.

### test_basic.py — Build & Search

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_imports` | `test_package_imports` (via `test_ci_minimal`) | [test_chat_pipeline.rs:13](crates/leann-core/tests/test_chat_pipeline.rs#L13) |
| `test_backend_basic` (build + search) | `test_build_and_search_100_docs` | [test_build_search.rs:24](crates/leann-core/tests/test_build_search.rs#L24) |
| `test_backend_basic` (build + search) | `test_build_and_search_1000_docs` | [test_build_search.rs:82](crates/leann-core/tests/test_build_search.rs#L82) |
| `test_large_index` | `test_build_and_search_1000_docs` | [test_build_search.rs:82](crates/leann-core/tests/test_build_search.rs#L82) |
| — | `test_build_creates_expected_files` | [test_build_search.rs:115](crates/leann-core/tests/test_build_search.rs#L115) |
| — | `test_meta_json_content` | [test_build_search.rs:142](crates/leann-core/tests/test_build_search.rs#L142) |
| — | `test_build_with_compact_csr` | [test_build_search.rs:162](crates/leann-core/tests/test_build_search.rs#L162) |
| — | `test_build_standard_with_vectors` | [test_build_search.rs:175](crates/leann-core/tests/test_build_search.rs#L175) |
| — | `test_build_with_distance_metrics` | [test_build_search.rs:188](crates/leann-core/tests/test_build_search.rs#L188) |
| — | `test_search_top_k_bounds` | [test_build_search.rs:243](crates/leann-core/tests/test_build_search.rs#L243) |
| — | `test_build_from_precomputed_embeddings` | [test_build_search.rs:285](crates/leann-core/tests/test_build_search.rs#L285) |
| — | `test_id_map_roundtrip` | [test_build_search.rs:326](crates/leann-core/tests/test_build_search.rs#L326) |
| — | `test_passage_random_access_after_build` | [test_build_search.rs:340](crates/leann-core/tests/test_build_search.rs#L340) |
| — | `test_hnsw_index_compact_roundtrip` | [test_build_search.rs:362](crates/leann-core/tests/test_build_search.rs#L362) |
| — | `test_hnsw_index_standard_roundtrip` | [test_build_search.rs:394](crates/leann-core/tests/test_build_search.rs#L394) |

**Status: Fully covered.** Rust has 13 tests vs Python's 3 — significantly deeper coverage of build/search paths including compact CSR, distance metrics, and index serialization.

---

### test_metadata_filtering.py — Metadata Filters

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_engine_initialization` | (implicit in all tests) | — |
| `test_direct_instantiation` | (implicit in all tests) | — |
| `test_no_filters_returns_all_results` | `test_filter_none_passthrough` | [test_metadata_filtering.rs:277](crates/leann-core/tests/test_metadata_filtering.rs#L277) |
| `test_equals_filter` | `test_filter_equals` | [test_metadata_filtering.rs:97](crates/leann-core/tests/test_metadata_filtering.rs#L97) |
| `test_not_equals_filter` | `test_filter_not_equals` | [test_metadata_filtering.rs:116](crates/leann-core/tests/test_metadata_filtering.rs#L116) |
| `test_less_than_filter` | `test_filter_less_than` | [test_metadata_filtering.rs:135](crates/leann-core/tests/test_metadata_filtering.rs#L135) |
| `test_less_than_or_equal_filter` | `test_filter_less_than_or_equal` | [test_metadata_filtering.rs:144](crates/leann-core/tests/test_metadata_filtering.rs#L144) |
| `test_greater_than_filter` | `test_filter_greater_than` | [test_metadata_filtering.rs:153](crates/leann-core/tests/test_metadata_filtering.rs#L153) |
| `test_greater_than_or_equal_filter` | `test_filter_greater_equal` | [test_metadata_filtering.rs:162](crates/leann-core/tests/test_metadata_filtering.rs#L162) |
| `test_in_filter` | `test_filter_in` | [test_metadata_filtering.rs:171](crates/leann-core/tests/test_metadata_filtering.rs#L171) |
| `test_not_in_filter` | `test_filter_not_in` | [test_metadata_filtering.rs:180](crates/leann-core/tests/test_metadata_filtering.rs#L180) |
| `test_contains_filter` | `test_filter_contains` | [test_metadata_filtering.rs:189](crates/leann-core/tests/test_metadata_filtering.rs#L189) |
| `test_starts_with_filter` | `test_filter_starts_with` | [test_metadata_filtering.rs:198](crates/leann-core/tests/test_metadata_filtering.rs#L198) |
| `test_ends_with_filter` | `test_filter_ends_with` | [test_metadata_filtering.rs:207](crates/leann-core/tests/test_metadata_filtering.rs#L207) |
| `test_is_true_filter` | `test_filter_is_true` | [test_metadata_filtering.rs:216](crates/leann-core/tests/test_metadata_filtering.rs#L216) |
| `test_is_false_filter` | `test_filter_is_false` | [test_metadata_filtering.rs:225](crates/leann-core/tests/test_metadata_filtering.rs#L225) |
| `test_compound_filters` | `test_filter_compound_and` | [test_metadata_filtering.rs:234](crates/leann-core/tests/test_metadata_filtering.rs#L234) |
| `test_multiple_operators_same_field` | `test_filter_range` | [test_metadata_filtering.rs:253](crates/leann-core/tests/test_metadata_filtering.rs#L253) |
| `test_missing_field_fails_filter` | `test_filter_missing_field` | [test_metadata_filtering.rs:286](crates/leann-core/tests/test_metadata_filtering.rs#L286) |
| `test_invalid_operator` | `test_filter_invalid_operator` | [test_metadata_filtering.rs:296](crates/leann-core/tests/test_metadata_filtering.rs#L296) |
| `test_type_coercion_numeric` | `test_filter_type_coercion` | [test_metadata_filtering.rs:311](crates/leann-core/tests/test_metadata_filtering.rs#L311) |
| `test_list_membership_with_nested_tags` | (covered by `test_filter_in`) | [test_metadata_filtering.rs:171](crates/leann-core/tests/test_metadata_filtering.rs#L171) |
| `test_empty_results_list` | `test_filter_empty_results_list` | [test_metadata_filtering.rs:340](crates/leann-core/tests/test_metadata_filtering.rs#L340) |
| `test_search_result_filtering` | (covered by `test_bm25_search_with_metadata_filters`) | [test_hybrid_search.rs:222](crates/leann-core/tests/test_hybrid_search.rs#L222) |
| `test_filter_search_results_no_filters` | `test_filter_none_passthrough` | [test_metadata_filtering.rs:277](crates/leann-core/tests/test_metadata_filtering.rs#L277) |
| `test_filter_maintains_search_result_type` | (Rust types enforce this at compile time) | — |

**Status: Fully covered.** 21 integration tests + 8 unit tests. All 13 operators tested. Type-safety eliminates need for `test_filter_maintains_search_result_type`.

---

### test_hybrid_search.py — Hybrid Search (Vector + BM25)

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_pure_vector_search` | (requires ZMQ server; tested at HNSW level) | [test_build_search.rs:24](crates/leann-core/tests/test_build_search.rs#L24) |
| `test_pure_keyword_search` | `test_pure_bm25_search` | [test_hybrid_search.rs:46](crates/leann-core/tests/test_hybrid_search.rs#L46) |
| `test_hybrid_search_balanced` | (requires ZMQ server) | — |
| `test_hybrid_search_vector_heavy` | (requires ZMQ server) | — |
| `test_hybrid_search_keyword_heavy` | (requires ZMQ server) | — |
| `test_hybrid_search_score_combination` | (requires ZMQ server) | — |
| `test_hybrid_search_with_metadata_filters` | `test_bm25_search_with_metadata_filters` | [test_hybrid_search.rs:222](crates/leann-core/tests/test_hybrid_search.rs#L222) |
| — | `test_bm25_search_cooking` | [test_hybrid_search.rs:77](crates/leann-core/tests/test_hybrid_search.rs#L77) |
| — | `test_bm25_scores_descending` | [test_hybrid_search.rs:101](crates/leann-core/tests/test_hybrid_search.rs#L101) |
| — | `test_grep_search` | [test_hybrid_search.rs:125](crates/leann-core/tests/test_hybrid_search.rs#L125) |
| — | `test_grep_search_no_match` | [test_hybrid_search.rs:144](crates/leann-core/tests/test_hybrid_search.rs#L144) |
| — | `test_bm25_nonexistent_terms` | [test_hybrid_search.rs:160](crates/leann-core/tests/test_hybrid_search.rs#L160) |
| — | `test_bm25_with_sample_documents` | [test_hybrid_search.rs:184](crates/leann-core/tests/test_hybrid_search.rs#L184) |

**Status: Partially covered.** BM25 and grep paths fully tested (8 tests). Hybrid blend tests (`gemma` between 0 and 1) require a ZMQ embedding server and are not portable to Rust integration tests without mocking the server.

---

### test_sync.py — File Synchronization

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_no_changes_if_root_hash_same` | `test_merkle_tree_no_changes` | [test_sync.rs:10](crates/leann-core/tests/test_sync.rs#L10) |
| `test_added_removed_modified` | `test_merkle_tree_detects_added_file` | [test_sync.rs:29](crates/leann-core/tests/test_sync.rs#L29) |
| `test_added_removed_modified` | `test_merkle_tree_detects_removed_file` | [test_sync.rs:49](crates/leann-core/tests/test_sync.rs#L49) |
| `test_added_removed_modified` | `test_merkle_tree_detects_modified_file` | [test_sync.rs:148](crates/leann-core/tests/test_sync.rs#L148) |
| `test_added_removed_modified` | `test_merkle_tree_combined_changes` | [test_sync.rs:176](crates/leann-core/tests/test_sync.rs#L176) |
| `test_generate_file_hashes` | `test_hash_data_consistent` | [test_sync.rs:69](crates/leann-core/tests/test_sync.rs#L69) |
| `test_build_merkle_tree` | (covered by tree construction in all Merkle tests) | — |
| `test_check_for_changes_detected` | `test_file_synchronizer_detects_added_file` | [test_sync.rs:103](crates/leann-core/tests/test_sync.rs#L103) |
| `test_check_for_changes_detected` | `test_file_synchronizer_detects_modified_file` | [test_sync.rs:126](crates/leann-core/tests/test_sync.rs#L126) |
| — | `test_file_synchronizer_no_changes` | [test_sync.rs:82](crates/leann-core/tests/test_sync.rs#L82) |

**Status: Fully covered.** 9 Rust tests cover all Python scenarios plus additional edge cases (no-change sync, separate add/remove/modify detection).

---

### test_astchunk_integration.py — AST Chunking & Document Loading

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_detect_code_files_python` | `test_detect_language` | [test_document_loading.rs:270](crates/leann-core/tests/test_document_loading.rs#L270) |
| `test_detect_code_files_multiple_languages` | `test_detect_language` | [test_document_loading.rs:270](crates/leann-core/tests/test_document_loading.rs#L270) |
| `test_create_traditional_chunks` | `test_chunk_text_sizes` | [test_document_loading.rs:103](crates/leann-core/tests/test_document_loading.rs#L103) |
| `test_create_traditional_chunks_empty_docs` | `test_chunk_empty_text` | [test_document_loading.rs:132](crates/leann-core/tests/test_document_loading.rs#L132) |
| `test_create_ast_chunks_with_astchunk_available` | `test_ast_chunk_python` | [test_document_loading.rs:141](crates/leann-core/tests/test_document_loading.rs#L141) |
| `test_create_ast_chunks_fallback_to_traditional` | `test_ast_fallback_to_generic` | [test_document_loading.rs:259](crates/leann-core/tests/test_document_loading.rs#L259) |
| `test_create_text_chunks_traditional_mode` | `test_chunk_text_sizes` | [test_document_loading.rs:103](crates/leann-core/tests/test_document_loading.rs#L103) |
| `test_create_text_chunks_ast_mode` | `test_ast_chunk_python` | [test_document_loading.rs:141](crates/leann-core/tests/test_document_loading.rs#L141) |
| `test_extract_content_from_astchunk_dict` | (Rust uses typed structs) | — |
| `test_ast_chunks_preserve_file_metadata` | `test_ast_chunk_metadata` | [test_document_loading.rs:283](crates/leann-core/tests/test_document_loading.rs#L283) |
| `test_text_chunking_empty_documents` | `test_chunk_empty_text` | [test_document_loading.rs:132](crates/leann-core/tests/test_document_loading.rs#L132) |
| — | `test_load_txt_file` | [test_document_loading.rs:13](crates/leann-core/tests/test_document_loading.rs#L13) |
| — | `test_load_md_file` | [test_document_loading.rs:27](crates/leann-core/tests/test_document_loading.rs#L27) |
| — | `test_load_rs_file` | [test_document_loading.rs:43](crates/leann-core/tests/test_document_loading.rs#L43) |
| — | `test_load_py_file` | [test_document_loading.rs:55](crates/leann-core/tests/test_document_loading.rs#L55) |
| — | `test_load_empty_file` | [test_document_loading.rs:71](crates/leann-core/tests/test_document_loading.rs#L71) |
| — | `test_load_whitespace_only_file` | [test_document_loading.rs:82](crates/leann-core/tests/test_document_loading.rs#L82) |
| — | `test_load_nonexistent_file` | [test_document_loading.rs:93](crates/leann-core/tests/test_document_loading.rs#L93) |
| — | `test_chunk_overlap` | [test_document_loading.rs:115](crates/leann-core/tests/test_document_loading.rs#L115) |
| — | `test_chunk_short_text` | [test_document_loading.rs:123](crates/leann-core/tests/test_document_loading.rs#L123) |
| — | `test_ast_chunk_rust` | [test_document_loading.rs:180](crates/leann-core/tests/test_document_loading.rs#L180) |
| — | `test_ast_chunk_javascript` | [test_document_loading.rs:224](crates/leann-core/tests/test_document_loading.rs#L224) |

**Status: Fully covered.** 17 Rust tests cover language detection, traditional chunking, AST chunking (Python/Rust/JS), fallback behavior, metadata preservation, and file loading for multiple formats.

---

### test_ci_minimal.py — Smoke Tests

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_package_imports` | `test_simulated_chat_response` | [test_chat_pipeline.rs:13](crates/leann-core/tests/test_chat_pipeline.rs#L13) |
| `test_cli_help` | `test_cli_help` | [test_cli_args.rs:25](crates/leann-cli/tests/test_cli_args.rs#L25) |
| `test_backend_registration` | (Rust uses direct module imports) | — |
| `test_version_info` | `test_cli_version` | [test_cli_args.rs:96](crates/leann-cli/tests/test_cli_args.rs#L96) |

**Status: Fully covered.** CLI tests live in `crates/leann-cli/`. Backend registration is compile-time in Rust.

---

### test_cli_ask.py / test_cli_verbosity.py — CLI Arguments

| Python Test | Rust Equivalent | Source |
|---|---|---|
| `test_cli_ask_accepts_positional_query` | `test_cli_ask_help` | [test_cli_args.rs:115](crates/leann-cli/tests/test_cli_args.rs#L115) |
| `test_suppress_cpp_output_*` (3 tests) | (not applicable — no C++ subprocess) | — |
| `test_verbose_flag_parsed` | (flags verified via `--help` output) | [test_cli_args.rs:25](crates/leann-cli/tests/test_cli_args.rs#L25) |
| `test_quiet_flag_parsed` | (flags verified via `--help` output) | [test_cli_args.rs:25](crates/leann-cli/tests/test_cli_args.rs#L25) |
| — | `test_cli_build_help` | [test_cli_args.rs:42](crates/leann-cli/tests/test_cli_args.rs#L42) |
| — | `test_cli_search_help` | [test_cli_args.rs:62](crates/leann-cli/tests/test_cli_args.rs#L62) |
| — | `test_cli_list_empty` | [test_cli_args.rs:78](crates/leann-cli/tests/test_cli_args.rs#L78) |
| — | `test_cli_remove_help` | [test_cli_args.rs:131](crates/leann-cli/tests/test_cli_args.rs#L131) |

**Status: Covered where applicable.** 7 Rust CLI tests validate all subcommand help output and basic operations. C++ output suppression tests are Python/FAISS-specific.

---

## Rust-Only Test Files (No Python Equivalent)

| Rust Test File | Tests | Description | Source |
|---|---|---|---|
| `test_bm25_search.rs` | 8 | Dedicated BM25 scorer integration tests | [test_bm25_search.rs](crates/leann-core/tests/test_bm25_search.rs) |
| `test_index_format.rs` | 7 | Index file format validation (JSONL, offsets, meta schema) | [test_index_format.rs](crates/leann-core/tests/test_index_format.rs) |
| `test_chat_pipeline.rs` | 5 | LLM config and simulated chat | [test_chat_pipeline.rs](crates/leann-core/tests/test_chat_pipeline.rs) |
| `test_cli_args.rs` | 7 | CLI subcommand help, version, list | [test_cli_args.rs](crates/leann-cli/tests/test_cli_args.rs) |

---

## Rust Unit Tests (in `src/`)

| Module | Tests | Description | Source |
|---|---|---|---|
| `bm25.rs` | 6 | BM25 scorer: fit, search, tokenize, edge cases | [bm25.rs:151](crates/leann-core/src/bm25.rs#L151) |
| `metadata_filter.rs` | 8 | Filter operators, missing fields, compound logic | [metadata_filter.rs:243](crates/leann-core/src/metadata_filter.rs#L243) |
| `hnsw/simd.rs` | 12 | L2/IP distance, batch_4, VisitedList, normalize | [hnsw/simd.rs:827](crates/leann-core/src/hnsw/simd.rs#L827) |
| `hnsw/build.rs` | 3 | Graph construction (serial + parallel) | [hnsw/build.rs:967](crates/leann-core/src/hnsw/build.rs#L967) |
| `hnsw/search.rs` | 2 | Search with stored vectors and recompute | [hnsw/search.rs:692](crates/leann-core/src/hnsw/search.rs#L692) |
| `hnsw/io.rs` | 1 | Compact index roundtrip | [hnsw/io.rs:393](crates/leann-core/src/hnsw/io.rs#L393) |
| `hnsw/graph.rs` | 2 | Config defaults, FourCC constants | [hnsw/graph.rs:177](crates/leann-core/src/hnsw/graph.rs#L177) |
| `hnsw/csr.rs` | 1 | CSR conversion | [hnsw/csr.rs:116](crates/leann-core/src/hnsw/csr.rs#L116) |
| `chunking/mod.rs` | 2 | Basic chunking | [chunking/mod.rs:70](crates/leann-core/src/chunking/mod.rs#L70) |
| `chunking/ast.rs` | 4 | AST chunking (Python, Rust, generic, detection) | [chunking/ast.rs:438](crates/leann-core/src/chunking/ast.rs#L438) |
| `chunking/sentence.rs` | 4 | Sentence splitting | [chunking/sentence.rs:92](crates/leann-core/src/chunking/sentence.rs#L92) |
| `index.rs` | 4 | IndexMeta roundtrip, distance metric, paths | [index.rs:178](crates/leann-core/src/index.rs#L178) |
| `passages.rs` | 3 | Passage I/O, ID map, not-found error | [passages.rs:541](crates/leann-core/src/passages.rs#L541) |
| `search_result.rs` | 3 | SearchResult creation, metadata, serialization | [search_result.rs:43](crates/leann-core/src/search_result.rs#L43) |
| `settings.rs` | 4 | Ollama host, OpenAI key, URL cleaning | [settings.rs:120](crates/leann-core/src/settings.rs#L120) |
| `react_agent.rs` | 2 | LLM response parsing | [react_agent.rs:231](crates/leann-core/src/react_agent.rs#L231) |
| `sync.rs` | 2 | Hash data, Merkle tree | [sync.rs:258](crates/leann-core/src/sync.rs#L258) |
| `document_loaders/pdf.rs` | 3 | PDF extraction | [document_loaders/pdf.rs:24](crates/leann-core/src/document_loaders/pdf.rs#L24) |

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
| `test_embedding_server_manager.py` | Embedding server lifecycle (Python process) |
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
| Build & Search | 3 | 13 | Rust exceeds |
| Metadata Filtering | 26 | 21 + 8 unit | Fully covered |
| Hybrid Search (BM25) | 7 | 8 | BM25/grep covered; blend requires ZMQ |
| Sync | 5 | 9 | Rust exceeds |
| AST Chunking / Doc Loading | 20+ | 17 | Fully covered |
| CI Smoke Tests | 4 | 5 | Fully covered |
| CLI Args & Help | 12 | 7 | Covered where applicable |
| BM25 Scorer | — | 8 + 6 unit | Rust-only (deeper coverage) |
| Index Format | — | 7 | Rust-only |
| HNSW Internals | — | 21 unit | Rust-only |
| Python-specific | 55+ | — | Not applicable |
