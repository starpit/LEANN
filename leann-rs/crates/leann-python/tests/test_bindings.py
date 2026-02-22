"""Tests for the LEANN Python bindings (PyO3).

These tests exercise the compiled wheel's API surface using both
the Python-compatible calling convention (matching leann.api) and
the Rust-native calling convention.

Integration tests use build_index_from_embeddings to avoid
requiring an embedding server or network access.
"""

import math
import os
import tempfile

import pytest

import leann
from leann import LeannBuilder, LeannChat, LeannSearcher, ReActAgent, SearchResult


# ---------------------------------------------------------------------------
# Import / introspection tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_all_classes(self):
        """All 5 classes are importable from the top-level package."""
        assert LeannBuilder is not None
        assert LeannSearcher is not None
        assert LeannChat is not None
        assert ReActAgent is not None
        assert SearchResult is not None

    def test_version_exists(self):
        assert hasattr(leann, "__version__")
        assert isinstance(leann.__version__, str)
        assert leann.__version__ == "0.1.0"

    def test_all_exports(self):
        assert hasattr(leann, "__all__")
        expected = {"LeannBuilder", "LeannSearcher", "LeannChat", "SearchResult", "ReActAgent"}
        assert set(leann.__all__) == expected


# ---------------------------------------------------------------------------
# LeannBuilder construction — Python-compatible calling convention
# ---------------------------------------------------------------------------


class TestBuilderPythonStyle:
    """Test the Python API calling convention (matching leann.api.LeannBuilder)."""

    def test_basic_construction(self):
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
        )
        assert builder is not None

    def test_with_dimensions(self):
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="model",
            dimensions=128,
        )
        assert builder is not None

    def test_python_style_kwargs(self):
        """Python API uses M, efConstruction, is_compact, is_recompute."""
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="model",
            dimensions=128,
            M=16,
            efConstruction=200,
        )
        assert builder is not None

    def test_compact_recompute_python_style(self):
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="model",
            dimensions=128,
            is_compact=False,
            is_recompute=False,
        )
        assert builder is not None

    def test_all_python_kwargs(self):
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="model",
            dimensions=64,
            embedding_mode="openai",
            M=32,
            efConstruction=200,
            is_compact=True,
            is_recompute=True,
        )
        assert builder is not None

    def test_unsupported_backend_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            LeannBuilder(backend_name="diskann", embedding_model="model", dimensions=64)

    def test_positional_backend_name(self):
        """backend_name can also be passed positionally."""
        builder = LeannBuilder("hnsw")
        assert builder is not None


# ---------------------------------------------------------------------------
# LeannBuilder construction — Rust-style calling convention
# ---------------------------------------------------------------------------


class TestBuilderRustStyle:
    """Test the Rust-native kwarg names (m, ef_construction, compact, recompute)."""

    def test_rust_style_kwargs(self):
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="model",
            dimensions=128,
            m=32,
            ef_construction=200,
        )
        assert builder is not None

    def test_compact_recompute_rust_style(self):
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="model",
            dimensions=128,
            compact=False,
            recompute=False,
        )
        assert builder is not None

    def test_defaults_only(self):
        """All defaults: backend_name='hnsw', embedding_model='facebook/contriever'."""
        builder = LeannBuilder()
        assert builder is not None


# ---------------------------------------------------------------------------
# add_text tests
# ---------------------------------------------------------------------------


class TestAddText:
    def test_add_text_plain(self):
        builder = LeannBuilder(backend_name="hnsw", dimensions=32)
        builder.add_text("Hello, world!")

    def test_add_text_with_metadata(self):
        builder = LeannBuilder(backend_name="hnsw", dimensions=32)
        builder.add_text("Hello", {"key": "value"})

    def test_add_text_metadata_types(self):
        """Metadata supports str, int, float, and bool values."""
        builder = LeannBuilder(backend_name="hnsw", dimensions=32)
        builder.add_text(
            "Hello",
            {"s": "string", "i": 42, "f": 3.14, "b": True},
        )


# ---------------------------------------------------------------------------
# LeannSearcher construction
# ---------------------------------------------------------------------------


class TestSearcherConstruction:
    def test_missing_index_raises(self):
        with pytest.raises(RuntimeError):
            LeannSearcher("/nonexistent/path/to/index")

    def test_python_style_params(self):
        """Searcher accepts enable_warmup and recompute_embeddings."""
        with pytest.raises(RuntimeError):
            LeannSearcher(
                "/nonexistent/path/to/index",
                enable_warmup=False,
                recompute_embeddings=False,
            )


# ---------------------------------------------------------------------------
# LeannChat construction
# ---------------------------------------------------------------------------


class TestChatConstruction:
    def test_missing_index_raises(self):
        with pytest.raises(RuntimeError):
            LeannChat("/nonexistent/path/to/index")

    def test_python_style_params(self):
        """Chat accepts enable_warmup kwarg."""
        with pytest.raises(RuntimeError):
            LeannChat("/nonexistent/path/to/index", enable_warmup=True)


# ---------------------------------------------------------------------------
# ReActAgent construction
# ---------------------------------------------------------------------------


class TestReActAgent:
    def test_construction_with_path(self):
        """Accept a string path (convenience)."""
        agent = ReActAgent("/some/path", max_iterations=3)
        assert agent is not None

    def test_default_max_iterations(self):
        agent = ReActAgent("/some/path")
        assert agent is not None

    def test_python_style_with_llm_config(self):
        """Accept llm_config dict like the Python API."""
        agent = ReActAgent("/some/path", llm_config={"type": "openai", "model": "gpt-4"})
        assert agent is not None


# ---------------------------------------------------------------------------
# Integration tests — build + search roundtrip
# ---------------------------------------------------------------------------


def _random_embeddings(n: int, dims: int, seed: int = 42) -> list[list[float]]:
    """Generate deterministic pseudo-random embeddings without numpy."""
    embeddings = []
    for i in range(n):
        row = []
        for j in range(dims):
            val = math.sin(seed * (i * dims + j + 1)) * 10000
            row.append(val - math.floor(val))
        embeddings.append(row)
    return embeddings


class TestBuildAndSearch:
    """Integration tests that build a real index and search it."""

    @pytest.fixture()
    def index_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def _build_test_index(
        self, index_dir: str, n: int = 20, dims: int = 32, python_style: bool = False
    ):
        """Helper: build a small test index with pre-computed embeddings.

        If python_style=True, uses the Python-compatible kwarg names.
        """
        if python_style:
            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                dimensions=dims,
                is_compact=False,
                is_recompute=False,
            )
        else:
            builder = LeannBuilder(
                backend_name="hnsw",
                embedding_model="test-model",
                dimensions=dims,
                compact=False,
                recompute=False,
            )
        ids = [f"doc_{i}" for i in range(n)]
        for i, doc_id in enumerate(ids):
            builder.add_text(
                f"This is test document number {i} about topic {i % 5}",
                {"id": doc_id, "doc_num": i, "topic": f"topic_{i % 5}"},
            )
        embeddings = _random_embeddings(n, dims)
        index_path = os.path.join(index_dir, "test_index")
        builder.build_index_from_embeddings(index_path, ids, embeddings)
        return index_path

    def test_roundtrip(self, index_dir):
        """Build index from embeddings, open searcher, cleanup."""
        index_path = self._build_test_index(index_dir)
        searcher = LeannSearcher(index_path)
        searcher.cleanup()

    def test_roundtrip_python_style(self, index_dir):
        """Same roundtrip using Python-style kwargs."""
        index_path = self._build_test_index(index_dir, python_style=True)
        searcher = LeannSearcher(index_path, enable_warmup=False, recompute_embeddings=False)
        searcher.cleanup()

    def test_index_files_created(self, index_dir):
        """Build creates the expected index files."""
        index_path = self._build_test_index(index_dir)
        assert os.path.exists(index_path + ".meta.json")
        assert os.path.exists(index_path + ".passages.jsonl")
        assert os.path.exists(index_path + ".passages.idx")
        assert os.path.exists(index_path + ".index")

    def test_index_files_nonempty(self, index_dir):
        """All index files have non-zero size."""
        index_path = self._build_test_index(index_dir)
        for suffix in [".meta.json", ".passages.jsonl", ".passages.idx", ".index"]:
            size = os.path.getsize(index_path + suffix)
            assert size > 0, f"{suffix} is empty"

    def test_meta_json_content(self, index_dir):
        """Meta JSON has correct fields."""
        import json

        index_path = self._build_test_index(index_dir, n=10, dims=32)
        with open(index_path + ".meta.json") as f:
            meta = json.load(f)
        assert meta["backend_name"] == "hnsw"
        assert meta["embedding_model"] == "test-model"
        assert meta["dimensions"] == 32
        assert "passage_sources" in meta

    def test_build_empty_embeddings_raises(self, index_dir):
        """Empty embeddings list raises ValueError."""
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="model",
            dimensions=32,
            is_compact=False,
            is_recompute=False,
        )
        builder.add_text("text")
        with pytest.raises(ValueError):
            builder.build_index_from_embeddings(
                os.path.join(index_dir, "empty"), [], []
            )

    def test_build_mismatched_ids_raises(self, index_dir):
        """Mismatched ids/embeddings count raises RuntimeError."""
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="model",
            dimensions=32,
            compact=False,
            recompute=False,
        )
        builder.add_text("text")
        with pytest.raises(RuntimeError):
            builder.build_index_from_embeddings(
                os.path.join(index_dir, "mismatch"),
                ["id_0"],
                [[0.1] * 32, [0.2] * 32],
            )

    def test_searcher_opens_built_index(self, index_dir):
        """LeannSearcher can open an index built via build_index_from_embeddings."""
        index_path = self._build_test_index(index_dir, n=50, dims=64)
        searcher = LeannSearcher(index_path)
        assert searcher is not None
        searcher.cleanup()

    def test_build_many_docs(self, index_dir):
        """Build with 200 docs succeeds without error."""
        index_path = self._build_test_index(index_dir, n=200, dims=64)
        assert os.path.exists(index_path + ".index")

    def test_build_compact_index(self, index_dir):
        """Build with is_compact=True creates a valid index."""
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="test-model",
            dimensions=32,
            is_compact=True,
            is_recompute=True,
        )
        n = 20
        ids = [f"doc_{i}" for i in range(n)]
        for i in range(n):
            builder.add_text(f"Document {i}")
        embeddings = _random_embeddings(n, 32)
        index_path = os.path.join(index_dir, "compact_index")
        builder.build_index_from_embeddings(index_path, ids, embeddings)
        assert os.path.exists(index_path + ".index")

    def test_react_agent_with_searcher(self, index_dir):
        """ReActAgent accepts a LeannSearcher (Python API convention)."""
        index_path = self._build_test_index(index_dir)
        searcher = LeannSearcher(index_path)
        agent = ReActAgent(searcher, max_iterations=3)
        assert agent is not None
        searcher.cleanup()
