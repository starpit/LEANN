"""Tests for the LEANN Python bindings (PyO3).

These tests exercise the compiled wheel's API surface.
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
# LeannBuilder construction tests
# ---------------------------------------------------------------------------


class TestBuilderConstruction:
    def test_basic_construction(self):
        builder = LeannBuilder("model", dimensions=128)
        assert builder is not None

    def test_with_embedding_mode(self):
        builder = LeannBuilder("model", dimensions=64, embedding_mode="openai")
        assert builder is not None

    def test_kwargs_m_ef(self):
        builder = LeannBuilder(
            "model", dimensions=128, m=32, ef_construction=200
        )
        assert builder is not None

    def test_kwargs_compact_recompute(self):
        builder = LeannBuilder(
            "model", dimensions=128, compact=False, recompute=False
        )
        assert builder is not None

    def test_add_text_plain(self):
        builder = LeannBuilder("model", dimensions=32)
        builder.add_text("Hello, world!")
        # Should not raise

    def test_add_text_with_metadata(self):
        builder = LeannBuilder("model", dimensions=32)
        builder.add_text("Hello", {"key": "value"})

    def test_add_text_metadata_types(self):
        """Metadata supports str, int, float, and bool values."""
        builder = LeannBuilder("model", dimensions=32)
        builder.add_text(
            "Hello",
            {"s": "string", "i": 42, "f": 3.14, "b": True},
        )


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_searcher_missing_index(self):
        with pytest.raises(RuntimeError):
            LeannSearcher("/nonexistent/path/to/index")

    def test_chat_missing_index(self):
        with pytest.raises(RuntimeError):
            LeannChat("/nonexistent/path/to/index")


# ---------------------------------------------------------------------------
# ReActAgent construction
# ---------------------------------------------------------------------------


class TestReActAgent:
    def test_construction(self):
        agent = ReActAgent("/some/path", max_iterations=3)
        assert agent is not None

    def test_default_max_iterations(self):
        agent = ReActAgent("/some/path")
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
            # Simple LCG-based pseudo-random
            val = math.sin(seed * (i * dims + j + 1)) * 10000
            row.append(val - math.floor(val))  # fractional part in [0, 1)
        embeddings.append(row)
    return embeddings


class TestBuildAndSearch:
    """Integration tests that build a real index and search it."""

    @pytest.fixture()
    def index_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def _build_test_index(self, index_dir: str, n: int = 20, dims: int = 32):
        """Helper: build a small test index with pre-computed embeddings."""
        builder = LeannBuilder(
            "test-model", dimensions=dims, compact=False, recompute=False
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
        """Build index from embeddings, search, get results."""
        index_path = self._build_test_index(index_dir)
        searcher = LeannSearcher(index_path)
        # Search won't use real embeddings (no ZMQ server), but we can
        # test that the searcher opens without error and the cleanup works.
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
        builder = LeannBuilder("model", dimensions=32, compact=False, recompute=False)
        builder.add_text("text")
        with pytest.raises(ValueError):
            builder.build_index_from_embeddings(
                os.path.join(index_dir, "empty"), [], []
            )

    def test_build_mismatched_ids_raises(self, index_dir):
        """Mismatched ids/embeddings count raises RuntimeError."""
        builder = LeannBuilder("model", dimensions=32, compact=False, recompute=False)
        builder.add_text("text")
        with pytest.raises(RuntimeError):
            builder.build_index_from_embeddings(
                os.path.join(index_dir, "mismatch"),
                ["id_0"],
                [[0.1] * 32, [0.2] * 32],  # 2 embeddings but 1 id
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
        """Build with compact=True creates a valid index."""
        builder = LeannBuilder(
            "test-model", dimensions=32, compact=True, recompute=True
        )
        n = 20
        ids = [f"doc_{i}" for i in range(n)]
        for i in range(n):
            builder.add_text(f"Document {i}")
        embeddings = _random_embeddings(n, 32)
        index_path = os.path.join(index_dir, "compact_index")
        builder.build_index_from_embeddings(index_path, ids, embeddings)
        assert os.path.exists(index_path + ".index")
