"""Type stubs for the native leann.leann module (compiled Rust/PyO3)."""

from typing import Any

def get_registered_backends() -> list[str]:
    """Get list of registered backend names."""
    ...

class SearchResult:
    """Search result returned from LEANN queries."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any]
    def __repr__(self) -> str: ...

class LeannBuilder:
    """Builder for creating LEANN indexes."""

    def __init__(
        self,
        backend_name: str = "hnsw",
        embedding_model: str = "facebook/contriever",
        dimensions: int | None = None,
        embedding_mode: str = "sentence-transformers",
        embedding_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def add_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def build_index(self, index_path: str) -> None: ...
    def build_index_from_embeddings(
        self,
        index_path: str,
        ids: list[str],
        embeddings: list[list[float]],
    ) -> None: ...

class LeannSearcher:
    """Searcher for querying LEANN indexes."""

    def __init__(
        self,
        index_path: str,
        enable_warmup: bool = True,
        recompute_embeddings: bool = True,
        **kwargs: Any,
    ) -> None: ...
    def search(
        self, query: str, top_k: int = 5, **kwargs: Any
    ) -> list[SearchResult]: ...
    def cleanup(self) -> None: ...

class LeannChat:
    """RAG chat interface combining search + LLM."""

    def __init__(
        self,
        index_path: str,
        llm_config: dict[str, Any] | None = None,
        enable_warmup: bool = False,
        **kwargs: Any,
    ) -> None: ...
    def ask(self, question: str, top_k: int = 5, **kwargs: Any) -> str: ...
    def cleanup(self) -> None: ...

class ReActAgent:
    """Multi-turn reasoning agent."""

    def __init__(
        self,
        searcher: LeannSearcher | str,
        llm: Any | None = None,
        llm_config: dict[str, Any] | None = None,
        max_iterations: int = 5,
    ) -> None: ...
    def run(self, question: str, top_k: int = 5) -> str: ...
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...
