"""LEANN - Lightweight vector database and RAG system.

This module provides Python bindings to the Rust LEANN core library,
offering high-performance vector search with graph-based selective
embedding recomputation.
"""

from leann.leann import (
    LeannBuilder,
    LeannChat,
    LeannSearcher,
    ReActAgent,
    SearchResult,
    get_registered_backends,
)

__all__ = [
    "LeannBuilder",
    "LeannSearcher",
    "LeannChat",
    "SearchResult",
    "ReActAgent",
    "get_registered_backends",
]

__version__ = "0.1.0"
