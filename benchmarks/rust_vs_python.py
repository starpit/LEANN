#!/usr/bin/env python3
"""
Benchmark FAISS HNSW (Python/C++) at matching parameters to the Rust implementation.

Uses the custom FAISS fork from leann_backend_hnsw.
Outputs JSON results to stdout when --json is passed.

Usage:
    uv run python benchmarks/rust_vs_python.py [--json] [--skip-large]
"""

import argparse
import json
import math
import sys
import time
from collections import OrderedDict

import numpy as np


def get_faiss():
    """Import faiss from the leann backend."""
    try:
        from leann_backend_hnsw import faiss

        return faiss
    except ImportError:
        try:
            import faiss

            return faiss
        except ImportError:
            print(
                "ERROR: Could not import faiss. Install via:\n"
                "  uv sync --extra diskann  # or pip install faiss-cpu",
                file=sys.stderr,
            )
            sys.exit(1)


def quantile(sorted_times, q):
    """Compute quantile from a sorted list (linear interpolation)."""
    if len(sorted_times) == 1:
        return sorted_times[0]
    pos = q * (len(sorted_times) - 1)
    lo = int(math.floor(pos))
    hi = lo + 1
    frac = pos - lo
    if hi >= len(sorted_times):
        return sorted_times[lo]
    return sorted_times[lo] * (1 - frac) + sorted_times[hi] * frac


def bench_fn(fn, warmup=3, repeats=10):
    """Run a benchmark: warmup, then timed repeats. Returns sorted list of elapsed seconds."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times.sort()
    return times


def make_result(times, scale=1.0):
    """Create result dict with quantiles from sorted times."""
    p5 = quantile(times, 0.05) * scale
    p25 = quantile(times, 0.25) * scale
    p50 = quantile(times, 0.50) * scale
    p75 = quantile(times, 0.75) * scale
    p95 = quantile(times, 0.95) * scale
    mean = sum(times) / len(times) * scale
    return {
        "p5_s": p5,
        "p25_s": p25,
        "p50_s": p50,
        "p75_s": p75,
        "p95_s": p95,
        "mean_s": mean,
        "n_iters": len(times),
        "median_s": p50,
        "median_ms": p50 * 1e3,
        "median_us": p50 * 1e6,
        "mean_ms": mean * 1e3,
        "mean_us": mean * 1e6,
    }


def make_size_result(size):
    return {
        "p5_s": float(size), "p25_s": float(size), "p50_s": float(size),
        "p75_s": float(size), "p95_s": float(size), "mean_s": float(size),
        "n_iters": 1,
        "median_s": float(size), "median_ms": float(size), "median_us": float(size),
        "mean_ms": float(size), "mean_us": float(size),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS HNSW")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument(
        "--skip-large", action="store_true", help="Skip 50K vector benchmarks"
    )
    args = parser.parse_args()

    import os

    faiss = get_faiss()
    rng = np.random.RandomState(42)
    results = OrderedDict()

    # Note: FAISS internal level-assignment RNG cannot be seeded from Python.
    # HNSW_SEED is used by the Rust benchmark for deterministic builds.
    hnsw_seed = os.environ.get("HNSW_SEED")
    if hnsw_seed:
        print(f"HNSW_SEED={hnsw_seed} (ignored — FAISS RNG not controllable)", file=sys.stderr)

    print("=== LEANN Python (FAISS C++) HNSW Benchmarks ===", file=sys.stderr)

    # ── Distance computation ────────────────────────────────────────
    print("\n[1/15] Distance computation...", file=sys.stderr)
    for dim in [128, 384, 768]:
        a = rng.rand(dim).astype(np.float32)
        b = rng.rand(dim).astype(np.float32)

        # L2 squared distance
        times = bench_fn(lambda: np.sum((a - b) ** 2), warmup=1000, repeats=10000)
        results[f"distance/l2/{dim}"] = make_result(times)

        # Inner product (negated)
        times = bench_fn(lambda: -np.dot(a, b), warmup=1000, repeats=10000)
        results[f"distance/ip/{dim}"] = make_result(times)

    # ── HNSW build ──────────────────────────────────────────────────
    print("[2/15] HNSW build...", file=sys.stderr)
    build_sizes = [100, 1_000, 10_000]
    if not args.skip_large:
        build_sizes.append(50_000)

    for n in build_sizes:
        print(f"  Building {n} vectors...", file=sys.stderr)
        data = rng.rand(n, 384).astype(np.float32)

        def build_faiss_hnsw(data=data):
            index = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_L2)
            index.hnsw.efConstruction = 200
            index.add(data)
            return index

        repeats = 5 if n >= 50_000 else 10 if n >= 10_000 else 20
        times = bench_fn(build_faiss_hnsw, warmup=1, repeats=repeats)
        results[f"hnsw_build/{n}"] = make_result(times)

    # ── HNSW search (stored vectors) ────────────────────────────────
    print("[3/15] HNSW search (stored vectors)...", file=sys.stderr)
    n = 10_000
    dim = 384
    top_k = 10

    data = rng.rand(n, dim).astype(np.float32)
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
    index.hnsw.efConstruction = 200
    index.add(data)

    query = rng.rand(1, dim).astype(np.float32)

    for ef in [16, 32, 64, 128, 256]:
        index.hnsw.efSearch = ef
        times = bench_fn(
            lambda: index.search(query, top_k), warmup=200, repeats=5000
        )
        results[f"hnsw_search/ef{ef}"] = make_result(times)

    # ── HNSW search (recompute simulation) ──────────────────────────
    # Note: FAISS doesn't have a native recompute mode, so we simulate
    # it by doing a normal search. This gives the baseline comparison.
    print(
        "[4/15] HNSW search (recompute baseline - FAISS stored search)...",
        file=sys.stderr,
    )
    for ef in [16, 32, 64, 128, 256]:
        index.hnsw.efSearch = ef
        times = bench_fn(
            lambda: index.search(query, top_k), warmup=200, repeats=5000
        )
        results[f"hnsw_search_recompute/ef{ef}"] = make_result(times)

    # ── Full pipeline (build + write + read + search) ───────────────
    print("[5/15] Full pipeline...", file=sys.stderr)
    import tempfile
    import os

    for n in [100, 1_000, 10_000]:
        print(f"  Pipeline {n} vectors...", file=sys.stderr)
        data = rng.rand(n, dim).astype(np.float32)
        query = rng.rand(1, dim).astype(np.float32)

        def pipeline(data=data, query=query):
            idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
            idx.hnsw.efConstruction = 200
            idx.add(data)

            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
                fname = f.name
            faiss.write_index(idx, fname)
            loaded = faiss.read_index(fname)
            loaded.hnsw.efSearch = 64
            loaded.search(query, top_k)
            os.unlink(fname)

        repeats = 5 if n >= 10_000 else 10
        times = bench_fn(pipeline, warmup=1, repeats=repeats)
        results[f"full_pipeline/{n}"] = make_result(times)

        # Index size
        idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
        idx.hnsw.efConstruction = 200
        idx.add(data)
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            fname = f.name
        faiss.write_index(idx, fname)
        size = os.path.getsize(fname)
        os.unlink(fname)
        results[f"index_size_bytes/{n}"] = make_size_result(size)

    # ── Passage lookup ────────────────────────────────────────────────
    print("[6/15] Passage lookup...", file=sys.stderr)
    import pickle

    for n in [1_000, 10_000]:
        print(f"  Passage lookup n={n}...", file=sys.stderr)

        with tempfile.TemporaryDirectory() as tmpdir:
            passages_path = os.path.join(tmpdir, "test.passages.jsonl")
            offset_path = os.path.join(tmpdir, "test.passages.idx")

            # Create passages with realistic-ish text
            offset_map = {}
            with open(passages_path, "w", encoding="utf-8") as f:
                for i in range(n):
                    offset = f.tell()
                    passage = {
                        "id": str(i),
                        "text": (
                            f"This is document number {i} about topic {i % 10}. "
                            "It contains some text that is representative of a "
                            "typical passage in a RAG system."
                        ),
                        "metadata": {"doc_num": i, "topic": f"topic_{i % 10}"},
                    }
                    json.dump(passage, f, ensure_ascii=False)
                    f.write("\n")
                    offset_map[str(i)] = offset

            with open(offset_path, "wb") as f:
                pickle.dump(offset_map, f)

            # Load offset map (as Python does)
            with open(offset_path, "rb") as f:
                loaded_offsets = pickle.load(f)

            # Generate random lookup indices
            lookup_rng = np.random.RandomState(42)
            lookup_ids = [str(x) for x in lookup_rng.randint(0, n, size=1000)]

            def do_lookups():
                with open(passages_path, encoding="utf-8") as pf:
                    for pid in lookup_ids:
                        offset = loaded_offsets[pid]
                        pf.seek(offset)
                        json.loads(pf.readline())

            times = bench_fn(do_lookups, warmup=5, repeats=100)
            results[f"passage_lookup/{n}"] = make_result(times, scale=1.0 / len(lookup_ids))

    # ── Text chunking ──────────────────────────────────────────────
    print("[7/15] Text chunking...", file=sys.stderr)

    BASE_SENTENCES = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning algorithms can identify patterns in large datasets.",
        "Vector databases enable similarity search across high-dimensional spaces.",
        "Rust provides memory safety without garbage collection.",
        "Neural networks are inspired by the structure of the human brain.",
        "Cloud computing offers scalable infrastructure for modern applications.",
        "Natural language processing has advanced significantly in recent years.",
        "Graph databases model relationships between entities using nodes and edges.",
        "Functional programming emphasizes immutability and pure functions.",
        "The human genome contains approximately three billion base pairs.",
        "Database systems store and retrieve data efficiently using indexing.",
        "Python is a versatile language used for web development and data science.",
        "JavaScript runs in web browsers and is essential for front-end development.",
        "Deep learning has revolutionized computer vision and speech recognition.",
        "Information retrieval systems rank documents by relevance to a query.",
        "Distributed systems coordinate multiple computers to achieve a common goal.",
        "Cryptographic hash functions map arbitrary data to fixed-size outputs.",
        "Operating systems manage hardware resources and provide services to applications.",
        "Compilers translate high-level programming languages into machine code.",
        "Parallel computing divides a problem into subproblems solved simultaneously.",
    ]

    import re

    def py_split_sentences(text):
        """Port of Rust's split_sentences heuristic."""
        if not text:
            return []
        sentences = []
        current = []
        chars = list(text)
        length = len(chars)
        i = 0
        while i < length:
            ch = chars[i]
            current.append(ch)
            if ch in '.!?' and current:
                is_boundary = True
                if ch == '.':
                    # Not boundary if followed by digit
                    if i + 1 < length and chars[i + 1].isdigit():
                        is_boundary = False
                    # Not boundary if preceded by single uppercase letter
                    elif (i >= 1 and chars[i - 1].isupper()
                          and (i < 2 or not chars[i - 2].isalnum())):
                        is_boundary = False
                    elif i + 1 >= length:
                        is_boundary = True
                    elif i + 2 < length and chars[i + 1].isspace() and chars[i + 2].isupper():
                        is_boundary = True
                    elif chars[i + 1] if i + 1 < length else '' == '\n':
                        is_boundary = True
                    elif i + 1 < length and chars[i + 1].isspace():
                        is_boundary = True
                    else:
                        is_boundary = False
                if is_boundary:
                    s = ''.join(current).strip()
                    if s:
                        sentences.append(s)
                    current = []
            i += 1
        s = ''.join(current).strip()
        if s:
            sentences.append(s)
        return sentences

    def py_chunk_text(text, chunk_size, chunk_overlap):
        """Port of Rust's chunk_text algorithm."""
        sentences = py_split_sentences(text)
        if not sentences:
            return []
        chunks = []
        current_chunk = ""
        current_len = 0
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:]
                    current_len = len(current_chunk)
                else:
                    current_chunk = ""
                    current_len = 0
            if sent_len > chunk_size and not current_chunk:
                offset = 0
                while offset < sent_len:
                    end = min(offset + chunk_size, sent_len)
                    chunks.append(sent[offset:end].strip())
                    offset = end
                continue
            if current_chunk:
                current_chunk += " "
                current_len += 1
            current_chunk += sent
            current_len += sent_len
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def gen_text(target_bytes):
        parts = []
        total = 0
        i = 0
        while total < target_bytes:
            s = BASE_SENTENCES[i % len(BASE_SENTENCES)]
            parts.append(s)
            total += len(s) + 1
            i += 1
        return " ".join(parts)

    for label, target_bytes in [("10KB", 10_000), ("100KB", 100_000), ("1MB", 1_000_000)]:
        print(f"  text_chunking/{label}...", file=sys.stderr)
        text = gen_text(target_bytes)
        times = bench_fn(lambda t=text: py_chunk_text(t, 512, 50), warmup=3, repeats=50)
        results[f"text_chunking/{label}"] = make_result(times)

    # ── Sentence splitting ───────────────────────────────────────────
    print("[8/15] Sentence splitting...", file=sys.stderr)
    for label, target_bytes in [("10KB", 10_000), ("100KB", 100_000), ("1MB", 1_000_000)]:
        print(f"  sentence_split/{label}...", file=sys.stderr)
        text = gen_text(target_bytes)
        times = bench_fn(lambda t=text: py_split_sentences(t), warmup=3, repeats=50)
        results[f"sentence_split/{label}"] = make_result(times)

    # ── BM25 index build ─────────────────────────────────────────────
    print("[9/15] BM25 fit...", file=sys.stderr)

    TOPICS = [
        "machine learning", "database systems", "web development",
        "cloud computing", "neural networks", "programming languages",
        "operating systems", "computer vision", "natural language",
        "distributed systems",
    ]

    class PyBM25Scorer:
        """Port of Rust BM25Scorer for apples-to-apples comparison."""

        def __init__(self, k1=1.2, b=0.75):
            self.k1 = k1
            self.b = b
            self.doc_freqs = {}
            self.doc_lengths = {}
            self.word_counts = {}
            self.avg_doc_length = 0.0
            self.corpus_size = 0
            self.id_set = set()
            self._tokenizer_re = re.compile(r'[^\w\s]')

        def _tokenize(self, text):
            cleaned = self._tokenizer_re.sub('', text)
            return cleaned.lower().split()

        def fit(self, documents):
            """documents: list of (id, text) tuples."""
            self.corpus_size = len(documents)
            self.doc_lengths = {}
            self.word_counts = {}
            self.id_set = set()
            doc_freqs = {}
            total_length = 0
            for doc_id, text in documents:
                words = self._tokenize(text)
                self.doc_lengths[doc_id] = len(words)
                total_length += len(words)
                unique_words = set(words)
                for w in unique_words:
                    doc_freqs[w] = doc_freqs.get(w, 0) + 1
                counts = {}
                for w in words:
                    counts[w] = counts.get(w, 0) + 1
                self.word_counts[doc_id] = counts
                self.id_set.add(doc_id)
            self.doc_freqs = doc_freqs
            self.avg_doc_length = total_length / self.corpus_size if self.corpus_size else 0.0

        def _score(self, query_words, doc_id):
            import math as m
            pw = self.word_counts.get(doc_id)
            if pw is None:
                return 0.0
            pl = sum(pw.values())
            score = 0.0
            for word in query_words:
                df = self.doc_freqs.get(word)
                if df is None:
                    continue
                wf = pw.get(word, 0)
                idf = m.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
                tf = (wf * (self.k1 + 1.0)) / (
                    wf + self.k1 * (1.0 - self.b + self.b * (pl / self.avg_doc_length))
                )
                score += idf * tf
            return score

        def search(self, query, top_k):
            query_words = self._tokenize(query)
            scores = [(doc_id, self._score(query_words, doc_id)) for doc_id in self.id_set]
            scores.sort(key=lambda x: -x[1])
            return scores[:top_k]

    for n_docs in [1_000, 10_000]:
        print(f"  bm25_fit/{n_docs}...", file=sys.stderr)
        docs = [
            (
                str(i),
                f"Document {i} about {TOPICS[i % len(TOPICS)]}. The quick brown fox jumps over the lazy dog. "
                f"This document discusses various aspects of {TOPICS[i % len(TOPICS)]} including theory and practice."
            )
            for i in range(n_docs)
        ]
        repeats = 10 if n_docs >= 10_000 else 30

        def bm25_fit_fn(d=docs):
            s = PyBM25Scorer()
            s.fit(d)

        times = bench_fn(bm25_fit_fn, warmup=2, repeats=repeats)
        results[f"bm25_fit/{n_docs}"] = make_result(times)

    # ── BM25 search ──────────────────────────────────────────────────
    print("[10/15] BM25 search...", file=sys.stderr)
    for n_docs in [1_000, 10_000]:
        print(f"  bm25_search/{n_docs}...", file=sys.stderr)
        docs = [
            (
                str(i),
                f"Document {i} about {TOPICS[i % len(TOPICS)]}. The quick brown fox jumps over the lazy dog. "
                f"This document discusses various aspects of {TOPICS[i % len(TOPICS)]} including theory and practice."
            )
            for i in range(n_docs)
        ]
        scorer = PyBM25Scorer()
        scorer.fit(docs)
        times = bench_fn(
            lambda s=scorer: s.search("machine learning neural networks", 10),
            warmup=50, repeats=500,
        )
        results[f"bm25_search/{n_docs}"] = make_result(times)

    # ── Metadata filtering ───────────────────────────────────────────
    print("[11/15] Metadata filtering...", file=sys.stderr)

    GENRES = ["fiction", "science", "history", "biography", "drama"]
    META_TOPICS = [f"topic_{i}" for i in range(10)]

    class PyMetadataFilterEngine:
        """Port of Rust MetadataFilterEngine for apples-to-apples comparison."""

        def apply_filters(self, results_list, filters):
            if not filters:
                return list(results_list)
            return [r for r in results_list if self._eval(r, filters)]

        def _eval(self, result, filters):
            for field, spec in filters.items():
                val = result.get("metadata", {}).get(field)
                if val is None:
                    val = result.get(field)
                if val is None:
                    return False
                for op, expected in spec.items():
                    if not self._op(op, val, expected):
                        return False
            return True

        def _op(self, op, val, expected):
            if op == "==":
                return val == expected
            elif op == "!=":
                return val != expected
            elif op == "<":
                return val < expected
            elif op == "<=":
                return val <= expected
            elif op == ">":
                return val > expected
            elif op == ">=":
                return val >= expected
            elif op == "in":
                return val in expected
            elif op == "not_in":
                return val not in expected
            elif op == "contains":
                return expected in str(val)
            elif op == "starts_with":
                return str(val).startswith(expected)
            elif op == "ends_with":
                return str(val).endswith(expected)
            elif op == "is_true":
                return bool(val)
            elif op == "is_false":
                return not bool(val)
            return False

    for n_res in [1_000, 10_000]:
        print(f"  metadata_filter/{n_res}...", file=sys.stderr)
        result_set = [
            {
                "id": f"doc{i}",
                "score": 0.95 - i * 0.0001,
                "text": f"Text for document {i}",
                "metadata": {
                    "chapter": (i % 20) + 1,
                    "genre": GENRES[i % len(GENRES)],
                    "topic": META_TOPICS[i % len(META_TOPICS)],
                    "word_count": 500 + (i * 37) % 5000,
                    "is_published": i % 3 != 0,
                },
            }
            for i in range(n_res)
        ]
        meta_filters = {
            "chapter": {"<=": 5},
            "genre": {"in": ["fiction", "science"]},
        }
        engine = PyMetadataFilterEngine()
        times = bench_fn(
            lambda rs=result_set, f=meta_filters: engine.apply_filters(rs, f),
            warmup=10, repeats=200,
        )
        results[f"metadata_filter/{n_res}"] = make_result(times)

    # ── Index I/O (write) ────────────────────────────────────────────
    print("[12/15] Index I/O (write)...", file=sys.stderr)
    for n_io in [1_000, 10_000]:
        print(f"  index_io_write/{n_io}...", file=sys.stderr)
        io_data = rng.rand(n_io, 384).astype(np.float32)
        idx = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_L2)
        idx.hnsw.efConstruction = 200
        idx.add(io_data)

        def write_idx(idx=idx):
            with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
                fname = f.name
            faiss.write_index(idx, fname)
            os.unlink(fname)

        times = bench_fn(write_idx, warmup=5, repeats=50)
        results[f"index_io_write/{n_io}"] = make_result(times)

    # ── Index I/O (read) ─────────────────────────────────────────────
    print("[13/15] Index I/O (read)...", file=sys.stderr)
    for n_io in [1_000, 10_000]:
        print(f"  index_io_read/{n_io}...", file=sys.stderr)
        io_data = rng.rand(n_io, 384).astype(np.float32)
        idx = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_L2)
        idx.hnsw.efConstruction = 200
        idx.add(io_data)
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            idx_fname = f.name
        faiss.write_index(idx, idx_fname)

        times = bench_fn(lambda fn=idx_fname: faiss.read_index(fn), warmup=5, repeats=50)
        results[f"index_io_read/{n_io}"] = make_result(times)
        os.unlink(idx_fname)

    # ── Passage file write ───────────────────────────────────────────
    print("[14/15] Passage write...", file=sys.stderr)
    for n_pw in [1_000, 10_000]:
        print(f"  passage_write/{n_pw}...", file=sys.stderr)
        pw_passages = [
            {
                "id": str(i),
                "text": (
                    f"This is document number {i} about topic {i % 10}. "
                    "It contains some text that is representative of a "
                    "typical passage in a RAG system."
                ),
                "metadata": {"doc_num": i, "topic": f"topic_{i % 10}"},
            }
            for i in range(n_pw)
        ]

        def write_passages_py(passages=pw_passages):
            with tempfile.TemporaryDirectory() as td:
                pp = os.path.join(td, "bench.passages.jsonl")
                op = os.path.join(td, "bench.passages.idx")
                offsets = {}
                with open(pp, "w", encoding="utf-8") as f:
                    for p in passages:
                        offsets[p["id"]] = f.tell()
                        json.dump(p, f, ensure_ascii=False)
                        f.write("\n")
                with open(op, "wb") as f:
                    pickle.dump(offsets, f)

        times = bench_fn(write_passages_py, warmup=2, repeats=30)
        results[f"passage_write/{n_pw}"] = make_result(times)

    # ── CLI startup time ─────────────────────────────────────────────
    print("[15/15] CLI startup time...", file=sys.stderr)
    import subprocess
    import shutil

    # Python CLI: use `uv run leann --help`
    uv_bin = shutil.which("uv")
    if uv_bin:
        print("  Measuring Python CLI startup (uv run leann --help)...", file=sys.stderr)
        # Warmup
        for _ in range(3):
            subprocess.run(
                [uv_bin, "run", "leann", "--help"],
                capture_output=True,
            )
        cli_times = []
        for _ in range(20):
            start = time.perf_counter()
            subprocess.run(
                [uv_bin, "run", "leann", "--help"],
                capture_output=True,
            )
            cli_times.append(time.perf_counter() - start)
        cli_times.sort()
        results["cli_startup"] = make_result(cli_times)
    else:
        print("  SKIP: uv not found, cannot measure CLI startup.", file=sys.stderr)

    print("\nDone!", file=sys.stderr)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Pretty-print to stderr for human consumption
        print("\n--- Results ---", file=sys.stderr)
        for key, val in results.items():
            if "index_size" in key:
                print(f"  {key}: {val['median_s']:.0f} bytes", file=sys.stderr)
            elif val["median_ms"] > 1000:
                print(f"  {key}: {val['median_s']:.3f} s", file=sys.stderr)
            elif val["median_ms"] > 1:
                print(f"  {key}: {val['median_ms']:.3f} ms", file=sys.stderr)
            else:
                print(f"  {key}: {val['median_us']:.1f} us", file=sys.stderr)


if __name__ == "__main__":
    main()
