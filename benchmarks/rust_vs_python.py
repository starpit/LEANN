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
    print("\n[1/6] Distance computation...", file=sys.stderr)
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
    print("[2/6] HNSW build...", file=sys.stderr)
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
    print("[3/6] HNSW search (stored vectors)...", file=sys.stderr)
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
        "[4/6] HNSW search (recompute baseline - FAISS stored search)...",
        file=sys.stderr,
    )
    for ef in [16, 32, 64, 128, 256]:
        index.hnsw.efSearch = ef
        times = bench_fn(
            lambda: index.search(query, top_k), warmup=200, repeats=5000
        )
        results[f"hnsw_search_recompute/ef{ef}"] = make_result(times)

    # ── Full pipeline (build + write + read + search) ───────────────
    print("[5/6] Full pipeline...", file=sys.stderr)
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
    print("[6/6] Passage lookup...", file=sys.stderr)
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
