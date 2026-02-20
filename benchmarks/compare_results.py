#!/usr/bin/env python3
"""
Compare Rust and Python benchmark results and produce a markdown table.

Reads results/rust_results.json and results/python_results.json,
outputs a formatted markdown comparison table to stdout.

Usage:
    uv run python benchmarks/compare_results.py
"""

import json
import sys
from pathlib import Path


def format_time(val):
    """Format a time value with appropriate units."""
    if val >= 1.0:
        return f"{val:.3f} s"
    elif val >= 1e-3:
        return f"{val * 1e3:.3f} ms"
    elif val >= 1e-6:
        return f"{val * 1e6:.1f} us"
    else:
        return f"{val * 1e9:.1f} ns"


def format_size(val):
    """Format a byte size."""
    if val >= 1e6:
        return f"{val / 1e6:.1f} MB"
    elif val >= 1e3:
        return f"{val / 1e3:.1f} KB"
    else:
        return f"{val:.0f} B"


def format_range(r):
    """Format a p5-p95 range compactly."""
    if r is None:
        return ""
    lo, hi = r
    return f"[{format_time(lo)}–{format_time(hi)}]"


def classify(r, p):
    """Classify a benchmark comparison as rust_better, python_better, or tie.

    Uses IQR (p25-p75) overlap: if the interquartile ranges don't overlap,
    the faster one is the clear winner. Otherwise it's a statistical tie.
    Returns (category, speedup_ratio) where speedup = python_p50 / rust_p50.
    """
    r_val = r.get("median_s") if r else None
    p_val = p.get("median_s") if p else None
    if not r_val or not p_val or r_val <= 0:
        return "unknown", None

    speedup = p_val / r_val

    # Check IQR overlap (lower time = faster)
    r_p25 = r.get("p25_s", r_val)
    r_p75 = r.get("p75_s", r_val)
    p_p25 = p.get("p25_s", p_val)
    p_p75 = p.get("p75_s", p_val)

    if r_p75 < p_p25:
        return "rust_better", speedup
    elif p_p75 < r_p25:
        return "python_better", speedup
    else:
        return "tie", speedup


def format_verdict(category, speedup):
    """Format the verdict column."""
    if category == "unknown" or speedup is None:
        return "N/A"
    if category == "rust_better":
        return f"**Rust {speedup:.1f}x**"
    elif category == "python_better":
        return f"Python {1/speedup:.1f}x"
    else:
        return f"~Tie ({speedup:.2f}x)"


def main():
    results_dir = Path(__file__).parent / "results"

    rust_path = results_dir / "rust_results.json"
    python_path = results_dir / "python_results.json"

    if not rust_path.exists():
        print(f"ERROR: {rust_path} not found. Run Rust benchmarks first.", file=sys.stderr)
        sys.exit(1)
    if not python_path.exists():
        print(f"ERROR: {python_path} not found. Run Python benchmarks first.", file=sys.stderr)
        sys.exit(1)

    with open(rust_path) as f:
        rust = json.load(f)
    with open(python_path) as f:
        python = json.load(f)

    # Collect all keys
    all_keys = sorted(set(list(rust.keys()) + list(python.keys())))

    # Group by category
    categories = {}
    for key in all_keys:
        category = key.split("/")[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(key)

    print("# Rust vs Python (FAISS C++) HNSW Benchmark Comparison\n")
    print(f"*Generated from benchmark results*\n")
    print("Verdict uses IQR (p25–p75) overlap: non-overlapping = clear winner, overlapping = ~Tie.\n")

    for category, keys in categories.items():
        is_size = "index_size" in category
        cat_title = category.replace("_", " ").title()
        print(f"\n## {cat_title}\n")

        if is_size:
            print("| Metric | Rust | Python (FAISS) | Ratio |")
            print("|--------|------|----------------|-------|")
        else:
            print("| Metric | Rust p50 | p5–p95 | Python p50 | p5–p95 | Verdict |")
            print("|--------|----------|--------|------------|--------|---------|")

        for key in keys:
            r = rust.get(key)
            p = python.get(key)

            label = "/".join(key.split("/")[1:])

            if is_size:
                r_val = r["median_s"] if r else None
                p_val = p["median_s"] if p else None
                r_str = format_size(r_val) if r_val is not None else "N/A"
                p_str = format_size(p_val) if p_val is not None else "N/A"
                if r_val and p_val and p_val > 0:
                    ratio = r_val / p_val
                    ratio_str = f"{ratio:.2f}x"
                else:
                    ratio_str = "N/A"
                print(f"| {label} | {r_str} | {p_str} | {ratio_str} |")
            else:
                r_val = r["median_s"] if r else None
                p_val = p["median_s"] if p else None
                r_str = format_time(r_val) if r_val is not None else "N/A"
                p_str = format_time(p_val) if p_val is not None else "N/A"

                # p5-p95 ranges
                if r and "p5_s" in r:
                    r_range = format_range((r["p5_s"], r["p95_s"]))
                else:
                    r_range = ""
                if p and "p5_s" in p:
                    p_range = format_range((p["p5_s"], p["p95_s"]))
                else:
                    p_range = ""

                cat, spd = classify(r, p)
                verdict = format_verdict(cat, spd)
                print(f"| {label} | {r_str} | {r_range} | {p_str} | {p_range} | {verdict} |")

    # Summary statistics
    print("\n## Summary\n")
    rust_wins = []
    python_wins = []
    ties = []
    all_speedups = []

    for key in all_keys:
        if "index_size" in key:
            continue
        r = rust.get(key)
        p = python.get(key)
        if not r or not p or not r.get("median_s") or r["median_s"] <= 0:
            continue

        cat, spd = classify(r, p)
        if spd is not None:
            all_speedups.append((key, spd))
        if cat == "rust_better":
            rust_wins.append((key, spd))
        elif cat == "python_better":
            python_wins.append((key, spd))
        elif cat == "tie":
            ties.append((key, spd))

    total = len(rust_wins) + len(python_wins) + len(ties)
    if all_speedups:
        all_s = [s for _, s in all_speedups]
        geo_mean = 1.0
        for s in all_s:
            geo_mean *= s
        geo_mean = geo_mean ** (1.0 / len(all_s))

        print(f"- **Geometric mean speedup**: {geo_mean:.2f}x")
        print(f"- Rust better: {len(rust_wins)}/{total}")
        print(f"- Python better: {len(python_wins)}/{total}")
        print(f"- Statistical tie: {len(ties)}/{total}")
        if rust_wins:
            best = max(rust_wins, key=lambda x: x[1])
            print(f"- Best Rust speedup: {best[0]} ({best[1]:.1f}x)")
        if python_wins:
            worst = min(python_wins, key=lambda x: x[1])
            print(f"- Best Python speedup: {worst[0]} ({1/worst[1]:.1f}x)")


if __name__ == "__main__":
    main()
