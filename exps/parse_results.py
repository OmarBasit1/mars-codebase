#!/usr/bin/env python3
"""Parse per-request CSVs from run_a40.sh and emit a summary CSV.

Usage:
    python parse_results.py <results_dir> [output.csv]

Filename convention: {policy}_{qps}.csv
Metrics: mean/P99 E2E latency (e2e_s) and TTFT (ttft_s).
"""

import sys
import csv
import os
import re
import statistics
from pathlib import Path


def p99(values):
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * 0.99) - 1)
    return sorted_vals[idx]


def parse_dir(results_dir: Path):
    pattern = re.compile(r"^(.+)_(\d+(?:\.\d+)?)\.csv$")
    rows = []

    for path in sorted(results_dir.glob("*.csv")):
        m = pattern.match(path.name)
        if not m:
            print(f"  skip (unrecognised name): {path.name}", file=sys.stderr)
            continue

        policy, qps = m.group(1), m.group(2)

        e2e_vals, ttft_vals = [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    e2e_vals.append(float(row["e2e_s"]))
                    ttft_vals.append(float(row["ttft_s"]))
                except (KeyError, ValueError):
                    pass

        if not e2e_vals:
            print(f"  skip (no data): {path.name}", file=sys.stderr)
            continue

        rows.append(
            {
                "policy": policy,
                "qps": float(qps),
                "n_requests": len(e2e_vals),
                "mean_e2e_s": round(statistics.mean(e2e_vals), 4),
                "p99_e2e_s": round(p99(e2e_vals), 4),
                "mean_ttft_s": round(statistics.mean(ttft_vals), 4),
                "p99_ttft_s": round(p99(ttft_vals), 4),
            }
        )

    rows.sort(key=lambda r: (r["qps"], r["policy"]))
    return rows


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir> [output.csv]", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else results_dir / "summary.csv"

    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = parse_dir(results_dir)
    if not rows:
        print("No valid CSV files found.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["policy", "qps", "n_requests",
                  "mean_e2e_s", "p99_e2e_s", "mean_ttft_s", "p99_ttft_s"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {out_path}")

    # Pretty-print to stdout as well
    col_w = [max(len(h), max(len(str(r[h])) for r in rows)) for h in fieldnames]
    header = "  ".join(h.ljust(w) for h, w in zip(fieldnames, col_w))
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w) for h, w in zip(fieldnames, col_w)))


if __name__ == "__main__":
    main()
