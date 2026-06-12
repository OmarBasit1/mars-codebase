#!/usr/bin/env python3
"""Parse per-request CSVs from run_a40*.sh and emit a summary CSV.

Usage:
    python parse_results.py <results_dir> [output.csv]

Filename convention: {policy}_{qps}.csv  (qps = the workload / offered send rate).
Each CSV may have a sibling {policy}_{qps}.summary.json (written by mars.bench.run)
with run-level wall time, used for the throughput / output-qps metrics; runs
without it still parse (those rate columns render blank).

Per-run metrics (latency stats are over COMPLETED requests only):
  qps_workload         offered send rate (from filename)
  qps_output           completed / wall            (achieved request throughput)
  n_requests           offered requests
  request_completed    finished requests
  output_tokens_per_s  sum(gen_tokens, completed) / wall
  input_tokens_per_s   sum(input_tokens, completed) / wall
  mean/p99 e2e_s, ttft_s, last_req_ttft_s   (last = final post-API-resume TTFT)
  preserve_reqs / swap_reqs / recompute_reqs   EXECUTED KV handling
  demoted_reqs         preserved reqs that were freed then reloaded on API return
"""

import sys
import csv
import json
import math
import re
import statistics
from pathlib import Path


def to_float(x):
    """Parse a float, treating '', 'nan', None (and unparseable) as missing -> None."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def to_int(x, default=0):
    v = to_float(x)
    return default if v is None else int(v)


def mean(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def p99(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = max(0, int(len(vals) * 0.99) - 1)
    return vals[idx]


def rnd(x, n=4):
    """Round for output; render missing (None) as '' so the CSV stays clean."""
    return round(x, n) if x is not None else ""


def executed_category(row):
    """Effective KV handling of one request, from policy letter + counters.

    P->preserve, D->recompute, S->swap. For MARS 'V' the pause always preserves;
    KV is only freed via demotion and a swap-demotion is the only thing that sets
    swap_reloads -- so demoted -> (swap if swap_reloads>0 else recompute), else
    preserve. Falls back to the classified arrival_strategy when the policy letter
    is absent (older CSVs); 'unknown' if neither is present.
    """
    pol = (row.get("kv_policy") or "").strip()
    if pol == "P":
        return "preserve"
    if pol == "D":
        return "recompute"
    if pol == "S":
        return "swap"
    if pol == "V":
        if to_int(row.get("demotions")) > 0:
            return "swap" if to_int(row.get("swap_reloads")) > 0 else "recompute"
        return "preserve"
    strat = (row.get("arrival_strategy") or "").strip()
    return strat if strat in ("preserve", "swap", "recompute") else "unknown"


def parse_dir(results_dir: Path):
    pattern = re.compile(r"^(.+)_(\d+(?:\.\d+)?)\.csv$")
    rows_out = []

    for path in sorted(results_dir.glob("*.csv")):
        m = pattern.match(path.name)
        if not m:
            print(f"  skip (unrecognised name): {path.name}", file=sys.stderr)
            continue
        policy, qps = m.group(1), m.group(2)

        with open(path, newline="") as f:
            reqs = list(csv.DictReader(f))
        if not reqs:
            print(f"  skip (no data): {path.name}", file=sys.stderr)
            continue

        # Run-level sidecar (wall time); tolerate absence / corruption.
        wall = None
        sidecar = path.with_name(path.stem + ".summary.json")
        if sidecar.is_file():
            try:
                wall = to_float(json.load(open(sidecar)).get("wall_s"))
            except Exception as e:
                print(f"  warn (bad sidecar {sidecar.name}): {e}", file=sys.stderr)

        n_requests = len(reqs)
        completed = [r for r in reqs if str(r.get("finished")).strip() == "True"]
        n_completed = len(completed)

        # Latency distributions over COMPLETED requests only (unfinished e2e is the
        # timeout value, which would inflate the stats).
        e2e = [to_float(r.get("e2e_s")) for r in completed]
        ttft = [to_float(r.get("ttft_s")) for r in completed]
        last = [to_float(r.get("last_resume_ttft_s")) for r in completed]

        out_tok = sum(to_int(r.get("gen_tokens")) for r in completed)
        in_tok = sum(to_int(r.get("input_tokens")) for r in completed)

        # Executed KV categories + demotions, over ALL offered requests.
        cats = {"preserve": 0, "swap": 0, "recompute": 0, "unknown": 0}
        demoted = 0
        for r in reqs:
            cats[executed_category(r)] += 1
            if to_int(r.get("demotions")) > 0:
                demoted += 1

        def rate(total):
            return (total / wall) if (wall and wall > 0) else None

        rows_out.append({
            "policy": policy,
            "qps_workload": float(qps),
            "qps_output": rnd(rate(n_completed)),
            "n_requests": n_requests,
            "request_completed": n_completed,
            "output_tokens_per_s": rnd(rate(out_tok), 2),
            "input_tokens_per_s": rnd(rate(in_tok), 2),
            "mean_e2e_s": rnd(mean(e2e)),
            "p99_e2e_s": rnd(p99(e2e)),
            "mean_ttft_s": rnd(mean(ttft)),
            "p99_ttft_s": rnd(p99(ttft)),
            "mean_last_req_ttft_s": rnd(mean(last)),
            "p99_last_req_ttft_s": rnd(p99(last)),
            "preserve_reqs": cats["preserve"],
            "swap_reqs": cats["swap"],
            "recompute_reqs": cats["recompute"],
            "demoted_reqs": demoted,
        })

    rows_out.sort(key=lambda r: (r["qps_workload"], r["policy"]))
    return rows_out


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

    fieldnames = ["policy", "qps_workload", "qps_output", "n_requests",
                  "request_completed", "output_tokens_per_s", "input_tokens_per_s",
                  "mean_e2e_s", "p99_e2e_s", "mean_ttft_s", "p99_ttft_s",
                  "mean_last_req_ttft_s", "p99_last_req_ttft_s",
                  "preserve_reqs", "swap_reqs", "recompute_reqs", "demoted_reqs"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to {out_path}")

    # Pretty-print to stdout as well.
    col_w = [max(len(h), max(len(str(r[h])) for r in rows)) for h in fieldnames]
    header = "  ".join(h.ljust(w) for h, w in zip(fieldnames, col_w))
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w) for h, w in zip(fieldnames, col_w)))


if __name__ == "__main__":
    main()
