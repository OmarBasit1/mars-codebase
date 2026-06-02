#!/usr/bin/env bash
# Port of mars-codebase/exps/6B_bench.sh to the vLLM-v1 MARS harness.
#
# Sweeps the MARS policies through mars.bench.run and writes a per-request CSV
# per policy. Override via env: MODEL, WL (workload json), N (num requests),
# QPS, OUT (output dir), GPU, POLICIES, SWAP_POLICIES.
#
# For meaningful policy differentiation (the "V beats P and D" result) use a
# REAL served model (memory pressure), a workload with substantial api_time, and
# enough load (N, QPS) to pressure GPU memory -- e.g.:
#   MODEL=<6B-model> N=2000 QPS=4 WL=<workload> bash run_experiments.sh
set -euo pipefail

PY=${PY:-/export2/obasit/MARS_derivative/vllm/.venv/bin/python}
WL=${WL:-/export2/obasit/MARS_derivative/mars-codebase/diverse_oneapi_merged_exp_uniform.json}
MODEL=${MODEL:-facebook/opt-125m}
N=${N:-32}
QPS=${QPS:-8}
OUT=${OUT:-/tmp/mars_exp}
GPU=${GPU:-0}
# swap-free policies, then swap policies (which add --swap for the connector).
POLICIES=${POLICIES:-"P D V H-D I"}
SWAP_POLICIES=${SWAP_POLICIES:-"S V"}

mkdir -p "$OUT"
# Run from a neutral dir so the old vendored vllm/ in mars-codebase can't shadow
# the installed vLLM (mars.bench.run also strips cwd from sys.path defensively).
cd "$OUT"

run() {  # api_policy [extra args...]
  local pol="$1"; shift
  local tag="$1"; shift
  echo "### policy=$pol ${*:-} ###"
  CUDA_VISIBLE_DEVICES="$GPU" VLLM_LOGGING_LEVEL=ERROR "$PY" -m mars.bench.run \
    --workload "$WL" --model "$MODEL" --num-requests "$N" --qps "$QPS" \
    --api-policy "$pol" --csv "$OUT/${tag}.csv" "$@" 2>/dev/null
  echo
}

for pol in $POLICIES; do
  run "$pol" "$pol"
done
for pol in $SWAP_POLICIES; do
  run "$pol" "${pol}_swap" --swap
done

echo "per-request CSVs written to $OUT/"
