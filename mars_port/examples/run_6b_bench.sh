#!/usr/bin/env bash
# Recreation of mars-codebase/exps/6B_bench.sh for the vLLM-v1 MARS port.
# A single call runs the full matrix (MARS / InferCept / vanilla) over a qps
# sweep via mars.bench.run, writing per-config metrics + a CSV each.
#
# Env overrides: MODEL, WL (workload json), WINDOW (s), QPS_LIST, OUT, GPU,
# CPU_GB, MAX_MODEL_LEN, GPU_MEM, DUMMY=1 (--load-format dummy, no download),
# EXTRA (extra flags appended to every run).
#
# Examples:
#   GPU=0 MODEL=Qwen/Qwen3-14B WINDOW=1800 bash run_6b_bench.sh   # full parity
#   GPU=0 MODEL=facebook/opt-125m WINDOW=20 QPS_LIST="2" bash run_6b_bench.sh  # smoke
set -euo pipefail

PY=${PY:-/export2/obasit/MARS_derivative/vllm/.venv/bin/python}
WL=${WL:-/export2/obasit/MARS_derivative/mars-codebase/diverse_oneapi_merged_exp_uniform.json}
MODEL=${MODEL:-Qwen/Qwen3-14B}
WINDOW=${WINDOW:-1800}
QPS_LIST=${QPS_LIST:-"3 4 5 6"}
OUT=${OUT:-/tmp/mars_6b}
GPU=${GPU:-0}
CPU_GB=${CPU_GB:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
GPU_MEM=${GPU_MEM:-0.9}
DUMMY=${DUMMY:-0}
EXTRA=${EXTRA:-}

mkdir -p "$OUT"
cd "$OUT"  # neutral cwd (avoid any vllm/ shadowing)
LOAD=""
[ "$DUMMY" = "1" ] && LOAD="--load-format dummy"

run() {  # tag qps policy-flags...
  local tag="$1"; shift
  local qps="$1"; shift
  echo "### $tag qps=$qps ###"
  CUDA_VISIBLE_DEVICES="$GPU" VLLM_LOGGING_LEVEL=ERROR "$PY" -m mars.bench.run \
    --workload "$WL" --model "$MODEL" --window "$WINDOW" --qps "$qps" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-mem "$GPU_MEM" --cpu-gb "$CPU_GB" \
    $LOAD $EXTRA "$@" --csv "$OUT/${tag}_${qps}.csv" 2>/dev/null
  echo
}

for q in $QPS_LIST; do
  # MARS: V + V2 queue + chunk-fill + CPU-offload swap + starvation + solver.
  run MARS "$q" --api-policy V --policy-config V2 --chunk-fill --swap --use-solver \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000
  # InferCept baseline: I + chunk-fill + swap.
  run InferCept "$q" --api-policy I --policy-config fcfs --chunk-fill --swap
  # Vanilla vLLM baseline: discard (recompute) + FCFS.
  run vanilla "$q" --api-policy D --policy-config fcfs
done

echo "per-config CSVs written to $OUT/"
