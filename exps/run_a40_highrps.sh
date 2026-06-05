#!/usr/bin/env bash
set -euo pipefail
# Trimmed runner for the high-RPS functionality check: MARS(V), MARS-no-demote, Swap only.
PY=${PY:-/export2/obasit/MARS_derivative/vllm/.venv/bin/python}
WL=${WL:-/export2/obasit/MARS_derivative/mars-codebase/diverse_oneapi_merged_exp_uniform.json}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
WINDOW=${WINDOW:-100}
QPS_LIST=${QPS_LIST:-"9 11 13"}
OUT=${OUT:-./results/sweep_qps3to7_a40_qwen2.5_14b}
GPU=${GPU:-0}
CPU_GB=${CPU_GB:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM=${GPU_MEM:-0.9}
SEED=${SEED:-42}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

mkdir -p "$OUT"; cd "$OUT"

run() {
  local tag="$1"; shift; local qps="$1"; shift
  echo "### $tag qps=$qps ###"
  CUDA_VISIBLE_DEVICES="$GPU" VLLM_LOGGING_LEVEL=INFO "$PY" -m mars.bench.run \
    --workload "$WL" --model "$MODEL" --window "$WINDOW" --qps "$qps" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-mem "$GPU_MEM" --cpu-gb "$CPU_GB" \
    --seed "$SEED" --max-num-seqs "$MAX_NUM_SEQS" \
    "$@" --csv "${tag}_${qps}.csv" &> "${tag}_${qps}.log"
  echo
}

for q in $QPS_LIST; do
  run MARS_sync "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 1024 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000 --sync-scheduling
  run MARS_no_demote_sync "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 1024 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000 --sync-scheduling --no-demote
  run Swap_sync "$q" --api-policy S --policy-config fcfs --swap --sync-scheduling
done
echo "high-rps CSVs written to $OUT/"
