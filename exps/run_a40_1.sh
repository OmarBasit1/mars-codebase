#!/usr/bin/env bash
set -euo pipefail
# GPU 1 arm of the main MARS bench sweep.
# Default: async scheduling + CUDA graphs (matches plain vLLM defaults).
# To reproduce the old eager+sync config: EXTRA="--enforce-eager --sync-scheduling"

PY=${PY:-/export2/obasit/MARS_derivative/vllm/.venv/bin/python}
WL=${WL:-/export2/obasit/MARS_derivative/mars-codebase/diverse_oneapi_converted.json}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
WINDOW=${WINDOW:-600}
QPS_LIST=${QPS_LIST:-"5 9"}    # qps=9 near A40 capacity for MARS; monitor kv_usage
OUT=${OUT:-./results/single_api_a40_qwen2.5_14b_600s_async_graphs}
GPU=${GPU:-1}
CPU_GB=${CPU_GB:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM=${GPU_MEM:-0.9}  # fall back to 0.88 if cuBLAS OOM at high KV usage + graphs
DUMMY=${DUMMY:-0}
EXTRA=${EXTRA:-}
SEED=${SEED:-42}        # fixed seed => same workload + arrivals per qps (repeatable)
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}   # concurrent-seq cap (vLLM auto-default is ~128)

mkdir -p "$OUT"
cd "$OUT"  # neutral cwd (avoid any vllm/ shadowing)
LOAD=""
[ "$DUMMY" = "1" ] && LOAD="--load-format dummy"

run() {  # tag qps policy-flags...
  local tag="$1"; shift
  local qps="$1"; shift
  local log="${tag}_${qps}.log"
  echo "### $tag qps=$qps ###"
  CUDA_VISIBLE_DEVICES="$GPU" VLLM_LOGGING_LEVEL=INFO "$PY" -m mars.bench.run \
    --workload "$WL" --model "$MODEL" --window "$WINDOW" --qps "$qps" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-mem "$GPU_MEM" --cpu-gb "$CPU_GB" \
    --seed "$SEED" --max-num-seqs "$MAX_NUM_SEQS" \
    $LOAD $EXTRA "$@" --csv "${tag}_${qps}.csv" &> "$log"
  echo
}

for q in $QPS_LIST; do
  # True vanilla vLLM baseline: stock scheduler, PRESERVE-only (no MARS policy).
  run Vanilla "$q" --stock-scheduler

  # MARS (headline, no chunk-fill): V + V2 + swap + starvation. Demotion is
  # ON-DEMAND by default (lazy/minimal free-on-admission).
  run MARS "$q" --api-policy V --policy-config V2 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000

  # MARS with chunk-fill 2048 (isolates the TTFT cost of the token-budget cap).
  run MARS_chunk2048 "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 2048 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000

  # MARS eager-drop ablation: frees recompute/swap-classified KV AT THE PAUSE
  # (--demote-eager) instead of lazy on-demand demotion.
  run MARS_eager "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 2048 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000 --demote-eager

  # Simple baselines: discard (recompute), preserve, swap.
  run Discard "$q" --api-policy D --policy-config fcfs
  run Preserve "$q" --api-policy P --policy-config fcfs
  run Swap "$q" --api-policy S --policy-config fcfs --swap
done

echo "per-config CSVs written to $OUT/"
