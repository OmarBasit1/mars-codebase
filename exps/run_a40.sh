#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/export2/obasit/MARS_derivative/vllm/.venv/bin/python}
WL=${WL:-/export2/obasit/MARS_derivative/mars-codebase/diverse_oneapi_merged_exp_uniform.json}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
WINDOW=${WINDOW:-300}
QPS_LIST=${QPS_LIST:-"5 6 7"}    # qps=9 exceeds A40 capacity for MARS (cuBLAS OOM at kv_usage≈1.0)
OUT=${OUT:-./results/single_api_a40_qwen2.5_14b}
GPU=${GPU:-0}
CPU_GB=${CPU_GB:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM=${GPU_MEM:-0.9}  # 0.88 leaves cuBLAS workspace headroom at high KV usage
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
  # # MARS: V + V2 queue + chunk-fill (token budget 1024) + CPU-offload swap + starvation.
  # run MARS_async "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 1024 --swap \
  #     --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000
  # # MARS ablation: full MARS but demotion disabled (isolate the demotion effect).
  # run MARS_no_demote_async "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 1024 --swap \
  #     --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000 --no-demote
  # # Vanilla vLLM baseline: discard (recompute) + FCFS.
  # run Discard_async "$q" --api-policy D --policy-config fcfs
  # # Always preserve
  # run Preserve_async "$q" --api-policy P --policy-config fcfs
  # # Always swap
  # run Swap_async "$q" --api-policy S --policy-config fcfs --swap

  # MARS: V + V2 queue + chunk-fill (token budget 1024) + CPU-offload swap + starvation.
  run MARS_sync "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 1024 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000 --sync-scheduling
  # MARS ablation: full MARS but demotion disabled (isolate the demotion effect).
  run MARS_no_demote_sync "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 1024 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000 --sync-scheduling --no-demote
  # Vanilla vLLM baseline: discard (recompute) + FCFS.
  run Discard_sync "$q" --api-policy D --policy-config fcfs --sync-scheduling
  # Always preserve
  run Preserve_sync "$q" --api-policy P --policy-config fcfs --sync-scheduling
  # Always swap
  run Swap_sync "$q" --api-policy S --policy-config fcfs --swap --sync-scheduling
done

echo "per-config CSVs written to $OUT/"
