#!/usr/bin/env bash
set -euo pipefail
# Max-RPS probe (GPU 0): same workload/configs as run_a40_0.sh but with the simulated
# API wait set to 0 (--zero-api-time). The pause/resume + KV-policy machinery still
# fires every API boundary; only the idle wall-clock is removed. Sweep qps to find
# the system's max sustainable RPS (the knee) and check whether MARS and the
# baselines converge there or a drain gap persists.
#
# Default: async scheduling + CUDA graphs (matches plain vLLM defaults).
# To reproduce the old eager+sync config: EXTRA="--enforce-eager --sync-scheduling"

PY=${PY:-/export2/obasit/MARS_derivative/vllm/.venv/bin/python}
WL=${WL:-/export2/obasit/MARS_derivative/mars-codebase/diverse_oneapi_converted.json}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
WINDOW=${WINDOW:-200}
QPS_LIST=${QPS_LIST:-"3 7 9"}   # start where the realistic load capped (~9-11) and climb
OUT=${OUT:-./results/maxrps_a40_qwen2.5_14b_200s_async_graphs}
GPU=${GPU:-0}
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
    --seed "$SEED" --max-num-seqs "$MAX_NUM_SEQS" --zero-api-time \
    $LOAD $EXTRA "$@" --csv "${tag}_${qps}.csv" &> "$log"
  echo
}

for q in $QPS_LIST; do
  # True vanilla vLLM baseline: stock scheduler, PRESERVE-only.
  run Vanilla "$q" --stock-scheduler

  # MARS (headline, no chunk-fill): V + V2 + swap + starvation.
  run MARS "$q" --api-policy V --policy-config V2 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000

  # MARS with chunk-fill 2048.
  run MARS_chunk2048 "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 2048 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000

  # MARS eager-drop ablation.
  run MARS_eager "$q" --api-policy V --policy-config V2 --chunk-fill --chunk-size 2048 --swap \
      --starvation-avoidance --starvation-threshold 100 --starvation-quantum 100000 --demote-eager

  # Simple baselines.
  run Discard "$q" --api-policy D --policy-config fcfs
  run Swap "$q" --api-policy S --policy-config fcfs --swap
done

echo "per-config CSVs written to $OUT/"
