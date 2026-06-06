# MARS v1 Port

Port of **MARS: Fast Inference for Augmented LLMs** (NeurIPS 2025) from its original vendored vLLM 0.2.0 fork onto modern **vLLM v1**.

MARS = pause LLM generation at a predetermined token boundary → call an external API/tool → resume with the result injected into the context, under smart KV-cache scheduling policies that decide what to do with the KV cache during the wait.

---

## Repository layout

```
MARS_derivative/
├── vllm/                        # vLLM v1 repo (branch: mars-port)
│   └── .venv/                   # Python 3.12 venv (uv-managed, installed editable)
├── LMCache/                     # LMCache repo (branch: mars-port, swap connector)
└── mars-codebase/               # MARS research repo (branch: mars-v1-port)
    ├── mars_port/               # *** THE PORT — all new code lives here ***
    │   ├── mars/                # Extension package (pip-installed editable)
    │   │   ├── config.py        # MarsConfig — all tunable knobs
    │   │   ├── cost_model.py    # Waste functions: discard / preserve / swap
    │   │   ├── orchestrator.py  # ApiOrchestrator: pause→API→resume driver
    │   │   ├── params.py        # MarsApiParams (per-request extra_args)
    │   │   ├── policies.py      # Policy-mode dispatch (P/D/S direct; V adaptive)
    │   │   ├── prediction.py    # OraclePredictor (uses known/actual values)
    │   │   ├── swap.py          # cpu_offload_kv_transfer_config helper
    │   │   ├── v1/
    │   │   │   ├── scheduler.py # MARSScheduler (the core: KV policy, demotion, queues)
    │   │   │   ├── queue.py     # MARSRequestQueue (SJF) + V2RequestQueue (cost-based)

    │   │   ├── bench/
    │   │   │   ├── run.py       # Benchmark harness (python -m mars.bench.run)
    │   │   │   └── policy_cost.py  # Predicted per-policy waste printer
    │   │   └── entrypoints/
    │   │       └── api_server.py   # Thin API server skeleton
    │   ├── examples/
    │   │   ├── calibrate_cost.py      # GPU/model profiling → paste-ready MarsConfig
    │   │   ├── run_experiments.sh     # Full policy sweep (ports 6B_bench.sh)
    │   │   ├── test_orchestrator.py   # E2E: pause/inject/resume correctness
    │   │   ├── test_kv_policies.py    # E2E: P vs D byte-identical output
    │   │   ├── test_vulcan_classify.py# E2E: V strategy assigned at arrival
    │   │   ├── test_swap.py           # E2E: P/S/D byte-identical with CPU offload
    │   │   ├── test_demotion.py       # E2E: KV demotion under memory pressure
    │   │   ├── test_cost_model_2way.py   # Unit: preserve/recompute crossover (no GPU)
    │   │   ├── test_cost_model_3way.py   # Unit: 3-way cost model with swap (no GPU)
    │   │   ├── test_classify_v2.py       # Unit: classify at arrival + V2 ordering (no GPU)
    │   │   ├── test_queue_sjf.py         # Unit: SJF queue ordering (no GPU)
    │   │   ├── test_queue_combined.py    # Unit: combined V2/SJF across both queues (no GPU)
    │   │   └── test_queue_starvation.py  # Unit: starvation front-boost + ordering (no GPU)
    │   └── COMPARISON.md           # Algorithm-level original-vs-port diff
    ├── diverse_oneapi_merged_exp_uniform.json  # Standard workload file
    ├── benchmarks/fixed_final_tput_bench_real.py  # Original vLLM 0.2.0 benchmark
    └── exps/6B_bench.sh            # Original experiment sweep script
```

---

## Which vLLM to install

Use the **`mars-port` branch** of `https://github.com/OmarBasit1/vllm`.

This branch adds three small, backward-compatible changes to vLLM v1:
1. `_reclaim_blocks_for_admission` hook in the waiting-admission loop (default `return False`)
2. Relaxed assert in `_update_from_kv_xfer_finished` so a MARS-cancelled WFRKV transfer's stale completion is ignored
3. `cancel_load_for_request` on the KV-connector base + `SimpleCPUOffloadConnector` + `simple_kv_offload` manager

The `mars` extension package otherwise requires **zero vLLM core edits** — everything is injected via `scheduler_cls`, `extra_args`, native resumable streaming, and the KV-connector interface.

---

## Installation

### Prerequisites

- `uv` package manager

### 1. Check out the correct branches

```bash
cd MARS_derivative/vllm
git checkout mars-port

cd MARS_derivative/mars-codebase
git checkout mars-v1-port
```

### 2. Create the venv and install vLLM

```bash
cd MARS_derivative/vllm
uv venv .venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

> **Note:** Always use `VLLM_USE_PRECOMPILED=1` to avoid a slow full source compile.

### 3. Install the MARS extension package

```bash
MARS_derivative/vllm/.venv/bin/pip install -e \
    MARS_derivative/mars-codebase/mars_port
```

### 4. Verify the install

```bash
# vllm must resolve to the v1 repo, NOT the old vendored copy in mars-codebase
MARS_derivative/vllm/.venv/bin/python -c \
    "import vllm; print(vllm.__file__)"
# Expected: MARS_derivative/vllm/vllm/__init__.py

# MARS package
MARS_derivative/vllm/.venv/bin/python -c "import mars; print('ok')"
```


---

## GPU/model profiling (new hardware or new model)

Run `examples/calibrate_cost.py` to measure the real forward-step and KV-transfer latencies and get a paste-ready `MarsConfig` block:

```bash
CUDA_VISIBLE_DEVICES=0 \
  MARS_derivative/vllm/.venv/bin/python \
  MARS_derivative/mars-codebase/mars_port/examples/calibrate_cost.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --max-model-len 32768

# Use --load-format dummy for a quick cost-only calibration (no weights downloaded):
#   --load-format dummy
```

The script outputs:
- Per-token swap latency (PCIe D2H+H2D round-trip)
- Linear prefill fit (`cost_a`, `cost_c`) for the Vulcan waste model

Copy the printed cost values and paste it into the config (see next section).

---

## Updating GPU/model cost coefficients

All cost coefficients live in [mars_port/mars/config.py](mars/config.py).

### Add a GPU profile (permanent, auto-detected)

In `_GPU_PROFILES` (around line 34), add a new key matching a substring of `torch.cuda.get_device_name(0)`:

```python
_GPU_PROFILES: dict[str, dict[str, float]] = {
    "A40": { ... },           # existing
    "A100": {                 # <-- add your GPU here
        "cost_a": ...,
        "cost_c": ...,
        "cost_swap_a1": ...,        # ms/tok
        "per_token_swap_latency": ...,  # s/tok
    },
}
```

The detected profile is merged over `_GENERIC_COST_DEFAULTS` in `MarsConfig.__post_init__`, so only fields that differ from the generic defaults need to be specified.

---

## Validation scripts (`examples/test_*.py`)

Six end-to-end tests require a GPU; six unit tests require only the `mars` package (no GPU).

| Script | GPU? | What it checks |
|--------|------|----------------|
| `test_orchestrator.py` | yes | Pause/inject/resume: injected tokens condition subsequent generation |
| `test_kv_policies.py` | yes | P and D produce byte-identical output (policy is a perf/memory choice) |
| `test_vulcan_classify.py` | yes | V assigns strategy at arrival from predicted API time, not at pause |
| `test_swap.py` | yes | P, S, D produce byte-identical output with the CPU-offload connector |
| `test_demotion.py` | yes | Dynamic KV demotion fires under memory pressure; all requests finish |
| `test_cost_model_2way.py` | no | Preserve/recompute crossover is monotonic; `w_d` is API-time-independent |
| `test_cost_model_3way.py` | no | 3-way `choose` selects SWAP only when available and cheapest |
| `test_classify_v2.py` | no | Arrival classify picks correct strategy; V2 reorders queue with live batch |
| `test_queue_sjf.py` | no | SJF queue pops shortest-remain-length first |
| `test_queue_combined.py` | no | Combined V2/SJF ordering picks the smaller-key head across both queues |
| `test_queue_starvation.py` | no | Starving requests sort to front; starvation boost expires after quantum |

Run GPU tests from a neutral cwd (e.g. `/tmp`) to avoid import shadowing.

---

## Running experiments

### Quick smoke test (small model, fast)

```bash
cd /tmp   # neutral cwd

CUDA_VISIBLE_DEVICES=0 \
  MARS_derivative/vllm/.venv/bin/python \
  -m mars.bench.run \
  --workload MARS_derivative/mars-codebase/diverse_oneapi_merged_exp_uniform.json \
  --model facebook/opt-125m \
  --num-requests 16 --qps 4 \
  --api-policy V \
  --csv /tmp/mars_smoke.csv
```

### Full policy sweep (ports `6B_bench.sh`)

```bash
# Override vars to point at your model and tuning
MODEL=Qwen/Qwen2.5-14B-Instruct \
N=2000 \
QPS=4 \
OUT=/tmp/mars_exp \
GPU=0 \
bash MARS_derivative/mars-codebase/mars_port/examples/run_experiments.sh
```

The script sweeps:
- Swap-free policies: `P D V`
- Swap policies (with `SimpleCPUOffloadConnector`): `S V`

Results are written as per-request CSVs to `$OUT/`.

### Key flags for `mars.bench.run`

| Flag | Description |
|------|-------------|
| `--api-policy` | KV policy: `P` preserve, `D` recompute, `S` swap, `V` Vulcan |
| `--policy-config` | Queue ordering: `fcfs` (default), `V2` |
| `--swap` | Enable `SimpleCPUOffloadConnector` (required for S and swap-aware V) |
| `--chunk-fill` | Enable per-step token-budget cap (decoupled from demotion) |
| `--chunk-size N` | Per-step token budget when chunk-fill is on (0 = engine default) |
| `--no-demote` | Disable dynamic memory-pressure demotion entirely |
| `--demote-proactive` | Use the original per-step proactive demotion instead of the default on-demand (lazy/minimal) demotion (ablation) |
| `--starvation-avoidance` | Enable anti-starvation boosting |
| `--starvation-threshold N` | Steps before a request is boosted |
| `--starvation-quantum N` | Steps the boost lasts |
| `--num-requests N` | Total requests from the workload |
| `--qps Q` | Poisson arrival rate (requests/second) |
| `--csv PATH` | Write per-request metrics CSV |
| `--window N` | Rolling window for throughput computation |

### Predict per-policy waste without running

```bash
MARS_derivative/vllm/.venv/bin/python \
  -m mars.bench.policy_cost \
  --api-time 3.0 --before-tokens 512 --running-tokens 1024
```

### Run multiple policies
```bash
bash MARS_derivative/mars-codebase/exps/run_a40.sh
```

---

## KV policies reference

| Letter | Policy | Behaviour at API pause |
|--------|--------|----------------------|
| `P` | Preserve | Keep KV blocks pinned; resume immediately |
| `D` | Discard/Recompute | Free KV; recompute on resume (wastes GPU compute) |
| `S` | Swap | Free KV to CPU host; reload async on resume |
| `V` | Vulcan (adaptive) | Classify at arrival (predict P/S/D by cost model); always pause as P; demote to the arrival strategy **on-demand** — only the minimum preserved KV freed when a new request can't allocate (best-measured policy; `--demote-proactive` for the original every-step variant) |

---

## Cost model (the `V` waste equations)

`V` classifies each request at arrival by computing three **wastes** and taking the
argmin (tie-break **preserve > swap > recompute**). Every waste is a
**GPU-memory × time** quantity in **token-seconds**, so the three are directly
comparable: each estimates the memory-time a strategy *denies to the rest of the
system*. The strategy that wins becomes the request's `arrival_strategy` — i.e.
whether it stays pinned (preserve) or, when on-demand demotion must free memory,
whether it is demoted to swap or to recompute. Code: [`mars/cost_model.py`](mars/cost_model.py).

**Notation** (coefficients are profiled per GPU/model by `calibrate_cost.py`):

| symbol | meaning |
|---|---|
| `B` | the request's KV size, tokens (`block_size · num_blocks` ≈ tokens before the API call) |
| `T` | predicted API-call duration, seconds (`api_exec_time`) |
| `R` | KV held by the running batch, tokens (`running_blocks · block_size`) |
| `M` | `max_ragged_batch` — tokens/step at which the forward becomes compute-bound |
| `c_h` | prefill headroom this step = `max(M − running_batch, 1)`, tokens/step |
| `n` | resume/refill iterations = `max(⌈B / c_h⌉ − 1, …)` |
| `a`, `c` | forward-step time model: one step ≈ `(a·tokens + c)` ms (`a` ms/tok, `c` ms) |
| `λ` | `per_token_swap_latency` — host→GPU KV reload, s/token |

### Preserve — `w_p = T · B`
The KV sits **idle on the GPU for the whole API wait**. Cost = its own memory `B`
held for time `T` — the memory-time that could otherwise have admitted other work.
(Charged in full; on-demand demotion is what actually reclaims it later if pressure
arrives, so `w_p` only needs to say "keeping this is cheap iff the wait is short.")

### Recompute / Discard — `w_d` (real compute stolen from the batch)
```
c_h = max(M − running_batch, 1)
n   = max(⌈B / c_h⌉ − 1, 0)
f_s = (a·M + c)/1000        # one saturated forward step, s
f_ch = (a·c_h)/1000         # a headroom-sized prefill step, s
w_d = f_s·(1+n)·n/2·c_h           # cumulative delay to the batch, quadratic in n
    + f_ch·n·R                    # each of n refill steps delays the running R
    + f_last·(R + last_resume_toks)   # the final partial chunk (last_resume_toks = B mod c_h)
```
Freeing the KV is instant, but on resume the request must **recompute** its `B`
tokens of prefill in `n` chunks of `c_h`, and that is **real GPU compute** that
steals iterations from the running batch — hence the term grows with both `n`
(iterations) and `R` (the memory delayed). This is the genuinely expensive option
and is unchanged from the original MARS.

### Swap — `w_s` (v1 write-through async reload)
```
transfer = λ · B                 # async host→GPU reload time, s
f_fwd    = (a·M + c)/1000         # one forward step, s
n        = max(⌈B / c_h⌉ − 1, 1)
exposed  = max(transfer − f_fwd·n, 0)   # reload time NOT hidden behind compute
w_s = transfer · B               # own KV resident through its reload
    + exposed · R                # only the un-hidden remainder stalls the batch
```
The CPU-offload connector is **write-through**: every request's KV is mirrored
GPU→CPU as it is computed, so freeing it (swap-**out**) is a sunk,
policy-independent cost. Choosing swap therefore only adds the **swap-IN reload**,
which is **async and overlaps compute** — so unlike the original V0 blocking model
(a per-iteration stall with swap-out+in contention, `×2`), only the part of the
reload *not* hidden behind the `n` resume iterations (`exposed`) contends with the
running batch (`×1`). When the reload hides fully (`exposed = 0`, the common case)
`w_s` reduces to `λ·B²`, which is small — so swap is the cheapest free-the-memory
option exactly where it wins. Uses only the profiled `λ` plus the forward
coefficients; the original `swap_a1/a2/c` magic numbers are retired.

> The swap equation is a **deliberate v1 re-derivation**, not a port of the
> original. See [COMPARISON.md](COMPARISON.md) → *Swap data-path* for why the V0
> blocking model (and its ~1000× mis-pricing) does not apply to v1.

---

## Algorithm discrepancies from the original

See [mars_port/COMPARISON.md](COMPARISON.md) for a full subsystem-by-subsystem diff. Key point:

- **Swap cost model**: Re-derived for vLLM v1's write-through async CPU-offload (vs. the original blocking V0 swap). The original formula mis-priced v1 swap by ~1000×.

---

