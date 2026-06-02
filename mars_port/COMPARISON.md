# MARS: original (vLLM 0.2.0) vs the v1 port — algorithm-level comparison

This document compares the original MARS (a heavily-modified fork of **vLLM
0.2.0**, in `mars-codebase/vllm/` on `main`) with the **vLLM v1 port** (the `mars`
extension package in `mars_port/`, branch `mars-v1-port`). It lists, per
subsystem, what the original did, what the port does, and any **discrepancy**.

## Approach

| | Original | Port |
|---|---|---|
| Integration | Vendored, edited copy of vLLM 0.2.0 (the `vllm` package *is* MARS) | Extension package on an **unmodified** vLLM v1 (`scheduler_cls`, `extra_args`, native streaming, KV-connector). **Zero vLLM core edits.** |
| Engine | V0 sync `LLMEngine` + custom `Scheduler` | V1 `EngineCore`; `MARSScheduler(Scheduler)` injected via `scheduler_cls` |

## Subsystem-by-subsystem

### Pause → API → resume
- **Original**: custom `LLMEngine.resume_request()`; new `SequenceStatus` values (`PAUSED_API`/`RESUMED_API`); pause detected via API stop-strings in `_check_stop`.
- **Port**: rides vLLM v1's **native resumable streaming** (`resumable=True` + `streaming_queue` + `StreamingUpdate`); `ApiOrchestrator` drives generate→pause→inject→resume; pause is the segment boundary (a token-interval, like the simulator).
- **Discrepancy**: none functionally; the resume is folded by `_update_request_as_session` (which discards the last sampled token — handled). Pause is token-interval (matching the simulator), not stop-string.

### KV-cache policies (per pause)
- **Original**: `PreemptionMode.{PRESERVE, SWAP, RECOMPUTE}` chosen in `pause_seq_group`.
- **Port**: same three modes. PRESERVE = keep blocks; RECOMPUTE = `kv_cache_manager.free` + reset `num_computed_tokens` (+ `skip_reading_prefix_cache`); SWAP = free + reload from the prefix cache / CPU-offload host on resume.
- **Faithful**: PRESERVE/RECOMPUTE produce **byte-identical** output to the cached path (verified). 

### Cost model (waste functions)
- **Original**: `discard_waste`, `swap_waste`, `classify()` (preserve/swap/recompute by min waste). Memory×time (token-seconds).
- **Port**: `mars.cost_model` ports the same formulas verbatim; greedy `choose` (2-way without swap, 3-way with). Coefficients re-calibrated via `examples/calibrate_cost.py`.
- **Discrepancy**: `classify` ran at **arrival** (predicted length); the port decides at the **pause** using the *actual* length (more accurate).

### Solver (the biggest discrepancy)
- **Original**: a Gurobi MIP (`core/solver.py`) splits ONE request's KV into preserved `s_p` + swapped `c_s`/iter + recomputed `c_d`/iter over `n_e` iterations, executed **incrementally across steps** (`resume_swap_blocks`/`resume_discard_tokens`/`running_inflight_tokens`).
- **Port**: same MIP (`mars/v1/solver.py`), but **decision-only** — it runs the MIP and applies the **dominant whole-request mode** (largest block share → PRESERVE/SWAP/RECOMPUTE). v1 has no per-request *partial*-KV-region execution; adding it would need large core surgery (model runner + attention metadata + scheduler), breaking the zero-core-edit design.
- **DISCREPANCY (largest)**: no partial split. A request that the original would *partially* swap and *partially* recompute is, in the port, swapped **or** recomputed as a whole. Gated by `use_solver`; falls back to the greedy cost model if Gurobi is unavailable/times out.

### Dynamic memory-pressure demotion (Greedy / InferCept)
- **Original**: in `_schedule_chunk_and_fill`, each step demote the cheaper preserved-paused requests by waste, keeping the highest-waste one; InferCept is recompute-only, V/G swap-aware.
- **Port**: a `schedule()` pre-pass (`_mars_demote_under_pressure`) does the same — under KV pressure (`usage ≥ demote_pressure_threshold`) with requests waiting, demote all but the costliest preserved-paused request; `I` recompute-only, others swap-aware. Pure `P` is never demoted.
- **Discrepancy**: the original demoted ~every step (the "fill" then admitted work); the port triggers **only under pressure** (v1's base scheduler owns admission/preemption). Same effect — free pinned KV so waiting work runs.

### Waiting-queue ordering (`policy_config`)
- **Original**: `PolicyFactory` — `fcfs` / `sjf` / `V2` (cost-based waste ranking, re-sorted each step with live `running_batch`).
- **Port**: FCFS native; `MARSRequestQueue` (SJF on `remain_length`); `V2RequestQueue` (ports `policy.py:V2.get_priority`). 
- **Discrepancy**: heap keys are fixed at insertion, so `V2` uses `running_batch=0` and the `preserve` strategy (can't re-rank with a live running batch each step). Relative ordering preserved.

### Starvation avoidance
- **Original**: per-seq `starvation_counter`/`quantum`; high-priority requests put at the front of the schedule targets for `quantum` steps.
- **Port**: per-request wait counter in a `schedule()` pre-pass; once `> starvation_threshold`, the request is marked *starving* (key `-inf` in the MARS queues, `prepend` for FCFS) and boosted to the front **until scheduled** (quantum ≈ "until scheduled").
- **Discrepancy**: the explicit per-step quantum countdown is simplified to "boosted until scheduled" (equivalent to the paper's quantum=100000).

### chunk-fill
- **Original**: fine-grained ragged-batch token shaping + partial swap-in chunks per step (`_set_max_ragged_batch`, `_active_discard`).
- **Port**: **approximated** by v1's native chunked prefill + a `token_budget` cap (`chunk_size`) + the dynamic demotion. `chunk_fill=True` enables the demotion machinery (as in the original).
- **DISCREPANCY**: the per-step token-shaping / partial swap-in is not reproduced (v1 owns batching); the headline behavior (demote under pressure to fill) is.

### Swap data-path
- **Original**: explicit V0 `BlockSpaceManager.swap_out/in` (GPU↔CPU blocks), partial swap (`first_n_blocks`).
- **Port**: vLLM's native **`SimpleCPUOffloadConnector`** (transparent CPU offload backed by the prefix cache) — free at pause, reload from host on resume. Enabled via `mars.swap`.
- **DISCREPANCY**: per-request *partial* swap is unavailable; swap is whole-request. Requires `enable_prefix_caching`.

### KV bookkeeping
- **Original**: `SequenceData` tracked KV regions (`discard_start_idx`/`swap_length`/`inflight_length`, `resume_*`).
- **Port**: replaced by `num_computed_tokens` reset + the prefix cache; a tiny per-request `_MarsReqState` (policy, pending_reset, pending_skip_prefix).
- **Discrepancy**: no per-region bookkeeping (subsumed by v1 block management); consistent with the whole-request solver.

### Calibration
- **Original**: coefficients hard-coded / sed-patched per model (`6B_bench.sh`: `f_ch a=0.0408,c=16.92`; swap poly `a=0.00462,b=108.99`).
- **Port**: `examples/calibrate_cost.py` measures them empirically per model/GPU (forward linear + quadratic batch-poly; swap host↔GPU round-trip; preserve≈0) and prints a paste-ready `MarsConfig`.

### Benchmark / experiments
- **Original**: synchronous `engine.step` loop in `fixed_final_tput_bench_real.py`; `exps/6B_bench.sh` sweep with `--load-format dummy`.
- **Port**: async `mars.bench.run` (AsyncLLM + `ApiOrchestrator`, Poisson arrivals, `--window`); `examples/run_6b_bench.sh` reproduces the matrix; `mars.bench.policy_cost` gives a predicted per-policy waste table.

## Discrepancy summary (ranked by impact)
1. **Solver is decision-only (whole-request), not partial-split** — the single largest algorithmic difference. May change absolute numbers on requests where the optimal is a genuine mix of swap+recompute; the *direction* of the decision is preserved.
2. **Swap is whole-request via the CPU-offload connector** (no partial block swap).
3. **chunk-fill is approximated** by token_budget + demotion (no per-step ragged-batch shaping).
4. **V2 queue uses a static (insertion-time) key** (running_batch=0), not re-ranked each step.
5. **Starvation quantum simplified** to "boost until scheduled".

Everything else (pause/resume, the three KV policies, the cost model formulas, the greedy 3-way / solver decision, demotion victim selection, SJF/FCFS ordering, calibration) is a **faithful** port. PRESERVE/SWAP/RECOMPUTE are verified to yield identical output, so the policies remain a pure performance/memory choice as in the paper.
