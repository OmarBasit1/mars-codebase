# MARS: original (vLLM 0.2.0) vs the v1 port — algorithm-level comparison

This document compares the original MARS (a heavily-modified fork of **vLLM
0.2.0**, in `mars-codebase/vllm/` on `main`) with the **vLLM v1 port** (the `mars`
extension package in `mars_port/`, branch `mars-v1-port`). It lists, per
subsystem, what the original did, what the port does, and any **discrepancy**.

## Approach

| | Original | Port |
|---|---|---|
| Integration | Vendored, edited copy of vLLM 0.2.0 (the `vllm` package *is* MARS) | Extension package on vLLM v1 with **minimal core edits on the `mars-port` branch** (`vllm/` repo, `mars-port` branch, editable install; `main` stays pristine). Core change: a single `_reclaim_blocks_for_admission` hook (~6 LOC) in the waiting-admission loop. Everything else: `scheduler_cls`, `extra_args`, native streaming, KV-connector. |
| Engine | V0 sync `LLMEngine` + custom `Scheduler` | V1 `EngineCore`; MARS overrides injected via `scheduler_cls` |
| Scheduling mode | Synchronous `engine.step()` loop only | **Both** vLLM modes. `scheduler_cls="…MARSScheduler"` is a dispatch factory: the MARS overrides are a mixin layered on `Scheduler` (sync) or `AsyncScheduler` (overlapped) per the resolved `async_scheduling` flag (default on). Async placeholder bookkeeping is picked up from `AsyncScheduler` via the MRO. |

## Subsystem-by-subsystem

### Pause → API → resume
- **Original**: custom `LLMEngine.resume_request()`; new `SequenceStatus` values (`PAUSED_API`/`RESUMED_API`); pause detected via API stop-strings in `_check_stop`.
- **Port**: rides vLLM v1's **native resumable streaming** (`resumable=True` + `streaming_queue` + `StreamingUpdate`); `ApiOrchestrator` drives generate→pause→inject→resume; pause is the segment boundary (a token-interval, like the simulator).
- **Discrepancy**: none functionally; the resume is folded by `_update_request_as_session` (which discards the last sampled token — handled). Pause is token-interval (matching the simulator), not stop-string.

### KV-cache policies (per pause)
- **Original**: `PreemptionMode.{PRESERVE, SWAP, RECOMPUTE}` chosen in `pause_seq_group`.
- **Port**: same three modes. PRESERVE = keep blocks; RECOMPUTE = `kv_cache_manager.free` + reset `num_computed_tokens` (+ `skip_reading_prefix_cache`); SWAP = free + reload from the prefix cache / CPU-offload host on resume.
- **Faithful**: PRESERVE/RECOMPUTE produce **byte-identical** output to the cached path (verified). 

### Cost model (waste functions) + classify() timing
- **Original**: `discard_waste`, `swap_waste`, `classify()` (preserve/swap/recompute by min waste). Memory×time (token-seconds). `classify()` runs **once at arrival** (`add_seq_group`) on the **predicted** length, stores `strategy`+`waste` on the request; a `V` request then **always pauses as PRESERVE** — the classified swap/recompute is applied only later by the every-step demotion, and `strategy`/`waste` also feed V2 ordering + the demotion victim ranking.
- **Port**: `mars.cost_model` ports the formulas verbatim; greedy `choose`. `_mars_classify` runs once at arrival via the `_enqueue_waiting_request` hook (predicted length + live running-batch), storing `arrival_strategy`/`arrival_waste` in `_MarsReqState`. `V` pauses as PRESERVE; demotion applies the arrival strategy; `_v2_score` reads it. Coefficients re-calibrated via `examples/calibrate_cost.py`.
- **Faithful**: classify is at **arrival on predicted length**, pause is preserve, strategy consumed by demotion + V2 — matching the original. (Earlier the port decided at the pause on the actual length; now reverted to the original's arrival/prediction.)

### Solver (NOT used in the original evaluation)
- **Original**: `core/solver.py` defines a Gurobi MIP that splits ONE request's KV into preserved `s_p` + swapped `c_s`/iter + recomputed `c_d`/iter over `n_e` iterations — **but it is never called by the schedulers.** `Solver` is only an *unused import* in `scheduler*.py`; the sole `Solver()` instantiation is solver.py's own `__main__` self-test (line 134). The evaluated MARS (`exps/6B_bench.sh`, `--api-policy V`) therefore decided KV strategy with the **greedy `classify()`** (min of `w_p`/`w_d`/`w_s` at arrival) + the V2 queue + chunk-fill demotion — NOT the MIP (a per-pause Gurobi solve is too slow for serving).
- **Port**: the greedy `cost_model.choose` (default, `use_solver=False`) **faithfully reproduces what the original evaluation actually ran**. We *additionally* wired the MIP **decision-only** (`use_solver=True`) — run the MIP **at arrival** (in `_mars_classify`) and apply the dominant whole-request mode as the arrival strategy — an enhancement the original never executed. (v1 has no per-request partial-KV execution, so the MIP's partial split is collapsed to a whole-request choice; falls back to greedy if Gurobi is unavailable/times out.)
- **Discrepancy**: none for the *default* config (greedy `V` matches the paper). For parity use `use_solver=False`; the `use_solver=True` path is *beyond* the original (and is whole-request, not partial-split).

### Dynamic memory-pressure demotion (Greedy / InferCept)
- **Original**: in `_schedule_chunk_and_fill`, **every step (no pressure gate)** keep only the single highest-`waste` preserved-paused request pinned (plus any `preserve`-classified ones) and demote the rest **to their arrival-classified `strategy`** (using the classify-stored `waste` for the max comparison); InferCept is recompute-only, V/G swap-aware.
- **Port**: a `schedule()` pre-pass (`_mars_demote_paused`) does the same — **every step**, `_demote_choice` reads each request's `arrival_strategy`/`arrival_waste` (from classify, NOT a live recompute); demote all whose waste `< max_waste` to that strategy, keeping the costliest (and `preserve`-classified) pinned; `I` recompute-only. Pure `P` is never demoted. Gated only on pending admission demand; optional `demote_pressure_threshold > 0` re-enables a usage floor (default 0 = faithful). Demoting each request right after it pauses spreads the CPU-offload (PCIe) traffic instead of bursting it at a memory wall.
- **Faithful**: matches the original's per-step "keep only max-waste preserved" cadence. **Caveat**: v1 has no mid-pause swap-back-in — a demoted-SWAP request stays freed until it *resumes* (reloads from the CPU-offload host then), whereas the original could swap it back in earlier; gating on `waiting` avoids needless free+reload round-trips.

### Waiting-queue ordering (`policy_config`)
- **Original**: `PolicyFactory` — `fcfs` / `sjf` / `V2` (cost-based memory-time ranking, re-sorted each step with live `running_batch`; `sort_by_priority` uses `reverse=True` on `-score` ⇒ smallest score first). Crucially, the sort was applied to `combined_targets = self.swapped + self.waiting`, so swapped-out (demoted) requests and new arrivals competed in the same priority order.
- **Port**: FCFS native; `MARSRequestQueue` (SJF on `remain_length`); `V2RequestQueue` (min-heap, smallest score first = matches the original's direction). **Both** `self.waiting` and `self.skipped_waiting` are replaced with MARS heaps (same `policy_config`, shared `mars_starving` set). `V2` is **re-keyed every step** on both heaps in the `schedule()` pre-pass (`_mars_rekey_v2`) with the **live `running_batch`** and the full **3-branch** (`preserve`/`recompute`/`swap`) score (`_v2_score`). `_select_waiting_queue_for_scheduling` is overridden to pick whichever non-empty queue has the smaller head `peek_key()` — the combined V2/SJF order. Starvation (`-inf` key) wins from either queue. O(n) re-heapify per heap per step, n ≤ `max_num_seqs`.
- **Faithful**: live per-step re-rank with actual running batch; combined ordering across resumed (swap-demoted) and new requests replicates the original `combined_targets` sort. `rekey_interval` config (default 1 = every step) mirrors the original's `skip_sorting_for_this_number_of_iterations`; set to 8 to match the original's amortization. Re-key is also skipped for parked requests (status ≠ WAITING → `+inf`), and `iter_unsorted()` is used in the starvation pass, eliminating the O(n log n) cost over thousands of parked requests in `skipped_waiting` at high qps.

### Starvation avoidance
- **Original**: per-seq `starvation_counter`/`quantum`; high-priority requests put at the front of the schedule targets for `quantum` steps.
- **Port**: per-request wait counter in a `schedule()` pre-pass; once `> starvation_threshold`, the request is marked *starving* (key `-inf` in the MARS queues, `prepend` for FCFS) and boosted to the front for exactly `starvation_quantum` steps (from `MarsConfig`), then un-boosted and the counter reset, matching the original's countdown. `remaining_quantum` tracked in `_MarsReqState`.
- **Faithful**: exact per-step quantum countdown restored (C1).

### chunk-fill
- **Original**: fine-grained ragged-batch token shaping + partial swap-in chunks per step (`_set_max_ragged_batch`, `_active_discard`).
- **Port**: **approximated** by v1's native chunked prefill + a `token_budget` cap (`chunk_size`) + the dynamic demotion. `chunk_fill=True` enables the demotion machinery (as in the original).
- **DISCREPANCY**: the per-step token-shaping / partial swap-in is not reproduced (v1 owns batching); the headline behavior (demote under pressure to fill) is.

### Swap data-path
- **Original**: explicit V0 `BlockSpaceManager.swap_out/in` (GPU↔CPU blocks), partial swap (`first_n_blocks`). When memory freed up the scheduler would proactively reload swapped KV from CPU to GPU *during the API wait*, before the API returned, because swapped requests were in `self.swapped` and `_schedule` issued `blocks_to_swap_in` for any request it admitted.
- **Port**: vLLM's native **`SimpleCPUOffloadConnector`** (transparent CPU offload backed by the prefix cache) — free at pause, reload from host on resume. Enabled via `mars.swap`. Reload path: `_update_request_as_session` sets `num_computed_tokens=0`; on next admission the connector's `get_num_new_matched_tokens` returns `(hit_length, is_async=True)`, triggering vLLM's `WAITING_FOR_REMOTE_KVS` async path — GPU blocks are allocated and the worker copies host→GPU on a low-priority CUDA stream, **overlapping other requests' compute**. The resumed request is scheduled the following step. Logged as `[MARS] swap reload req=... (async host->GPU via connector)`; the `mars_swap_reloads` counter tracks occurrences.
- **DISCREPANCY (known gap — measure before building)**: v1 cannot start the reload *before* the API returns. vLLM binds all KV loads to a compute admission (`num_new_tokens > 0`), and `_try_promote_blocked_waiting_request` refuses to promote a still-parked `WAITING_FOR_STREAMING_REQ` request. Starting an early reload requires a new `WAITING_FOR_STREAMING_REQ_PRELOADING` sub-state + a `_mars_preload_parked` pre-pass + core edits (see plan). **However**: in the *evaluated* original (`scheduler_v2.py`), demoted-SWAP requests only re-entered `self.swapped` (reload-eligible) at `resume_seq_group` ([scheduler_v2.py:276](../vllm/core/scheduler_v2.py#L276)) — so the V2 results reloaded **at/after resume** as the port already does. The "reload during the API wait" was a latent V0 data-path capability, not what the paper measured. Decision: instrument `post_resume_ttft_s` (per-request CSV column, mean time from API-return to first token) and build the proactive preload only if the reload latency is demonstrably on the critical path.
- **Also**: per-request *partial* swap is unavailable; swap is whole-request. Requires `enable_prefix_caching`.

### Memory-pressure admission / passive_discard
- **Original**: `_schedule` in `scheduler_v2.py` freed memory *inline* during admission via `passive_discard_by_order` over `combined_targets = self.swapped + self.waiting` — when a candidate couldn't be allocated, the lowest-priority victim (any request in `combined_targets` or `self.paused`) was evicted (swap/recompute), and admission was retried. The safety net made the original immune to memory-over-commit deadlocks: a victim was always reachable regardless of whether `self.running` was empty.
- **Port (before fix)**: vLLM v1 only preempts *running* requests ([scheduler.py:486](../vllm/vllm/v1/core/sched/scheduler.py#L486)) and merely defers waiting requests that can't allocate ([scheduler.py:739](../vllm/vllm/v1/core/sched/scheduler.py#L739)). MARS demotion only targeted `mars_paused_preserved` (paused, not yet resumed). A resumed-PRESERVE request — one that resumed still holding its KV blocks and is now `WAITING` — was neither a demotion candidate nor a vLLM preemption target. At qps≥11 on Qwen2.5-14B these accumulated until KV usage hit 100% with `running=0`; nothing could be admitted, demotions froze, and the engine livelocked (confirmed via the `MARS_HANG_DIAG` diagnostic thread).
- **Port (fixed)**: a `_reclaim_blocks_for_admission` hook (~6 LOC) was added to the vLLM `mars-port` branch's waiting-admission loop (default `return False` = backward-compatible). The MARS mixin overrides it: when admission is blocked, picks the lowest-V2-priority request from `mars_resumed_preserved` (resumed-PRESERVE, WAITING, not in-flight), calls `_demote` to free its KV, and returns `True` for a retry — mirroring the original's `passive_discard_by_order`. `mars_resumed_preserved` is populated in `_update_request_as_session` when a request resumes still holding blocks (`pending_reset=False`), and cleaned up on demote, schedule, and `_free_request`. Counter `mars_reclaims` + `[MARS] reclaim req=... for=...` log.
- **Faithful**: admission is now blocked only when no KV-holding waiting request exists — matching the original's safety property.

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
1. **chunk-fill is approximated** by `token_budget` + demotion (no per-step ragged-batch token shaping / partial swap-in) — the largest fidelity gap in the *evaluated* path, since the original `V` relied on chunk-fill's fine-grained scheduling.
2. **Swap reload starts at API-return, not during the API wait.** v1 cannot initiate the host→GPU KV copy before a request is admitted for compute; the original could start it mid-wait when memory freed up. The reload still overlaps the following step's compute via `WAITING_FOR_REMOTE_KVS` async path. **Note**: in the *evaluated* original, demoted-SWAP requests only became reload-eligible at `resume_seq_group` anyway — the "proactive reload" was a latent capability, not the measured path. The `post_resume_ttft_s` CSV column instruments this for a decision on building the proactive preload.
3. **Swap is whole-request via the CPU-offload connector** (no partial block swap).
4. **The MILP solver was UNUSED in the original evaluation** (the greedy `classify()` was used). The port's default greedy `V` (`use_solver=False`) reproduces that; `use_solver=True` is an *optional* decision-only enhancement (whole-request, not partial-split) — **not** a fidelity gap for the default config.
5. ~~Starvation quantum simplified to "boost until scheduled"~~ — **fixed (C1)**: per-step quantum countdown with `remaining_quantum` in `_MarsReqState`.
6. ~~V2 `skip_sorting` amortization not ported~~ — **fixed (C2)**: `rekey_interval` config mirrors the original; parked requests skip scoring; `iter_unsorted()` eliminates O(n log n) in the starvation pass.
7. **Memory-pressure deadlock** (qps≥11, `running=0`, KV full, no victim) — **fixed (A)**: `_reclaim_blocks_for_admission` hook + `mars_resumed_preserved` passive_discard, faithful to the original's `passive_discard_by_order`.

Everything else (pause/resume, the three KV policies, the greedy cost-model formulas the eval actually used, **arrival-time classify() on predicted length**, **V-pauses-preserve + demote-by-arrival-strategy**, demotion victim selection, **live-`running_batch` V2 re-ranking each step**, **combined V2/SJF ordering across resumed-swap and new arrivals** (faithful to original `combined_targets`), SJF/FCFS ordering, calibration) is a **faithful** port. PRESERVE/SWAP/RECOMPUTE are verified to yield identical output, so the policies remain a pure performance/memory choice as in the paper.
