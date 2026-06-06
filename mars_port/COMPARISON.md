# MARS: original (vLLM 0.2.0) vs the v1 port — algorithm-level comparison

This document compares the original MARS (a heavily-modified fork of **vLLM
0.2.0**, in `mars-codebase/vllm/` on `main`) with the **vLLM v1 port** (the `mars`
extension package in `mars_port/`, branch `mars-v1-port`). It lists, per
subsystem, what the original did, what the port does, and any **discrepancy**.

## Approach

| | Original | Port |
|---|---|---|
| Integration | Vendored, edited copy of vLLM 0.2.0 (the `vllm` package *is* MARS) | Extension package on vLLM v1 with **minimal core edits on the `mars-port` branch** (~63 LOC across 4 files): (1) `_reclaim_blocks_for_admission` hook in the waiting-admission loop (default `return False`); (2) relaxed assert in `_update_from_kv_xfer_finished` so a MARS-cancelled WFRKV stale completion is ignored; (3) `cancel_load_for_request` on the KV-connector base + `SimpleCPUOffloadConnector` + `simple_kv_offload` manager. Everything else via `scheduler_cls`, `extra_args`, native streaming, KV-connector. |
| Engine | V0 sync `LLMEngine` + custom `Scheduler` | V1 `EngineCore`; MARS overrides injected via `scheduler_cls` |
| Scheduling mode | Synchronous `engine.step()` loop only | **Both** vLLM modes. `scheduler_cls="…MARSScheduler"` is a dispatch factory: the MARS overrides are a mixin layered on `Scheduler` (sync) or `AsyncScheduler` (overlapped) per the resolved `async_scheduling` flag (default on). |

## Subsystem-by-subsystem

### Pause → API → resume
- **Original**: custom `LLMEngine.resume_request()`; new `SequenceStatus` values (`PAUSED_API`/`RESUMED_API`); pause detected via API stop-strings in `_check_stop`.
- **Port**: rides vLLM v1's **native resumable streaming** (`resumable=True` + `streaming_queue` + `StreamingUpdate`); `ApiOrchestrator` drives generate→pause→inject→resume; pause is the segment boundary (a token-interval, like the simulator).
- **Faithful**: no functional discrepancy; the resume is folded by `_update_request_as_session` (discards the last sampled token — handled). Pause is token-interval (matching the simulator), not stop-string.

### KV-cache policies (per pause)
- **Original**: `PreemptionMode.{PRESERVE, SWAP, RECOMPUTE}` chosen in `pause_seq_group`.
- **Port**: same three modes. PRESERVE = keep blocks; RECOMPUTE = `kv_cache_manager.free` + reset `num_computed_tokens` (+ `skip_reading_prefix_cache`); SWAP = free + reload from the prefix cache / CPU-offload host on resume.
- **Faithful**: PRESERVE/RECOMPUTE produce **byte-identical** output to the cached path (verified).

### Cost model (waste functions) + classify() timing
- **Original**: `discard_waste`, `swap_waste`, `classify()` (preserve/swap/recompute by min waste). Memory×time (token-seconds). `classify()` runs **once at arrival** (`add_seq_group`) on the **predicted** length, stores `strategy`+`waste` on the request; a `V` request then **always pauses as PRESERVE** — the classified swap/recompute is applied only later by the every-step demotion, and `strategy`/`waste` also feed V2 ordering + the demotion victim ranking.
- **Port**: `mars.cost_model` ports `discard_waste`/`preserve_waste` verbatim. `_mars_classify` runs once at arrival via the `_enqueue_waiting_request` hook (predicted length + live running-batch), storing `arrival_strategy`/`arrival_waste` in `_MarsReqState`. `V` pauses as PRESERVE; demotion applies the arrival strategy; `_v2_score` reads it. Strategy is re-computed on every resume (matching the original's per-resume `classify()` call). Coefficients re-calibrated via `examples/calibrate_cost.py`.
- **DELIBERATE DEVIATION — `swap_waste` re-derived for v1.** The original `swap_waste` modelled the **V0 *blocking* swap** (a per-iteration GPU stall, swap-out+swap-in contention ×2). On v1 the CPU-offload connector is a **write-through cache**: every request's KV is mirrored GPU→CPU as it is computed, so the swap-OUT is a sunk, policy-independent cost, and the swap-IN reload is async and overlaps compute. The original blocking formula mis-prices v1 swap by ~1000×. The port replaces it with a reload-only, overlap-aware model: `T = per_token_swap_latency·req_tokens`; `exposed = max(T − f_fwd·n, 0)`; `w_s = T·req_tokens + exposed·running_blocks·bs` (×1, swap-in only). Uses only the profiled `per_token_swap_latency` (s/tok) + the forward coeffs.
- **Faithful**: classify at arrival on predicted length; pause is preserve; strategy consumed by demotion + V2 — matching the original.

### Dynamic memory-pressure demotion
- **Original**: in `_schedule_chunk_and_fill`, **every step (no pressure gate)** keep only the single highest-`waste` preserved-paused request pinned and demote the rest **to their arrival-classified `strategy`**; V is swap-aware. Pure P is never demoted.
- **Port**: a `schedule()` pre-pass (`_mars_demote_paused`) does the same — every step, `_demote_choice` reads each request's `arrival_strategy`/`arrival_waste`; `max_waste` is taken over **all** paused-preserve requests (including `preserve`-classified ones) and every demotable request with `waste < max_waste` is demoted to its strategy; the costliest stays pinned. Pure `P` is never demoted. Gated on a `demote_paused` flag (default on; `--no-demote` disables) and pending admission demand.
- **Faithful**: matches the original's per-step "keep only max-waste preserved" cadence. **Caveat**: v1 has no mid-pause swap-back-in — a demoted-SWAP request stays freed until it resumes (reloads from the CPU-offload host then); gating on `waiting` avoids needless free+reload round-trips.
- **Empirical (A40 / Qwen2.5-14B, qps 3–13)**: per-step demotion is a **net latency cost** on this single-API workload — it churns swap-out→reload that lands on the resume critical path, and under load can tip into a near-livelock (e.g. qps9 norm-latency 11.3 s/tok @ 89% drained vs 0.20 s/tok @ 100% for `--no-demote`). `--no-demote` (keep the arrival classification, skip the per-step churn) is the **best-performing config** across the sweep; static `S` (swap) is the most robust at saturation. A pressure-gated variant (only demote above a KV-usage floor of 0.6/0.8) removed the meltdowns and restored drain but **still lost to `--no-demote`**, so it was removed. Demotion is retained for original-faithfulness but is **not recommended** for this workload (use `--no-demote`).

### Waiting-queue ordering (`policy_config`)
- **Original**: `PolicyFactory` — `fcfs` / `sjf` / `V2` (cost-based memory-time ranking, re-sorted each step with live `running_batch`; smallest score first). Sort applied to `combined_targets = self.swapped + self.waiting`, so swapped-out and new arrivals competed in the same priority order.
- **Port**: FCFS native; `MARSRequestQueue` (SJF on `remain_length`); `V2RequestQueue` (min-heap, smallest score first). Both `self.waiting` and `self.skipped_waiting` are replaced with MARS heaps. `V2` is **re-keyed every step** in the `schedule()` pre-pass with the **live `running_batch`** and the full 3-branch score (`_v2_score`). `_select_waiting_queue_for_scheduling` picks whichever non-empty queue has the smaller head `peek_key()` — combined V2/SJF order. `rekey_interval` config (default 1) mirrors the original's `skip_sorting_for_this_number_of_iterations`. Parked requests skip scoring (`+inf` key) to avoid O(n log n) cost at high qps.
- **Faithful**: live per-step re-rank with actual running batch; combined ordering across resumed-swap and new arrivals replicates the original `combined_targets` sort.

### Starvation avoidance
- **Original**: per-seq `starvation_counter`/`quantum`; boosted to front for `quantum` steps.
- **Port**: per-request wait counter in a `schedule()` pre-pass; once `> starvation_threshold`, marked *starving* (key `-inf` in MARS queues, `prepend` for FCFS) and boosted for exactly `starvation_quantum` steps, then un-boosted and counter reset. `remaining_quantum` tracked in `_MarsReqState`.
- **Faithful**: exact per-step quantum countdown.

### chunk-fill
- **Original**: fine-grained ragged-batch token shaping + partial swap-in chunks per step (`_set_max_ragged_batch`, `_active_discard`).
- **Port**: **approximated** by v1's native chunked prefill + a `token_budget` cap (`chunk_size`). `chunk_fill` now controls **only** the per-step token-budget cap; demotion is decoupled and gated solely by `demote_paused` (`--no-demote`). **Empirical**: enabling the `chunk_size` cap (1024 or 2048) is a throughput throttle under load — peeling it off (no chunk-fill) lowers tail latency, e.g. qps11 norm-latency 2.0 vs 1.7 s/tok with the cap removed.
- **DISCREPANCY**: per-step token-shaping / partial swap-in is not reproduced (v1 owns batching); the headline behavior (demote under pressure to fill) is faithfully approximated.

### Swap data-path
- **Original**: explicit V0 `BlockSpaceManager.swap_out/in` (GPU↔CPU blocks). Proactive mid-wait reload was possible because swapped requests were in `self.swapped` and `_schedule` issued `blocks_to_swap_in` for any request it promoted.
- **Port**: vLLM's native **`SimpleCPUOffloadConnector`** — free at pause, reload from host on resume via the `WAITING_FOR_REMOTE_KVS` async path (GPU blocks allocated + host→GPU copy on a low-priority CUDA stream, overlapping other requests' compute). Swap requires `enable_prefix_caching`.
- **DISCREPANCY — reload starts at resume, not during the API wait.** In practice, the evaluated original (`scheduler_v2.py`) only re-entered swapped requests at `resume_seq_group` anyway, so the "reload during wait" was a latent V0 capability, not what the paper measured. Measured on A40 / Qwen2.5-14B: `post_resume_ttft_s` ≈ 0.17–0.31 s mean, only **0.8–1.3 % of e2e** — the existing async path already overlaps the copy effectively. A proactive mid-wait preload was prototyped (port-only, zero vLLM core edits) and then **removed**: it is inert under v1's write-through CPU-offload cache — the GPU prefix cache re-serves swap-freed KV locally, so even forcing the headroom gate at high load it fired **0 times** across qps 3–13. A microbenchmark confirmed concurrent host↔GPU KV copies (D2H+H2D, up to ~800 MB) inflate a 2048-token prefill iteration by only **2–5 %**, i.e. reload bandwidth overlaps compute almost for free — so demotion's real cost is **admission/queueing** (a resuming request stalling for GPU blocks under pressure), not transfer overhead.
- **Also**: swap is whole-request (no partial block swap).

### Memory-pressure admission / passive_discard
- **Original**: `passive_discard_by_order` over `combined_targets` evicted the lowest-priority victim inline when admission was blocked — immune to deadlock.
- **Port**: `_reclaim_blocks_for_admission` hook (vLLM `mars-port` edit, default `return False`). When admission is blocked **and `running==0`**, picks the lowest-V2-priority KV-holder from `mars_resumed_preserved` (resumed-PRESERVE holders), `skipped_waiting` (WAITING with `num_computed_tokens>0`), or in-flight `WAITING_FOR_REMOTE_KVS` loads, calls `_demote` to free its KV, and returns False — admission resumes next step. The base call site ignores the return and always breaks (no same-step retry; the bool is advisory). Counter `mars_reclaims` + `[MARS] reclaim …` log.
- **Faithful**: admission is blocked only when no KV-holding waiting request exists — matching the original's safety property.

### KV bookkeeping
- **Original**: `SequenceData` tracked KV regions (`discard_start_idx`/`swap_length`/`inflight_length`, `resume_*`).
- **Port**: replaced by `num_computed_tokens` reset + the prefix cache; a tiny per-request `_MarsReqState` (policy, pending_reset, pending_skip_prefix).
- **Discrepancy**: no per-region bookkeeping (subsumed by v1 block management); consistent with whole-request swap.

### Calibration
- **Original**: coefficients hard-coded / sed-patched per model (`6B_bench.sh`: `f_ch a=0.0408,c=16.92`; swap poly `a=0.00462,b=108.99`).
- **Port**: `examples/calibrate_cost.py` measures forward linear fit (`cost_a`, `cost_c`) and swap host↔GPU round-trip (`per_token_swap_latency`, `cost_swap_a1`) empirically per model/GPU and prints a paste-ready `MarsConfig`.

### Benchmark / experiments
- **Original**: synchronous `engine.step` loop in `fixed_final_tput_bench_real.py`; `exps/6B_bench.sh` sweep.
- **Port**: async `mars.bench.run` (AsyncLLM + `ApiOrchestrator`, Poisson arrivals, `--window`); `examples/run_experiments.sh` reproduces the policy matrix; `mars.bench.policy_cost` gives a predicted per-policy waste table.

## Remaining discrepancies (open)

1. **chunk-fill is approximated** by `token_budget` + demotion (no per-step ragged-batch token shaping / partial swap-in) — the largest fidelity gap in the evaluated path, since the original `V` relied on chunk-fill's fine-grained scheduling.
2. **Swap reload starts at API-return, not during the API wait** — the mid-wait-preload prototype was **removed** (inert under v1 write-through caching: 0 fires across qps 3–13 even forcing the gate; and per-iteration host↔GPU copy contention is only **2–5 %**). Measured `post_resume_ttft_s` is ~1 % of e2e on the steady path — the async `WAITING_FOR_REMOTE_KVS` reload already overlaps compute. Not a fidelity gap worth pursuing.
3. **Swap is whole-request** via the CPU-offload connector (no partial block swap). Requires `enable_prefix_caching`.
