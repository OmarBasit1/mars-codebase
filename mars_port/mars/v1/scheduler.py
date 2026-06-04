"""MARS scheduler for vLLM v1.

Injected via ``SchedulerConfig.scheduler_cls`` (``--scheduler-cls
mars.v1.scheduler.MARSScheduler``). Every override *wraps* ``super()`` rather
than replacing it, so the base scheduling machinery — including the scheduler-
side KV-connector hooks — keeps working unchanged.

Phase 3 adds the per-pause **KV-cache policy** on top of vLLM's native resumable
streaming:

  * A resumable request that stops mid-session is *parked*
    (``WAITING_FOR_STREAMING_REQ``) by the base ``_handle_stopped_request``,
    **keeping its KV blocks** — i.e. PRESERVE is the native default.
  * For RECOMPUTE we free the request's KV blocks at the pause (so the GPU
    memory is available during the API wait) and reset ``num_computed_tokens``
    after the resume-fold so the whole sequence is recomputed.
  * SWAP degrades to the configured fallback until a CPU-offload connector is
    wired (Phase 6).

The policy is purely a performance/memory choice: PRESERVE and RECOMPUTE must
produce identical output tokens (verified by the Phase 3 smoke test).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from mars.config import MarsConfig
from mars.cost_model import CostModel, CostModelCoeffs
from mars.params import MarsApiParams
from mars.policies import (
    PauseMode,
    decide_pause_mode,
    degrade_swap,
)
from mars.v1.queue import _MarsHeapQueue, make_mars_queue
from mars.v1.solver import GUROBI_AVAILABLE, MarsSolver, SolverParams, split_to_mode

# Name under the "vllm" namespace so MARS logs inherit vLLM's log handler
# (vLLM configures a handler on the "vllm" logger only, propagate=False).
logger = init_logger("vllm.mars.scheduler")

# Cost-model PauseMode <-> the original's strategy strings (stored at arrival by
# classify() and consumed by demotion + the V2 ordering branch).
_MODE_TO_STRATEGY = {
    PauseMode.PRESERVE: "preserve",
    PauseMode.RECOMPUTE: "recompute",
    PauseMode.SWAP: "swap",
}


@dataclass
class _MarsReqState:
    """Per-request MARS bookkeeping (replaces old SequenceData KV-region fields)."""

    policy_letter: str
    pauses: int = 0
    # Set when blocks were freed at a pause (SWAP or RECOMPUTE); triggers a
    # num_computed_tokens reset after the next resume-fold.
    pending_reset: bool = False
    # For RECOMPUTE, also bypass the prefix cache on resume (force a true
    # recompute). For SWAP this stays False so the prefix cache / CPU-offload
    # connector serves the KV back from host instead of recomputing.
    pending_skip_prefix: bool = False
    # Arrival-time classify() result (predicted length): the KV strategy the
    # request will be demoted to, and its waste. Faithful to the original
    # classify(): consumed by the every-step demotion (which mode + the
    # max-waste victim ranking) and the V2 ordering branch. The V pause itself
    # always PRESERVEs; the strategy is applied only at demotion.
    arrival_strategy: str = ""  # "preserve" / "recompute" / "swap"
    arrival_waste: float = 0.0
    swap_reloads: int = 0  # times KV was reloaded from CPU (SWAP resumes)
    remaining_quantum: int = 0  # starvation-boost steps left (0 = not boosted)


class _MARSSchedulerMixin:
    """MARS per-pause KV-cache policy overrides.

    A mixin (no scheduler base of its own) so it can be layered on top of *either*
    vLLM scheduler base: the synchronous :class:`Scheduler` or the overlapped
    :class:`AsyncScheduler`. Every override calls ``super()`` so it composes with
    whichever base it is combined with (see :class:`MARSSyncScheduler` /
    :class:`MARSAsyncScheduler` and the :class:`MARSScheduler` dispatch factory).
    Under ``AsyncScheduler`` the base's output-placeholder bookkeeping
    (``_update_after_schedule`` / ``_update_request_with_output``) is picked up
    via the MRO, so MARS works correctly with vLLM's async scheduling too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mars_config = MarsConfig.from_vllm_config(self.vllm_config)
        self.mars_state: dict[str, _MarsReqState] = {}
        # Swap is only real once a CPU-offload KV connector is configured (Phase 6).
        # SWAP needs a CPU-offload KV connector AND prefix caching (the connector
        # is backed by the prefix cache). Otherwise swap-ish policies degrade.
        self._swap_available = (
            self.connector is not None
            and bool(getattr(self.cache_config, "enable_prefix_caching", False))
        )
        self._block_size = getattr(self.cache_config, "block_size", None) or 16
        # cost_max_ragged_batch: if not set, read the per-step token budget from
        # vllm's scheduler_config (the point where compute saturates). Falls back
        # to 384 if the attribute is unavailable (old configs).
        if self.mars_config.cost_max_ragged_batch is None:
            max_batched = getattr(
                getattr(self.vllm_config, "scheduler_config", None),
                "max_num_batched_tokens",
                None,
            )
            self.mars_config.cost_max_ragged_batch = int(max_batched) if max_batched else 384
        # Cost model for the adaptive 'V' (Vulcan) policy.
        self.cost_model = CostModel(
            CostModelCoeffs(
                a=self.mars_config.cost_a,
                c=self.mars_config.cost_c,
                max_ragged_batch=self.mars_config.cost_max_ragged_batch,
                block_size=self._block_size,
                swap_a1=self.mars_config.cost_swap_a1,
                swap_a2=self.mars_config.cost_swap_a2,
                swap_c=self.mars_config.cost_swap_c,
            )
        )
        # Optional Gurobi solver for 'V' (decision-only): falls back to the
        # greedy cost model when unavailable.
        self.solver: MarsSolver | None = None
        if self.mars_config.use_solver:
            if GUROBI_AVAILABLE:
                self.solver = MarsSolver(
                    SolverParams(
                        block_size=self._block_size,
                        target=self.mars_config.solver_target,
                        timeout=self.mars_config.solver_timeout,
                        free_swap_tokens=self.mars_config.solver_free_swap_tokens,
                        per_token_swap_latency=self.mars_config.solver_per_token_swap_latency,
                        poly_a=self.mars_config.solver_poly_a,
                        poly_b=self.mars_config.solver_poly_b,
                        poly_c=self.mars_config.solver_poly_c,
                    )
                )
            else:
                logger.warning(
                    "[MARS] use_solver set but gurobipy unavailable; "
                    "falling back to the greedy cost model."
                )
        # chunk-fill (simplified): cap the per-step token budget so paused
        # (preserved) KV plus new work fit. Full dynamic chunk-fill is Phase 6.
        if self.mars_config.chunk_fill and self.mars_config.chunk_size > 0:
            self.max_num_scheduled_tokens = min(
                self.max_num_scheduled_tokens, self.mars_config.chunk_size
            )
        # Starvation tracking (shared with the MARS queues so boosted requests
        # sort to the front via key -inf).
        self.mars_starving: set[str] = set()
        self.mars_wait_counter: dict[str, int] = {}
        # Waiting-queue ordering: SJF / V2 replace both native FCFS queues (both
        # are empty at construction time, so swapping is safe).  ``skipped_waiting``
        # holds parked-for-API requests and resumed requests (they flip status in
        # place and stay in skipped_waiting).  Replacing it with the same MARS queue
        # (shared starving set, same key function) enables combined V2/SJF ordering
        # across both queues — faithful to the original's combined_targets sort.
        mars_queue = make_mars_queue(
            self.mars_config.policy_config,
            self.mars_starving,
            self.mars_config.cost_max_ragged_batch,
        )
        if mars_queue is not None:
            self.waiting = mars_queue
            self.skipped_waiting = make_mars_queue(
                self.mars_config.policy_config,
                self.mars_starving,
                self.mars_config.cost_max_ragged_batch,
            )
        # Gate combined-queue overrides on MARS queues actually being active.
        self._mars_queues: bool = isinstance(self.waiting, _MarsHeapQueue)
        # Async scheduling pipelines outputs: a request scheduled in a prior step
        # is still in flight (its KV blocks are referenced by an un-retired batch)
        # when the next step's MARS pre-pass runs. Freeing those blocks then (a
        # demotion) corrupts the in-flight batch and wedges the engine. Track the
        # mode so the demotion pre-pass can defer in-flight requests by a step.
        self._async_sched: bool = bool(
            getattr(self.scheduler_config, "async_scheduling", False)
        )
        # Observability counters (read by tests / logged).
        self.mars_total_pauses = 0
        self.mars_preserve_count = 0
        self.mars_recompute_count = 0
        self.mars_swap_count = 0
        self.mars_swap_reloads = 0  # SWAP resumes -> async host->GPU reload engaged
        # Preserved-paused requests holding pinned KV -> candidates for dynamic
        # memory-pressure demotion.
        self.mars_paused_preserved: dict[str, Request] = {}
        self.mars_demotions = 0
        # Resumed-PRESERVE requests: still hold KV blocks but are back in
        # skipped_waiting/waiting (no longer in mars_paused_preserved). These are
        # the deadlock culprits at high qps: vLLM's native preemption only evicts
        # running requests, so when memory fills with running=0 nothing can be
        # admitted. _reclaim_blocks_for_admission evicts the lowest-priority victim
        # from this pool, mirroring the original's passive_discard over
        # combined_targets.
        self.mars_resumed_preserved: dict[str, Request] = {}
        self.mars_reclaims = 0  # passive_discard invocations
        # V2 amortization counter (C2): rekey only every rekey_interval steps.
        self._mars_rekey_counter: int = 0
        logger.info(
            "[MARS] scheduler active: api_policy=%s policy_config=%s "
            "swap_available=%s block_size=%s chunk_fill=%s per_req_stats=%r",
            self.mars_config.api_policy,
            self.mars_config.policy_config,
            self._swap_available,
            self._block_size,
            self.mars_config.chunk_fill,
            self.mars_config.per_req_stats_path or "(disabled)",
        )
        # Optional hang diagnostic (env-gated, no overhead when off): a daemon
        # thread that every MARS_HANG_DIAG seconds logs queue/memory state so a
        # stall can be characterised (slow-drain vs memory deadlock) without
        # ptrace/py-spy. Also arms a periodic faulthandler traceback dump.
        if os.environ.get("MARS_HANG_DIAG"):
            self._start_hang_diag(int(os.environ["MARS_HANG_DIAG"]))

    def _start_hang_diag(self, interval: int) -> None:
        import threading

        def _diag():
            import time as _t
            while True:
                _t.sleep(interval)
                try:
                    w = len(self.waiting)
                    sw = len(self.skipped_waiting)
                    run = len(self.running)
                    pp = len(self.mars_paused_preserved)
                    usage = getattr(self.kv_cache_manager, "usage", -1)
                    # count schedulable (WAITING) vs parked in skipped_waiting
                    sched = parked = 0
                    if isinstance(self.skipped_waiting, _MarsHeapQueue):
                        for r in self.skipped_waiting.iter_unsorted():
                            if r.status == RequestStatus.WAITING:
                                sched += 1
                            else:
                                parked += 1
                    rp = len(self.mars_resumed_preserved)
                    logger.info(
                        "[MARS-DIAG] running=%d waiting=%d skipped=%d "
                        "(sched=%d parked=%d) paused_preserved=%d "
                        "resumed_preserved=%d kv_usage=%.3f "
                        "demotions=%d reclaims=%d swap_reloads=%d",
                        run, w, sw, sched, parked, pp, rp, usage,
                        self.mars_demotions, self.mars_reclaims,
                        self.mars_swap_reloads,
                    )
                except Exception as e:
                    logger.info("[MARS-DIAG] error: %r", e)

        t = threading.Thread(target=_diag, daemon=True, name="mars-hang-diag")
        t.start()

    # --- policy resolution -------------------------------------------------

    def _api_policy_for(self, request: Request) -> str:
        """Per-request ``api_policy`` (extra_args) overriding the engine default."""
        mp = MarsApiParams.from_sampling_params(request.sampling_params)
        if mp is not None and mp.api_policy:
            return mp.api_policy
        return self.mars_config.api_policy

    def _running_contention(self) -> tuple[int, int]:
        """Live (running_batch, running_blocks) — the recompute/swap contention.

        ``running_batch`` ~ tokens the running requests compute this step (the
        original ``inflight_length`` sum); ``running_blocks`` ~ their GPU blocks.
        """
        bs = self._block_size
        running_batch = sum(
            max(1, r.num_tokens - r.num_computed_tokens) for r in self.running
        )
        running_blocks = sum(
            (r.num_computed_tokens + bs - 1) // bs for r in self.running
        )
        return running_batch, running_blocks

    # --- arrival hook: classify() once, on predicted length ----------------

    def _enqueue_waiting_request(self, request: Request) -> None:
        # Faithful to the original ``add_seq_group -> classify``: pick the KV
        # strategy once, at arrival, from the *predicted* length. Gated on
        # ``mars_state`` so resumes / preemption re-enqueues never re-classify.
        if request.request_id not in self.mars_state:
            self._mars_classify(request)
        super()._enqueue_waiting_request(request)

    def _mars_classify(self, request: Request) -> None:
        """Choose strategy + waste at arrival using the *predicted* length.

        Ports the original ``classify()``: predict blocks from
        ``prompt_len + predicted_api_invoke_interval``, score preserve/recompute/
        swap (with the live running-batch contention), store the argmin. The
        result drives demotion (which mode, and the max-waste ranking) and the
        V2 ordering branch — NOT the pause itself (V always preserves).
        """
        letter = self._api_policy_for(request)
        st = _MarsReqState(policy_letter=letter)
        self.mars_state[request.request_id] = st
        mp = MarsApiParams.from_sampling_params(request.sampling_params)
        if mp is None:
            st.arrival_strategy = "preserve"
            return
        bs = self._block_size
        prompt_len = getattr(request, "num_prompt_tokens", 0) or 0
        seq_blocks = max(1, (prompt_len + mp.predicted_api_invoke_interval + bs - 1) // bs)
        before_api_tokens = seq_blocks * bs
        running_batch, running_blocks = self._running_contention()
        mode, wastes = self.cost_model.choose(
            api_exec_time=mp.predicted_api_exec_time,
            before_api_tokens=before_api_tokens,
            num_blocks=seq_blocks,
            running_batch=running_batch,
            running_blocks=running_blocks,
            swap_available=self._swap_available,
        )
        # Optional decision-only solver (V only) overrides the greedy mode; the
        # waste for ranking is still taken from the greedy cost model.
        if self.solver is not None and letter == "V":
            smode = self._solver_mode(request, mp, seq_blocks, running_batch, running_blocks)
            if smode is not None and smode in wastes:
                mode = smode
        st.arrival_strategy = _MODE_TO_STRATEGY[mode]
        st.arrival_waste = wastes[mode]
        logger.info(
            "[MARS] classify req=%s policy=%s -> strategy=%s waste=%.4g (predicted)",
            request.request_id, letter, st.arrival_strategy, st.arrival_waste,
        )

    def _solver_mode(self, request, mp, num_blocks, running_batch, running_blocks):
        """Decision-only Gurobi solve -> dominant whole-request mode (or None)."""
        res = self.solver.solve_blocks(
            num_tokens=request.num_tokens,
            num_active_gpu_blocks=running_blocks,
            api_exec_time=mp.predicted_api_exec_time,
            api_return_length=mp.api_return_length,
            arrival_time=request.arrival_time,
            now=time.time(),
            running_query_head=running_batch,
            running_query_tail=running_batch,
            swap_in_chunks_head=self.solver.free_swap,
            swap_in_chunks_tail=self.solver.free_swap,
        )
        if res is None:
            return None
        c_s, c_d, n_e = res
        mode, _ = split_to_mode(
            c_s, c_d, n_e, num_blocks,
            swap_available=self._swap_available,
            swap_fallback=self.mars_config.swap_fallback,
        )
        return mode

    # --- pause hook --------------------------------------------------------

    def _handle_stopped_request(self, request: Request) -> bool:
        finished = super()._handle_stopped_request(request)
        # A resumable request that just *parked* awaiting the next input chunk
        # is paused for an API call -> apply the KV policy now (during the wait).
        if not finished and request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self._apply_pause_policy(request)
        return finished

    def _apply_pause_policy(self, request: Request) -> None:
        st = self.mars_state.get(request.request_id)
        if st is None:
            st = _MarsReqState(policy_letter=self._api_policy_for(request))
            self.mars_state[request.request_id] = st
        st.pauses += 1
        self.mars_total_pauses += 1

        mode, detail = self._decide_pause_mode(request, st.policy_letter)
        if mode is PauseMode.PRESERVE:
            # Keep blocks resident; track as a demotion candidate (unless pure P,
            # which is never demoted -- the baseline that just pins memory).
            self.mars_preserve_count += 1
            if st.policy_letter != "P":
                self.mars_paused_preserved[request.request_id] = request
            logger.info(
                "[MARS] pause req=%s policy=%s mode=preserve (kept KV)%s",
                request.request_id,
                st.policy_letter,
                detail,
            )
            return
        # SWAP or RECOMPUTE: free the paused request's KV now (idempotent
        # req_to_blocks.pop) so the GPU memory is available during the API wait.
        # num_computed_tokens is left intact so the resume-fold can still locate
        # the kept output tokens; it is reset (and, for RECOMPUTE, the prefix
        # cache bypassed) after the fold.
        self.kv_cache_manager.free(request)
        st.pending_reset = True
        if mode is PauseMode.SWAP:
            # Reload the KV from the prefix cache / CPU-offload host on resume.
            st.pending_skip_prefix = False
            self.mars_swap_count += 1
            logger.info(
                "[MARS] pause req=%s policy=%s mode=swap (freed KV -> host)%s",
                request.request_id,
                st.policy_letter,
                detail,
            )
        else:  # RECOMPUTE
            st.pending_skip_prefix = self.mars_config.recompute_skip_prefix_cache
            self.mars_recompute_count += 1
            logger.info(
                "[MARS] pause req=%s policy=%s mode=recompute (freed KV)%s",
                request.request_id,
                st.policy_letter,
                detail,
            )

    def _decide_pause_mode(self, request: Request, policy_letter: str):
        """Resolve the concrete pause mode and a log-detail string.

        Routing by ``policy_letter``:
          * ``V`` -> always PRESERVE at the pause (faithful to the original: the
            swap/recompute classified at arrival is applied later by the every-step
            demotion in :meth:`schedule` -- ``_mars_demote_paused``/
            ``_demote_choice`` using the arrival-classified strategy).
          * ``P``/``D``/``S`` -> direct mode (swap degraded when unavailable).
        """
        if policy_letter == "V":
            # Static PRESERVE at the pause; demotion applies the arrival strategy.
            st = self.mars_state.get(request.request_id)
            strat = st.arrival_strategy if st is not None else ""
            return PauseMode.PRESERVE, f" (preserve; arrival_strategy={strat or '?'})"
        # Direct P / D / S.
        mode = decide_pause_mode(
            policy_letter,
            swap_available=self._swap_available,
            swap_fallback=self.mars_config.swap_fallback,
        )
        return mode, ""

    # --- V2 ordering + dynamic demotion -----------------------------------

    def schedule(self):
        # Pre-pass before the base scheduler admits/preempts:
        #  0. clean up mars_resumed_preserved for requests that became RUNNING or
        #     FINISHED since the last step (they were admitted or finished normally).
        #  1. starvation — boost long-waiting requests to the front;
        #  2. V2 re-key — re-rank the waiting queue with the live running_batch;
        #  3. demotion — apply the arrival-classified strategy to preserved KV.
        if self.mars_resumed_preserved:
            done = [
                rid for rid, req in self.mars_resumed_preserved.items()
                if req.status != RequestStatus.WAITING
            ]
            for rid in done:
                self.mars_resumed_preserved.pop(rid, None)
        if self.mars_config.starvation_avoidance:
            self._mars_starvation_pass()
        # Re-rank V2 with the live running_batch every rekey_interval steps
        # (faithful to the original's skip_sorting_for_this_number_of_iterations).
        if self.mars_config.policy_config == "V2":
            self._mars_rekey_counter += 1
            if self._mars_rekey_counter >= self.mars_config.rekey_interval:
                self._mars_rekey_counter = 0
                self._mars_rekey_v2()
        # chunk_fill (the original's switch for the demotion machinery) or the
        # explicit demote_under_pressure knob enables dynamic demotion.
        if (
            (self.mars_config.demote_under_pressure or self.mars_config.chunk_fill)
            and len(self.mars_paused_preserved) > 1
        ):
            self._mars_demote_paused()
        return super().schedule()

    def _select_waiting_queue_for_scheduling(self):
        """Combined V2/SJF ordering across ``waiting`` and ``skipped_waiting``.

        When both are MARS heaps, pick whichever non-empty queue has the smaller
        head key (true combined order — faithful to the original's
        ``combined_targets = self.swapped + self.waiting`` sorted together). The
        base's pop→try-promote→requeue loop handles unpromotable heads (still
        waiting for API) gracefully: they move to ``step_skipped_waiting`` for
        the remainder of the pass, so there is no infinite loop.
        For FCFS (no MARS queues) we fall through to the base implementation.
        """
        if not self._mars_queues:
            return super()._select_waiting_queue_for_scheduling()
        w_has = bool(self.waiting)
        s_has = bool(self.skipped_waiting)
        if not w_has and not s_has:
            return None
        if w_has and not s_has:
            return self.waiting
        if s_has and not w_has:
            return self.skipped_waiting
        # Both non-empty: the head with the smaller key goes first.
        return (
            self.waiting
            if self.waiting.peek_key() <= self.skipped_waiting.peek_key()
            else self.skipped_waiting
        )

    def _mars_rekey_v2(self) -> None:
        """Re-rank both V2 heaps with the live ``running_batch`` (O(n) each).

        Faithful to the original ``sort_by_priority(running_batch=live)`` run
        every step: the V2 score's compute terms scale with contention, so the
        ordering adapts to load (the insertion-time ``running_batch=0`` key is a
        cheap placeholder that this overwrites). Both ``waiting`` and
        ``skipped_waiting`` are re-keyed so resumed swap requests in
        ``skipped_waiting`` compete under the same live score — replicating the
        original's combined swapped+waiting sort.

        Only *schedulable* (``WAITING``) requests get a live V2 score. Parked
        API requests (``WAITING_FOR_STREAMING_REQ`` / ``WAITING_FOR_REMOTE_KVS``)
        sit in ``skipped_waiting`` until they resume and cannot be admitted, so
        re-parsing + re-scoring them every step is pure waste — and pathological
        at high qps, where thousands park at the tail (``skipped_waiting`` is NOT
        bounded by ``max_num_seqs``). They get a cheap ``+inf`` key (sorted to the
        back, behind every schedulable request) instead. This avoids an O(n)
        ``MarsApiParams`` parse per step that otherwise pegs a CPU core and
        stalls the engine. See COMPARISON.md.
        """
        running_batch, running_blocks = self._running_contention()

        def key_fn(r: Request) -> float:
            if r.status != RequestStatus.WAITING:
                return float("inf")  # parked: not admittable; skip the score+parse
            return self._v2_score(r, running_batch, running_blocks)

        for q in (self.waiting, self.skipped_waiting):
            if isinstance(q, _MarsHeapQueue) and q:
                q.rekey(key_fn)

    def _mem_time(self, num_blocks: int, running_batch: int) -> float:
        """V2 memory-time of computing ``num_blocks`` (ports calculate_memory_time_blocks).

        Uses V2's own coefficients (a=0.1, c=10), distinct from the cost model's.
        """
        co = self.cost_model.coeffs
        c_h = max(co.max_ragged_batch - running_batch, 1)
        n = max((co.block_size * num_blocks + c_h - 1) // c_h, 1)
        f_s = (0.1 * co.max_ragged_batch + 10.0) / 1000.0
        return f_s * (1 + n) * n / 2 * c_h

    def _v2_score(self, request: Request, running_batch: int, running_blocks: int) -> float:
        """V2 ordering score (lower = scheduled first), live ``running_batch``.

        Ports ``policy.py:V2.get_priority``'s three strategy branches, keyed off
        the request's arrival-classified strategy. The original returns ``-score``
        sorted ``reverse=True`` (== smallest score first); the MARS min-heap pops
        the smallest key first, so we return ``score`` directly.
        """
        mp = MarsApiParams.from_sampling_params(request.sampling_params)
        if mp is None:
            return float("inf")
        bs = self._block_size
        prompt_len = getattr(request, "num_prompt_tokens", 0) or 0
        before_blocks = (prompt_len + mp.predicted_api_invoke_interval + bs - 1) // bs
        after_blocks = (mp.predicted_api_invoke_interval + mp.api_return_length + bs - 1) // bs
        before = self._mem_time(before_blocks, running_batch)
        after = self._mem_time(after_blocks, running_batch)
        api_exec_time = mp.predicted_api_exec_time
        api_complete = mp.api_max_calls == 0
        st = self.mars_state.get(request.request_id)
        strat = st.arrival_strategy if st is not None and st.arrival_strategy else "preserve"
        if strat == "recompute":
            before_after = self._mem_time(before_blocks + after_blocks, running_batch)
            return before_after if api_complete else before + 0.0 + before_after
        if strat == "swap":
            cpu_to_gpu_transfer_rate = 2  # original hyperparameter
            swap = before_blocks * (before_blocks / cpu_to_gpu_transfer_rate) / 2
            api = api_exec_time * 0.1  # integral_swap_weight
            return (swap + after) if api_complete else before + swap + api + swap + after
        # preserve
        api_memory = before_blocks * bs * api_exec_time
        return before + api_memory + after

    def _mars_starvation_pass(self) -> None:
        """Boost requests that have waited too long to the front of the queue.

        Each step, increment a wait counter for every *schedulable* request in
        either ``self.waiting`` or ``self.skipped_waiting`` (status==WAITING only;
        parked WAITING_FOR_STREAMING_REQ / WAITING_FOR_REMOTE_KVS are skipped —
        boosting something that cannot run yet is pointless). Once the counter
        exceeds ``starvation_threshold`` the request is marked *starving* (key
        ``-inf`` in both MARS heaps) so it sorts to the front of the combined
        order (the original's ``quantum`` is "boosted until scheduled").
        """
        threshold = self.mars_config.starvation_threshold
        quantum = self.mars_config.starvation_quantum  # steps to keep boosted
        waiting_ids: set[str] = set()
        boosted = 0
        expired = 0
        # Scan both queues; only count requests that are currently schedulable.
        queues = [self.waiting]
        if self._mars_queues:
            queues.append(self.skipped_waiting)
        for queue in queues:
            # iter_unsorted (MARS heaps) avoids the O(n log n) sort of
            # list(queue)/__iter__; order is irrelevant here (we only scan +
            # boost by status), and at high qps skipped_waiting holds thousands
            # of parked requests. Native FCFS queues fall back to plain iteration.
            scan = (
                queue.iter_unsorted()
                if isinstance(queue, _MarsHeapQueue)
                else list(queue)
            )
            for req in scan:
                if req.status != RequestStatus.WAITING:
                    continue  # parked / in-KV-transfer: skip
                rid = req.request_id
                waiting_ids.add(rid)
                st = self.mars_state.get(rid)
                if rid in self.mars_starving:
                    # Quantum countdown: decrement each step while boosted.
                    if st is not None and st.remaining_quantum > 0:
                        st.remaining_quantum -= 1
                        if st.remaining_quantum == 0:
                            # Quantum expired: un-boost and reset wait counter so
                            # the request has a fresh chance before re-boosting.
                            self.mars_starving.discard(rid)
                            self.mars_wait_counter.pop(rid, None)
                            self._mars_boost(req, queue)  # re-insert with normal key
                            expired += 1
                    continue
                count = self.mars_wait_counter.get(rid, 0) + 1
                self.mars_wait_counter[rid] = count
                if count > threshold:
                    self.mars_starving.add(rid)
                    if st is not None:
                        st.remaining_quantum = max(1, quantum)
                    self._mars_boost(req, queue)
                    boosted += 1
        # Drop bookkeeping for requests that left waiting (scheduled/finished).
        for rid in list(self.mars_wait_counter):
            if rid not in waiting_ids:
                self.mars_wait_counter.pop(rid, None)
                self.mars_starving.discard(rid)
        if boosted:
            logger.info("[MARS] starvation: boosted %d request(s) to front", boosted)
        if expired:
            logger.debug("[MARS] starvation: quantum expired for %d request(s)", expired)

    def _mars_boost(self, request: Request, queue=None) -> None:
        # Remove from the source queue (which was already determined by the
        # starvation pass or the caller) and re-add; the -inf starvation key is
        # applied by _key() because the request is already in mars_starving.
        if queue is None:
            queue = self.waiting
        queue.remove_request(request)
        queue.prepend_request(request)

    def _mars_demote_paused(self) -> None:
        # Faithful to the original _schedule_chunk_and_fill: every step keep only
        # the single highest-waste preserved-paused request pinned and demote the
        # rest. Demoting each request right after it pauses (rather than waiting
        # for a memory wall) spreads the CPU-offload (PCIe) traffic across the
        # workload instead of bursting it in one step. Gated only on pending
        # admission demand -- there is no point freeing KV nothing is waiting for
        # (v1 reloads at resume, so a needless demote is a wasted host round-trip).
        # Gate on work actually waiting (either queue); resumed swap requests
        # in skipped_waiting also need memory, so they count as demand.
        if not self.waiting and not (self._mars_queues and self.skipped_waiting):
            return
        usage = self.kv_cache_manager.usage
        # Optional usage floor (0 => no gate, the faithful default).
        threshold = self.mars_config.demote_pressure_threshold
        if threshold > 0 and usage < threshold:
            return
        # Faithful to the original _schedule_chunk_and_fill: keep the single
        # highest-(arrival-)waste request pinned (and any 'preserve'-classified
        # ones), demote the rest to their arrival-classified strategy. Modes and
        # wastes come from classify() (predicted), NOT a live recompute.
        # 'preserve'-classified candidates (mode None) are never demoted; among
        # the demotable ones keep exactly the costliest (sorting breaks ties so
        # identical-waste requests don't all stay pinned -> no deadlock).
        demotable = [
            (waste, rid, req, mode)
            for rid, req in self.mars_paused_preserved.items()
            if not self._mars_request_inflight(req)
            for mode, waste in [self._demote_choice(rid, req)]
            if mode is not None
        ]
        if len(demotable) <= 1:
            return
        demotable.sort(key=lambda x: x[0])
        for waste, rid, req, mode in demotable[:-1]:  # all but the costliest
            self._demote(rid, req, mode, waste, usage)

    def _mars_request_inflight(self, request: Request) -> bool:
        """True if the request still has async outputs in flight.

        Under async scheduling the engine schedules the next batch before the
        previous one retires, so a request scheduled in a prior step still has
        its KV blocks referenced by an un-retired batch (tracked by
        ``num_output_placeholders`` and ``prev_step_scheduled_req_ids``). Freeing
        those blocks now -- e.g. a demotion -- corrupts the in-flight batch and
        wedges the engine. Deferring the free by one step (until the in-flight
        outputs drain) is safe and faithful. Always ``False`` under sync
        scheduling (no pipeline window), so this is a no-op there.
        """
        if getattr(request, "num_output_placeholders", 0) > 0:
            return True
        return (
            self._async_sched
            and request.request_id in self.prev_step_scheduled_req_ids
        )

    def _reclaim_blocks_for_admission(
        self, request: Request, num_new_tokens: int
    ) -> bool:
        """Passive-discard: free the lowest-priority KV-holding waiting request.

        Called by the base when allocate_slots returns None for a waiting
        request.  Picks the victim with the smallest V2 waste (least worth
        keeping) from ``mars_resumed_preserved`` (resumed-PRESERVE requests that
        still hold GPU blocks but are waiting, not running — the deadlock
        culprits), frees its KV, and returns True so the base retries admission.
        Returns False when no eligible victim exists (base falls back to break).

        Mirrors the original's ``passive_discard_by_order`` over
        ``combined_targets`` — the safety net that made the original immune to
        this deadlock.
        """
        if not self.mars_resumed_preserved:
            return False
        usage = self.kv_cache_manager.usage
        # Build candidate list: resumed-PRESERVE, WAITING, not in-flight,
        # not the request being admitted itself.
        admit_id = request.request_id
        candidates = [
            (self._v2_score(req, 0, 0), rid, req)
            for rid, req in self.mars_resumed_preserved.items()
            if rid != admit_id
            and req.status == RequestStatus.WAITING
            and not self._mars_request_inflight(req)
        ]
        if not candidates:
            return False
        # Evict the LOWEST priority (highest V2 score = least worth keeping).
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, victim_rid, victim = candidates[0]
        mode, waste = self._demote_choice(victim_rid, victim)
        if mode is None:
            # arrival_strategy == 'preserve': fall back to recompute so we
            # actually free something (matches the original's fallback).
            mode = PauseMode.RECOMPUTE
            waste = 0.0
        self._demote(victim_rid, victim, mode, waste, usage)
        self.mars_reclaims += 1
        logger.info(
            "[MARS] reclaim: freed KV of req=%s (mode=%s) to admit req=%s",
            victim_rid,
            mode.value,
            admit_id,
        )
        return True

    def _demote_choice(self, rid: str, request: Request):
        """``(mode, waste)`` to demote a preserved-paused request, from classify().

        Applies the arrival-classified strategy (swap/recompute). A
        'preserve'-classified request returns ``mode=None`` (it is never demoted),
        matching the original (a PRESERVE-strategy candidate stays pinned).
        """
        st = self.mars_state.get(rid)
        waste = st.arrival_waste if st is not None else 0.0
        strat = st.arrival_strategy if st is not None else "recompute"
        if strat == "swap" and self._swap_available:
            return PauseMode.SWAP, waste
        if strat == "recompute":
            return PauseMode.RECOMPUTE, waste
        return None, waste  # 'preserve' (or 'swap' w/o connector): stay pinned

    def _demote(self, rid, request, mode, waste, usage) -> None:
        # Free the preserved-paused KV now; resume reloads (SWAP) or recomputes
        # (RECOMPUTE) -- the same mechanism as a SWAP/RECOMPUTE pause.
        self.kv_cache_manager.free(request)
        st = self.mars_state.get(rid)
        if st is None:
            st = _MarsReqState(policy_letter=self._api_policy_for(request))
            self.mars_state[rid] = st
        st.pending_reset = True
        self.mars_preserve_count -= 1
        if mode is PauseMode.SWAP:
            st.pending_skip_prefix = False
            self.mars_swap_count += 1
        else:
            st.pending_skip_prefix = self.mars_config.recompute_skip_prefix_cache
            self.mars_recompute_count += 1
        self.mars_paused_preserved.pop(rid, None)
        self.mars_resumed_preserved.pop(rid, None)
        self.mars_demotions += 1
        logger.info(
            "[MARS] demote req=%s -> %s waste=%.4g (usage=%.2f, %d preserved left)",
            rid,
            mode.value,
            waste,
            usage,
            len(self.mars_paused_preserved),
        )

    # --- resume hook -------------------------------------------------------

    def _update_request_as_session(self, session: Request, update) -> None:
        super()._update_request_as_session(session, update)
        # Resuming -> no longer a preserved-paused demotion candidate.
        self.mars_paused_preserved.pop(session.request_id, None)
        st = self.mars_state.get(session.request_id)
        if st is not None and not st.pending_reset:
            # KV blocks were NOT freed at pause (PRESERVE mode). The request
            # resumes holding its GPU blocks; track it so passive_discard can
            # reclaim them if memory is exhausted during admission.
            self.mars_resumed_preserved[session.request_id] = session
        if st is not None and st.pending_reset:
            # Blocks were released at the pause. Recompute from num_computed=0;
            # SWAP lets the prefix cache / CPU-offload host serve the prefix back,
            # while RECOMPUTE additionally bypasses the cache to force a true
            # recompute (prompt + kept output + injected API tokens).
            is_swap = not st.pending_skip_prefix  # SWAP=False/RECOMPUTE=True
            session.num_computed_tokens = 0
            if st.pending_skip_prefix:
                session.skip_reading_prefix_cache = True
            if is_swap:
                # The connector will serve KV from host asynchronously
                # (WAITING_FOR_REMOTE_KVS overlap) when the request is next admitted.
                self.mars_swap_reloads += 1
                st.swap_reloads += 1
                logger.info(
                    "[MARS] swap reload req=%s (async host->GPU via connector)",
                    session.request_id,
                )
            st.pending_reset = False
            st.pending_skip_prefix = False

    # --- cleanup -----------------------------------------------------------

    def _free_request(self, request: Request, *args, **kwargs):
        st = self.mars_state.pop(request.request_id, None)
        self.mars_paused_preserved.pop(request.request_id, None)
        self.mars_resumed_preserved.pop(request.request_id, None)
        path = self.mars_config.per_req_stats_path
        if path and st is not None:
            # vLLM appends "-{8hex}" to every request_id for internal uniqueness
            # (input_processor.py:240).  Strip the suffix so the bench can look up
            # records by the original caller-supplied ID.
            rid = request.request_id
            ext_id = rid.rsplit("-", 1)[0] if len(rid) > 9 and rid[-9] == "-" else rid
            with open(path, "a") as _f:
                _f.write(json.dumps({
                    "request_id": ext_id,
                    "policy": st.policy_letter,
                    "arrival_strategy": st.arrival_strategy or "preserve",
                    "swap_reloads": st.swap_reloads,
                }) + "\n")
        return super()._free_request(request, *args, **kwargs)


# --- concrete schedulers + dispatch factory --------------------------------
# vLLM picks the scheduler class from ``scheduler_cls`` *before* it resolves the
# ``async_scheduling`` flag, and (unlike the no-custom-class path) it never swaps
# in ``AsyncScheduler`` for a custom class. So we expose a factory under the
# stable name ``MARSScheduler`` that instantiates the right base per the resolved
# flag, layering the MARS overrides on top of either base via the MRO.


class MARSSyncScheduler(_MARSSchedulerMixin, Scheduler):
    """MARS over vLLM's synchronous scheduler (``async_scheduling=False``)."""


class MARSAsyncScheduler(_MARSSchedulerMixin, AsyncScheduler):
    """MARS over vLLM's overlapped scheduler (``async_scheduling=True``).

    ``AsyncScheduler`` adds the in-flight-token placeholder bookkeeping that
    overlapped scheduling needs; the MARS overrides sit above it in the MRO and
    pick it up through their ``super()`` calls.
    """


class MARSScheduler:
    """Dispatch factory: select the sync or async MARS scheduler.

    Used as ``scheduler_cls="mars.v1.scheduler.MARSScheduler"``. vLLM instantiates
    it with all-keyword args (``vllm_config=...`` etc.); we read
    ``vllm_config.scheduler_config.async_scheduling`` and return a fully built
    :class:`MARSAsyncScheduler` or :class:`MARSSyncScheduler`. Because those are
    not subclasses of this factory, Python skips ``MARSScheduler.__init__`` and
    the chosen class's ``__init__`` is the only one that runs.
    """

    def __new__(cls, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        async_sched = bool(
            getattr(getattr(vllm_config, "scheduler_config", None),
                    "async_scheduling", False)
        )
        target = MARSAsyncScheduler if async_sched else MARSSyncScheduler
        logger.info(
            "[MARS] using %s (async_scheduling=%s)", target.__name__, async_sched
        )
        return target(*args, **kwargs)
