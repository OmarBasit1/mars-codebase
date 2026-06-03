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
    THRESHOLD_POLICIES,
    PauseMode,
    decide_pause_mode,
    decide_threshold_mode,
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
        # Waiting-queue ordering: SJF / V2 replace the native FCFS queue (it is
        # empty at construction time, so swapping is safe).
        mars_queue = make_mars_queue(self.mars_config.policy_config, self.mars_starving)
        if mars_queue is not None:
            self.waiting = mars_queue
        # Observability counters (read by tests / logged).
        self.mars_total_pauses = 0
        self.mars_preserve_count = 0
        self.mars_recompute_count = 0
        self.mars_swap_count = 0
        # Preserved-paused requests holding pinned KV -> candidates for dynamic
        # memory-pressure demotion.
        self.mars_paused_preserved: dict[str, Request] = {}
        self.mars_demotions = 0
        logger.info(
            "[MARS] scheduler active: api_policy=%s policy_config=%s "
            "swap_available=%s block_size=%s chunk_fill=%s",
            self.mars_config.api_policy,
            self.mars_config.policy_config,
            self._swap_available,
            self._block_size,
            self.mars_config.chunk_fill,
        )

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
          * ``V``/``G``/``I`` -> always PRESERVE at the pause (faithful to the
            original: a Vulcan request pauses preserved; the swap/recompute it
            was classified for at arrival is applied later by the every-step
            demotion in :meth:`schedule` -- ``_mars_demote_paused`` /
            ``_demote_choice``, where ``I`` demotes recompute-only and ``V``/``G``
            use the arrival-classified strategy).
          * ``H``/``H-S``/``H-D``/``H-B`` -> ``api_exec_time`` threshold
            heuristics (swap arms degraded when swap is unavailable).
          * ``P``/``D``/``S`` -> direct mode (swap degraded when unavailable).
        """
        if policy_letter in ("V", "G", "I"):
            # Static PRESERVE at the pause; demotion applies the arrival strategy.
            st = self.mars_state.get(request.request_id)
            strat = st.arrival_strategy if st is not None else ""
            return PauseMode.PRESERVE, f" (preserve; arrival_strategy={strat or '?'})"
        if policy_letter in THRESHOLD_POLICIES:
            mp = MarsApiParams.from_sampling_params(request.sampling_params)
            api_exec_time = mp.api_exec_time if mp is not None else 1.0
            mode = decide_threshold_mode(
                policy_letter,
                api_exec_time=api_exec_time,
                heuristic_coef=self.mars_config.heuristic_coef,
            )
            mode = degrade_swap(
                mode,
                swap_available=self._swap_available,
                swap_fallback=self.mars_config.swap_fallback,
            )
            return mode, f" api_t={api_exec_time:g}"
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
        #  1. starvation — boost long-waiting requests to the front;
        #  2. V2 re-key — re-rank the waiting queue with the live running_batch;
        #  3. demotion — apply the arrival-classified strategy to preserved KV.
        if self.mars_config.starvation_avoidance:
            self._mars_starvation_pass()
        # Re-rank V2 with the live running_batch each step (after starvation so
        # the -inf boosts survive), faithful to the original per-step re-sort.
        if self.mars_config.policy_config == "V2":
            self._mars_rekey_v2()
        # chunk_fill (the original's switch for the demotion machinery) or the
        # explicit demote_under_pressure knob enables dynamic demotion.
        if (
            (self.mars_config.demote_under_pressure or self.mars_config.chunk_fill)
            and len(self.mars_paused_preserved) > 1
        ):
            self._mars_demote_paused()
        return super().schedule()

    def _mars_rekey_v2(self) -> None:
        """Re-rank the V2 waiting heap with the live ``running_batch`` (O(n)).

        Faithful to the original ``sort_by_priority(running_batch=live)`` run
        every step: the V2 score's compute terms scale with contention, so the
        ordering adapts to load (the insertion-time ``running_batch=0`` key is a
        cheap placeholder that this overwrites). ``n`` is bounded by
        ``max_num_seqs`` (≤512), so the heapify cost is negligible.
        """
        if not isinstance(self.waiting, _MarsHeapQueue) or not self.waiting:
            return
        running_batch, running_blocks = self._running_contention()
        self.waiting.rekey(lambda r: self._v2_score(r, running_batch, running_blocks))

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

        Each step, increment a wait counter for every request still in
        ``self.waiting``; once it exceeds ``starvation_threshold`` the request is
        marked *starving* (key ``-inf`` in the MARS queues) and moved to the
        front, where it stays until scheduled (the original's ``quantum`` is
        treated as "boosted until scheduled"). FCFS uses ``prepend_request``.
        """
        threshold = self.mars_config.starvation_threshold
        waiting_ids: set[str] = set()
        boosted = 0
        for req in list(self.waiting):
            rid = req.request_id
            waiting_ids.add(rid)
            if rid in self.mars_starving:
                continue
            count = self.mars_wait_counter.get(rid, 0) + 1
            self.mars_wait_counter[rid] = count
            if count > threshold:
                self.mars_starving.add(rid)
                self._mars_boost(req)
                boosted += 1
        # Drop bookkeeping for requests that left the waiting queue (scheduled).
        for rid in list(self.mars_wait_counter):
            if rid not in waiting_ids:
                self.mars_wait_counter.pop(rid, None)
                self.mars_starving.discard(rid)
        if boosted:
            logger.info("[MARS] starvation: boosted %d request(s) to front", boosted)

    def _mars_boost(self, request: Request) -> None:
        try:
            self.waiting.remove_request(request)
        except (ValueError, KeyError):
            return
        # FCFS deque -> appendleft (front); MARS heap -> re-add with key -inf
        # (the request is already in mars_starving at this point).
        self.waiting.prepend_request(request)

    def _mars_demote_paused(self) -> None:
        # Faithful to the original _schedule_chunk_and_fill: every step keep only
        # the single highest-waste preserved-paused request pinned and demote the
        # rest. Demoting each request right after it pauses (rather than waiting
        # for a memory wall) spreads the CPU-offload (PCIe) traffic across the
        # workload instead of bursting it in one step. Gated only on pending
        # admission demand -- there is no point freeing KV nothing is waiting for
        # (v1 reloads at resume, so a needless demote is a wasted host round-trip).
        if not self.waiting:
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
            for mode, waste in [self._demote_choice(rid, req)]
            if mode is not None
        ]
        if len(demotable) <= 1:
            return
        demotable.sort(key=lambda x: x[0])
        for waste, rid, req, mode in demotable[:-1]:  # all but the costliest
            self._demote(rid, req, mode, waste, usage)

    def _demote_choice(self, rid: str, request: Request):
        """``(mode, waste)`` to demote a preserved-paused request, from classify().

        InferCept ('I') demotes recompute-only; other policies apply the
        arrival-classified strategy (swap/recompute). A 'preserve'-classified
        request returns ``mode=None`` (it is never demoted), matching the
        original (a PRESERVE-strategy candidate falls through both branches).
        """
        st = self.mars_state.get(rid)
        letter = st.policy_letter if st is not None else self.mars_config.api_policy
        waste = st.arrival_waste if st is not None else 0.0
        strat = st.arrival_strategy if st is not None else "recompute"
        if letter == "I":
            return PauseMode.RECOMPUTE, waste
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
        if st is not None and st.pending_reset:
            # Blocks were released at the pause. Recompute from num_computed=0;
            # SWAP lets the prefix cache / CPU-offload host serve the prefix back,
            # while RECOMPUTE additionally bypasses the cache to force a true
            # recompute (prompt + kept output + injected API tokens).
            session.num_computed_tokens = 0
            if st.pending_skip_prefix:
                session.skip_reading_prefix_cache = True
            st.pending_reset = False
            st.pending_skip_prefix = False

    # --- cleanup -----------------------------------------------------------

    def _free_request(self, request: Request, *args, **kwargs):
        self.mars_state.pop(request.request_id, None)
        self.mars_paused_preserved.pop(request.request_id, None)
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
