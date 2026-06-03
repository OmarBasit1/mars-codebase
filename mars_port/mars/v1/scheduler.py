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
from mars.v1.queue import make_mars_queue
from mars.v1.solver import GUROBI_AVAILABLE, MarsSolver, SolverParams, split_to_mode

# Name under the "vllm" namespace so MARS logs inherit vLLM's log handler
# (vLLM configures a handler on the "vllm" logger only, propagate=False).
logger = init_logger("vllm.mars.scheduler")


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


class MARSScheduler(Scheduler):
    """vLLM v1 scheduler with MARS per-pause KV-cache policies."""

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
          * ``V`` -> cost-model choice (2-way, or 3-way when swap is available)
            via :meth:`_vulcan_decision`.
          * ``H``/``H-S``/``H-D``/``H-B`` -> ``api_exec_time`` threshold
            heuristics (swap arms degraded when swap is unavailable).
          * ``G``/``I`` -> their *static* pause mode (PRESERVE) here; the dynamic
            memory-pressure demotion that distinguishes Greedy / InferCept runs
            in :meth:`schedule` (``_mars_demote_paused``), where ``I``
            demotes recompute-only and ``G`` is swap-aware.
          * ``P``/``D``/``S`` -> direct mode (swap degraded when unavailable).
        """
        if policy_letter == "V":
            return self._vulcan_decision(request)
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
        if policy_letter in ("G", "I"):
            # Static pause mode is PRESERVE; the dynamic demotion that makes
            # Greedy / InferCept distinct runs in schedule() under memory pressure.
            return PauseMode.PRESERVE, " (static; demote under pressure)"
        # Direct P / D / S.
        mode = decide_pause_mode(
            policy_letter,
            swap_available=self._swap_available,
            swap_fallback=self.mars_config.swap_fallback,
        )
        return mode, ""

    def _vulcan_decision(self, request: Request):
        """2-way Vulcan: pick Preserve vs Recompute by the cost model.

        Uses the request's *actual* current length (known at the pause) and the
        current running-batch state as the recompute-contention context.
        """
        bs = self._block_size
        mp = MarsApiParams.from_sampling_params(request.sampling_params)
        api_exec_time = mp.predicted_api_exec_time if mp is not None else 1.0
        num_blocks = max(1, (request.num_tokens + bs - 1) // bs)
        before_api_tokens = num_blocks * bs
        # Competition for GPU during a recompute = the other running requests.
        running_batch = sum(
            max(1, r.num_tokens - r.num_computed_tokens) for r in self.running
        )
        running_blocks = sum(
            (r.num_computed_tokens + bs - 1) // bs for r in self.running
        )
        # Solver path (decision-only): solve the optimal KV split and apply the
        # dominant whole-request mode; fall back to the greedy choice on failure.
        if self.solver is not None:
            res = self.solver.solve_blocks(
                num_tokens=request.num_tokens,
                num_active_gpu_blocks=running_blocks,
                api_exec_time=api_exec_time,
                api_return_length=(mp.api_return_length if mp is not None else 0),
                arrival_time=request.arrival_time,
                now=time.time(),
                running_query_head=running_batch,
                running_query_tail=running_batch,
                swap_in_chunks_head=self.solver.free_swap,
                swap_in_chunks_tail=self.solver.free_swap,
            )
            if res is not None:
                c_s, c_d, n_e = res
                mode, _ = split_to_mode(
                    c_s, c_d, n_e, num_blocks,
                    swap_available=self._swap_available,
                    swap_fallback=self.mars_config.swap_fallback,
                )
                return mode, f" solver(c_s={c_s},c_d={c_d},n_e={n_e})"
        mode, wastes = self.cost_model.choose(
            api_exec_time=api_exec_time,
            before_api_tokens=before_api_tokens,
            num_blocks=num_blocks,
            running_batch=running_batch,
            running_blocks=running_blocks,
            swap_available=self._swap_available,
        )
        label = {
            PauseMode.PRESERVE: "w_p",
            PauseMode.RECOMPUTE: "w_d",
            PauseMode.SWAP: "w_s",
        }
        detail = " " + " ".join(f"{label[m]}={w:.4g}" for m, w in wastes.items())
        return mode, detail

    # --- dynamic memory-pressure demotion ---------------------------------

    def schedule(self):
        # Pre-pass before the base scheduler admits/preempts:
        #  1. starvation — boost long-waiting requests to the front;
        #  2. demotion — free the cheapest preserved-paused KV under pressure.
        if self.mars_config.starvation_avoidance:
            self._mars_starvation_pass()
        # chunk_fill (the original's switch for the demotion machinery) or the
        # explicit demote_under_pressure knob enables dynamic demotion.
        if (
            (self.mars_config.demote_under_pressure or self.mars_config.chunk_fill)
            and len(self.mars_paused_preserved) > 1
        ):
            self._mars_demote_paused()
        return super().schedule()

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
        # Rank candidates by demote waste; demote all but the costliest one
        # (faithful to the original: keep the single highest-waste request
        # preserved, demote the cheaper ones to reclaim memory).
        cands = []
        for rid, req in self.mars_paused_preserved.items():
            mode, waste = self._demote_choice(rid, req)
            cands.append((waste, rid, req, mode))
        if len(cands) <= 1:
            return
        cands.sort(key=lambda x: x[0])
        for waste, rid, req, mode in cands[:-1]:
            self._demote(rid, req, mode, waste, usage)

    def _demote_choice(self, rid: str, request: Request):
        """Pick the demote mode + waste for a preserved-paused request.

        InferCept ('I') demotes recompute-only (2-way); other policies pick the
        cheaper of swap/recompute when swap is available (3-way).
        """
        bs = self._block_size
        num_blocks = max(1, (request.num_tokens + bs - 1) // bs)
        running_batch = sum(
            max(1, r.num_tokens - r.num_computed_tokens) for r in self.running
        )
        running_blocks = sum(
            (r.num_computed_tokens + bs - 1) // bs for r in self.running
        )
        w_d = self.cost_model.discard_waste(num_blocks, running_batch, running_blocks)
        st = self.mars_state.get(rid)
        letter = st.policy_letter if st is not None else self.mars_config.api_policy
        if letter == "I" or not self._swap_available:
            return PauseMode.RECOMPUTE, w_d
        w_s = self.cost_model.swap_waste(num_blocks, running_batch, running_blocks)
        return (PauseMode.SWAP, w_s) if w_s < w_d else (PauseMode.RECOMPUTE, w_d)

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
