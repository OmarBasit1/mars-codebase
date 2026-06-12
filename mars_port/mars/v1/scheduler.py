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
    demotions: int = 0  # times this request was demoted (free KV under pressure)
    remaining_quantum: int = 0  # starvation-boost steps left (0 = not boosted)
    # Remaining API calls (seeded from api_max_calls, decremented per resume).
    # Drives the V2 score's ``api_complete`` branch after the final call, like
    # the original's per-call api_max_calls decrement in llm_engine.
    remaining_api_calls: int = 0


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
                # v1 async/write-through swap model: profiled reload latency
                # (s/tok) + the forward coeffs. swap_a1/a2/c are no longer used.
                per_token_swap_latency=self.mars_config.per_token_swap_latency,
                swap_a1=self.mars_config.cost_swap_a1,
                swap_a2=self.mars_config.cost_swap_a2,
                swap_c=self.mars_config.cost_swap_c,
            )
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
            self.cost_model.coeffs.a,
            self.cost_model.coeffs.c,
            self._block_size,
        )
        if mars_queue is not None:
            self.waiting = mars_queue
            self.skipped_waiting = make_mars_queue(
                self.mars_config.policy_config,
                self.mars_starving,
                self.mars_config.cost_max_ragged_batch,
                self.cost_model.coeffs.a,
                self.cost_model.coeffs.c,
                self._block_size,
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
        # Stall diagnostics (enable with env MARS_DIAG=1): a throttled per-second
        # heartbeat of the scheduler/memory state for debugging GPU-idle/CPU-busy
        # spins (schedule() looping without dispatching a GPU batch). Near-zero
        # overhead when disabled.
        self._mars_diag: bool = bool(os.environ.get("MARS_DIAG"))
        self._mars_diag_last: float = 0.0
        self._mars_diag_steps: int = 0   # schedule() calls since last heartbeat
        self._mars_reclaim_calls: int = 0  # reclaim invocations since last heartbeat
        self._mars_reclaim_freed: int = 0  # victims actually freed since last heartbeat
        self._mars_reclaim_earlyret: int = 0  # reclaim 'enough free' early-returns
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
        """Create MARS state + classify strategy/waste at arrival.

        Ports the original ``add_seq_group -> classify``: create per-request
        state, seed ``remaining_api_calls`` from the workload, then compute the
        KV strategy from the *predicted* length. The strategy is refreshed on
        every resume by :meth:`_classify_strategy` (faithful to the original
        ``resume_seq_group`` re-running ``classify()``).
        """
        letter = self._api_policy_for(request)
        st = _MarsReqState(policy_letter=letter)
        self.mars_state[request.request_id] = st
        mp = MarsApiParams.from_sampling_params(request.sampling_params)
        # Seed the remaining-call counter so the V2 ``api_complete`` branch can
        # activate after the final call (the original decrements api_max_calls
        # per completed call in llm_engine; here it lives on the request state).
        st.remaining_api_calls = mp.api_max_calls if mp is not None else 0
        self._classify_strategy(request, st)

    def _classify_strategy(self, request: Request, st: "_MarsReqState") -> None:
        """(Re)compute ``arrival_strategy``/``arrival_waste`` for ``st``.

        Ports the original ``classify()``: predict blocks from
        ``prompt_len + predicted_api_invoke_interval``, score preserve/recompute/
        swap (with the live running-batch contention), store the chosen mode. The
        result drives demotion (which mode, and the max-waste ranking) and the
        V2 ordering branch — NOT the pause itself (V always preserves). Called at
        arrival and again on each resume (refreshing the strategy with the
        then-current contention), matching the original's per-resume classify.
        """
        letter = st.policy_letter
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
        st.arrival_strategy = _MODE_TO_STRATEGY[mode]
        st.arrival_waste = wastes[mode]
        logger.info(
            "[MARS] classify req=%s policy=%s -> strategy=%s waste=%.4g (predicted)",
            request.request_id, letter, st.arrival_strategy, st.arrival_waste,
        )

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
        # For adaptive 'V', freeing KV at the pause IS an eager demotion to the
        # arrival-classified strategy (demote_eager) -- count it like a lazy demotion
        # so the per-request stats categorize it as swap/recompute (not preserve) and
        # it shows in demoted_reqs. Direct D/S letters are NOT counted here (the
        # parser keys those off the policy letter; they aren't "demoted").
        if st.policy_letter == "V":
            st.demotions += 1
            self.mars_demotions += 1
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
          * ``V`` -> PRESERVE at the pause by default, with the swap/recompute
            classified at arrival applied later by demotion (``_mars_demote_paused``
            / ``_reclaim_ondemand`` using the arrival-classified strategy). When
            ``demote_eager`` is set, a recompute/swap-classified request is instead
            freed AT THE PAUSE (like D/S); only 'preserve'-classified stays pinned.
          * ``P``/``D``/``S`` -> direct mode (swap degraded when unavailable).
        """
        if policy_letter == "V":
            st = self.mars_state.get(request.request_id)
            strat = st.arrival_strategy if st is not None else ""
            # Eager-drop: apply the arrival strategy now instead of preserving and
            # demoting lazily. Gated on demotion being enabled at all (--no-demote
            # forces pure preserve and wins).
            if (
                self.mars_config.demote_eager
                and self.mars_config.demote_paused
                and strat in ("recompute", "swap")
            ):
                mode = PauseMode.SWAP if strat == "swap" else PauseMode.RECOMPUTE
                mode = degrade_swap(
                    mode,
                    swap_available=self._swap_available,
                    swap_fallback=self.mars_config.swap_fallback,
                )
                return mode, f" (eager-drop; arrival_strategy={strat})"
            # Static PRESERVE at the pause; lazy demotion applies the arrival strategy.
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
        # Dynamic memory-pressure demotion (default on; --no-demote disables).
        # Decoupled from chunk_fill, which now controls only the per-step
        # token-budget cap. The on-demand mode skips this proactive pass and frees
        # KV lazily in _reclaim_blocks_for_admission instead.
        if (
            self.mars_config.demote_paused
            and not self.mars_config.demote_ondemand
            and len(self.mars_paused_preserved) > 1
        ):
            self._mars_demote_paused()
        out = super().schedule()
        self._mars_diag_heartbeat(out)
        return out

    def _mars_diag_heartbeat(self, out) -> None:
        """Throttled (1/s) scheduler-state heartbeat for stall debugging (MARS_DIAG=1).

        Captures the frozen state during a GPU-idle/CPU-busy spin:
          * ``steps`` -- schedule() calls in the last second; a huge value is the hot
            CPU spin (scheduler looping, dispatching nothing); a tiny value means the
            engine is blocked elsewhere (model exec / a stuck KV transfer).
          * ``sched_tok`` -- tokens dispatched this step (0 => GPU got no work).
          * running / waiting / skipped / paused_pre / resumed_pre counts.
          * ``wait_status`` -- status histogram across the waiting queues;
            ``WAITING_FOR_REMOTE_KVS`` = in-flight swap reloads holding GPU blocks.
          * kv usage / free blocks, and reclaim activity (calls/freed) in the second.
        """
        if not self._mars_diag:
            return
        self._mars_diag_steps += 1
        now = time.monotonic()
        if now - self._mars_diag_last < 1.0:
            return
        try:
            hist: dict[str, int] = {}
            for q in (self.waiting, self.skipped_waiting):
                it = q.iter_unsorted() if hasattr(q, "iter_unsorted") else iter(q)
                for r in it:
                    hist[r.status.name] = hist.get(r.status.name, 0) + 1
            free_blk = self.kv_cache_manager.block_pool.get_num_free_blocks()
            kvhold = len(self._mars_kv_holding_waiting())
            rp_wait = sum(1 for r in self.mars_resumed_preserved.values()
                          if r.status == RequestStatus.WAITING)
            rp_infl = sum(1 for r in self.mars_resumed_preserved.values()
                          if self._mars_request_inflight(r))
            logger.info(
                "[MARS][diag] dt=%.2fs steps=%d sched_tok=%s running=%d waiting=%d "
                "skipped=%d paused_pre=%d resumed_pre=%d(wait=%d infl=%d) kvhold=%d "
                "kv_usage=%.3f free_blk=%d reclaim(calls=%d freed=%d earlyret=%d) "
                "wait_status=%s",
                now - self._mars_diag_last, self._mars_diag_steps,
                getattr(out, "total_num_scheduled_tokens", "?"),
                len(self.running), len(self.waiting), len(self.skipped_waiting),
                len(self.mars_paused_preserved), len(self.mars_resumed_preserved),
                rp_wait, rp_infl, kvhold,
                self.kv_cache_manager.usage, free_blk,
                self._mars_reclaim_calls, self._mars_reclaim_freed,
                self._mars_reclaim_earlyret, hist,
            )
        except Exception as e:  # diagnostics must never break the run
            logger.warning("[MARS][diag] heartbeat error: %s", e)
        finally:
            self._mars_diag_last = now
            self._mars_diag_steps = 0
            self._mars_reclaim_calls = 0
            self._mars_reclaim_freed = 0
            self._mars_reclaim_earlyret = 0

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
        """V2 memory-time of computing ``num_blocks`` (token-seconds).

        Uses the **profiled** forward coefficients (``cost_a``/``cost_c``), the same
        ones the cost model uses -- so the V2 ordering score and the cost model are
        on one calibrated scale (the original hand-tuned ``a=0.1, c=10`` are gone).
        """
        co = self.cost_model.coeffs
        c_h = max(co.max_ragged_batch - running_batch, 1)
        n = max((co.block_size * num_blocks + c_h - 1) // c_h, 1)
        f_s = (co.a * co.max_ragged_batch + co.c) / 1000.0
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
        st = self.mars_state.get(request.request_id)
        # api_complete tracks the original's per-call api_max_calls decrement:
        # use the live remaining-call counter on the request state (falling back
        # to the static param when no state exists yet).
        api_complete = (
            st.remaining_api_calls == 0 if st is not None else mp.api_max_calls == 0
        )
        strat = st.arrival_strategy if st is not None and st.arrival_strategy else "preserve"
        if strat == "recompute":
            before_after = self._mem_time(before_blocks + after_blocks, running_batch)
            return before_after if api_complete else before + 0.0 + before_after
        if strat == "swap":
            # v1 write-through swap: swap-OUT is free (the original's first `swap`
            # term is gone), and the swap-IN reload is priced by the SAME profiled
            # cost_model.swap_waste used for classify/demotion -- so V2 ordering and
            # the cost model agree (now both on the calibrated _mem_time scale).
            reload_s = self.cost_model.swap_waste(
                before_blocks, running_batch, running_blocks
            )
            api = api_exec_time * 0.1  # KV off-GPU during the wait: tiny residual
            return (reload_s + after) if api_complete else before + api + reload_s + after
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
        # Faithful to the original _schedule_chunk_and_fill: max_waste is taken
        # over ALL paused-preserve requests -- including 'preserve'-classified
        # ones (which are never demoted) and in-flight ones (still pinned) -- and
        # every demotable request whose waste is strictly below max_waste is
        # demoted to its arrival-classified strategy. Modes and wastes come from
        # classify() (predicted), NOT a live recompute. So when a
        # 'preserve'-classified request holds the global max, ALL recompute/swap
        # candidates demote (none kept), matching the original; ties at max_waste
        # stay pinned. The reclaim hook (_reclaim_blocks_for_admission) is the
        # hard-pressure backstop, so demoting everything below max never deadlocks.
        all_wastes = [
            self._demote_choice(rid, req)[1]
            for rid, req in self.mars_paused_preserved.items()
        ]
        if not all_wastes:
            return
        max_waste = max(all_wastes)
        for rid, req in list(self.mars_paused_preserved.items()):
            if self._mars_request_inflight(req):
                continue  # async-safety: KV still referenced by an un-retired batch
            mode, waste = self._demote_choice(rid, req)
            if mode is not None and waste < max_waste:
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

    def _mars_kv_holding_waiting(self) -> list[tuple[str, Request]]:
        """Return all WAITING requests that currently hold GPU KV blocks.

        Covers two classes:
        1. ``mars_resumed_preserved``: resumed with PRESERVE KV intact
           (``pending_reset=False`` at resume).
        2. Requests in ``skipped_waiting`` with ``num_computed_tokens > 0``:
           went through ``WAITING_FOR_REMOTE_KVS`` (async KV reload allocated
           blocks) and returned to WAITING with those blocks resident.

        Both classes hold GPU memory that can be freed (at the cost of a future
        reload/recompute) to unblock admission.
        """
        seen: set[str] = set()
        result: list[tuple[str, Request]] = []
        for rid, req in self.mars_resumed_preserved.items():
            if req.status == RequestStatus.WAITING and not self._mars_request_inflight(req):
                seen.add(rid)
                result.append((rid, req))
        if self._mars_queues and self.skipped_waiting:
            for req in self.skipped_waiting.iter_unsorted():
                rid = req.request_id
                st = self.mars_state.get(rid)
                # skip if already demoted (pending_reset=True means blocks freed)
                if st is not None and st.pending_reset:
                    continue
                if rid not in seen and not self._mars_request_inflight(req):
                    if (req.status == RequestStatus.WAITING
                            and req.num_computed_tokens > 0):
                        result.append((rid, req))
                    elif req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        # In-progress async KV loads also hold GPU blocks; include
                        # them so the reclaim can cancel the load and free memory.
                        result.append((rid, req))
        return result

    def _reclaim_blocks_for_admission(
        self, request: Request, num_new_tokens: int
    ) -> bool:
        """Passive-discard: free the lowest-priority KV-holding waiting request.

        Called by the base when allocate_slots returns None for a waiting
        request.  Picks the victim with the smallest V2 waste (least worth
        keeping) from all WAITING requests that hold GPU KV blocks:
          - ``mars_resumed_preserved``: never-demoted PRESERVE holders.
          - WAITING requests in ``skipped_waiting`` with ``num_computed_tokens>0``:
            went through ``WAITING_FOR_REMOTE_KVS`` and hold reloaded blocks.
        Frees the victim's KV and returns True so the base retries admission.
        Returns False when no eligible victim exists (base falls back to break).

        Mirrors the original's ``passive_discard_by_order`` over
        ``combined_targets`` — the safety net that made the original immune to
        memory-over-commit deadlocks.
        """
        if self._mars_diag:
            self._mars_reclaim_calls += 1
        if self.mars_config.demote_paused and self.mars_config.demote_ondemand:
            return self._reclaim_ondemand(request, num_new_tokens)
        # Only intervene when running=0 (true deadlock: vLLM's native
        # running-queue preemption has no victims). When running > 0, native
        # preemption already handles memory pressure; intervening causes a
        # free→re-admit→re-hold churn cycle that makes things worse.
        if len(self.running) > 0:
            return False
        admit_id = request.request_id
        holders = [
            (rid, req)
            for rid, req in self._mars_kv_holding_waiting()
            if rid != admit_id
        ]
        if not holders:
            # Emergency fallback: _mars_request_inflight excluded every
            # candidate (phantom num_output_placeholders>0 from a prior async
            # step with running=0 — those outputs will never arrive, so the
            # guard is a false positive).  Do a second pass without the inflight
            # filter; safe because running=0 means no model batch is in flight.
            seen: set[str] = set()
            for rid, req in self.mars_resumed_preserved.items():
                if req.status == RequestStatus.WAITING and rid != admit_id:
                    seen.add(rid)
                    holders.append((rid, req))
            if self._mars_queues and self.skipped_waiting:
                for req in self.skipped_waiting.iter_unsorted():
                    rid = req.request_id
                    if rid in seen or rid == admit_id:
                        continue
                    st = self.mars_state.get(rid)
                    if st is not None and st.pending_reset:
                        continue
                    if (req.status == RequestStatus.WAITING
                            and req.num_computed_tokens > 0):
                        holders.append((rid, req))
            if not holders:
                return False
        usage = self.kv_cache_manager.usage
        # Evict the LOWEST priority (highest V2 score = least worth keeping).
        candidates = [(self._v2_score(req, 0, 0), rid, req) for rid, req in holders]
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
        # Return False: defer this request to the next schedule() step with
        # the freed memory available. Returning True (retry same step) causes a
        # free→re-admit→re-hold churn cycle at high load.
        return False

    def _reclaim_ondemand(self, request: Request, num_new_tokens: int) -> bool:
        """Lazy, minimal demotion: free only enough preserved-paused KV to admit.

        On-demand counterpart to the proactive ``_mars_demote_paused`` (used when
        ``demote_ondemand``). Fires when a request fails to allocate GPU blocks.
        Frees the MINIMUM number of preserved API-waiting requests -- lowest
        arrival-waste first, so the costliest-to-redo stay pinned longest --
        needed to cover this (chunked) admission, then defers one step (returns
        False) so the freed memory admits the request next ``schedule()``.
        Demoted requests reload only at their API resume (``pending_reset``),
        never proactively. Unlike the running==0 backstop this runs under any
        running count: parked ``WAITING_FOR_STREAMING_REQ`` holders are invisible
        to vLLM's native running-queue preemption, so only MARS can reclaim them.

        Two tiers of victim (both from ``mars_paused_preserved``):
          * tier 1 -- demotable (swap/recompute-classified): freed by the cost
            model, cheapest-waste first.
          * tier 2 (emergency) -- preserve-classified pins (``mode=None``), which
            are normally never freed, are FORCE-FREED here as a last resort when
            tier 1 can't cover the admission, each via the cheaper of recompute/swap
            (``_forced_free_mode``). So a memory-blocked waiting
            chunk never stalls on idle preserved API KV -- instead of waiting for
            running to drain to 0 (the running==0 backstop). Tier 2 stays empty in
            lazy mode whenever swap/recompute victims remain (the documented lazy
            behavior is unchanged); it is the ONLY tier in eager mode, where every
            paused request is preserve-classified.
        """
        admit_id = request.request_id
        bs = self._block_size
        blocks_needed = (num_new_tokens + bs - 1) // bs
        pool = self.kv_cache_manager.block_pool
        if pool.get_num_free_blocks() >= blocks_needed and self.running:
            # Enough raw blocks free and something is running -> the alloc failed for
            # another reason (e.g. a KV reload needs its whole context, which
            # blocks_needed under-counts); let native preemption handle it. At
            # running==0 do NOT bail here: that is a true deadlock and tier-3 must run.
            if self._mars_diag:
                self._mars_reclaim_earlyret += 1
            return False
        # Eligible holders: preserved API-waiting requests, not in-flight. Split by
        # arrival strategy into demotable victims (tier 1) and preserve-classified
        # pins (tier 2 emergency, force-freed via the cheaper of recompute/swap).
        victims = []
        preserve_pins = []
        for rid, req in self.mars_paused_preserved.items():
            if rid == admit_id or self._mars_request_inflight(req):
                continue
            mode, waste = self._demote_choice(rid, req)
            if mode is None:  # 'preserve'-classified -> tier-2 last resort only
                preserve_pins.append((waste, rid, req))
            else:
                victims.append((waste, rid, req, mode))
        # Nothing in the paused-preserve pool: bail only if something is running
        # (native preemption can handle it). At running==0 fall through to the
        # tier-3 deadlock backstop, which frees the resumed/skipped KV-holders.
        if not victims and not preserve_pins and self.running:
            return False
        usage = self.kv_cache_manager.usage
        freed = 0
        # Tier 1: demote by arrival strategy; cheapest-to-lose first so the
        # costliest-to-redo stay pinned longest.
        for waste, rid, req, mode in sorted(victims, key=lambda x: x[0]):
            if pool.get_num_free_blocks() >= blocks_needed:
                break  # minimum reached
            self._demote(rid, req, mode, waste, usage)
            self.mars_reclaims += 1
            freed += 1
        # Tier 2 (emergency): preserve-classified pins must be freed too if tier 1
        # didn't cover the chunk. Preserve was the classified-cheapest mode, but
        # forced to free now we pick the cheaper of RECOMPUTE vs SWAP per pin on live
        # contention (the v1 write-through cache already mirrored the KV to host, so
        # swap-OUT is free and only the reload is priced -- often cheaper than a full
        # recompute). Cheapest-to-redo first.
        if pool.get_num_free_blocks() < blocks_needed and preserve_pins:
            rb, rblk = self._running_contention()
            scored = []
            for _, rid, req in preserve_pins:
                mode, w = self._forced_free_mode(req, rb, rblk)
                scored.append((w, mode, rid, req))
            scored.sort(key=lambda x: x[0])
            for w, mode, rid, req in scored:
                if pool.get_num_free_blocks() >= blocks_needed:
                    break
                self._demote(rid, req, mode, w, usage)
                self.mars_reclaims += 1
                freed += 1
        # Tier 3 (deadlock backstop): nothing running == a true memory deadlock
        # (vLLM's native running-queue preemption has no victims). The reclaim hook
        # only fires when admission already failed, so at running==0 we MUST free GPU
        # blocks regardless of the `blocks_needed` estimate -- which UNDER-counts for
        # KV-reload admissions (a resumed SWAP request reallocates its whole KV, not
        # just num_new_tokens), the exact case that otherwise early-returns "enough
        # free" forever (kvhold>0 but freed=0, the engine spinning at running==0).
        # Free the resumed/skipped KV-holders via RECOMPUTE -- a definitive GPU free
        # with no connector/reload dependency (the reload path is often the wedged
        # one) -- cheapest-to-redo first, up to a batch-step of headroom so a real
        # prefill/reload fits and running recovers, handing back to normal scheduling.
        if not self.running:
            rb, rblk = self._running_contention()
            bs2 = self._block_size
            scored = []
            for rid, req in self._mars_kv_holding_waiting():
                if rid == admit_id:
                    continue
                nb = max(1, (req.num_computed_tokens + bs2 - 1) // bs2)
                scored.append((self.cost_model.discard_waste(nb, rb, rblk), rid, req))
            scored.sort(key=lambda x: x[0])
            target = max(blocks_needed,
                         self.mars_config.cost_max_ragged_batch // self._block_size)
            for w, rid, req in scored:
                if pool.get_num_free_blocks() >= target:
                    break
                self._demote(rid, req, PauseMode.RECOMPUTE, w, usage)
                self.mars_reclaims += 1
                freed += 1
        if freed:
            self._mars_reclaim_freed += freed
            logger.info(
                "[MARS] ondemand reclaim: freed %d preserved-paused req(s) "
                "(need %d blk) to admit req=%s", freed, blocks_needed, admit_id,
            )
        return False  # defer one step; next schedule() admits with freed memory

    def _demote_choice(self, rid: str, request: Request):
        """``(mode, waste)`` to demote a preserved-paused request, from classify().

        Applies the arrival-classified strategy (swap/recompute) for ``V`` and
        direct policies. A 'preserve'-classified request returns ``mode=None``
        (never demoted, stays pinned).
        """
        st = self.mars_state.get(rid)
        waste = st.arrival_waste if st is not None else 0.0
        strat = st.arrival_strategy if st is not None else "recompute"
        if strat == "swap" and self._swap_available:
            return PauseMode.SWAP, waste
        if strat == "recompute":
            return PauseMode.RECOMPUTE, waste
        return None, waste  # 'preserve' (or 'swap' w/o connector): stay pinned

    def _forced_free_mode(self, request, running_batch, running_blocks):
        """Cheaper of RECOMPUTE vs SWAP for a FORCED free of a preserve-classified
        pin under memory pressure (tier-2 reclaim in :meth:`_reclaim_ondemand`).

        Preserve was the classified-cheapest mode, but admission pressure forces the
        KV to be freed anyway -- so choose the cheaper of the two free-the-memory
        modes on live contention. The v1 write-through CPU-offload cache already
        mirrored this request's KV to host (swap-OUT is a sunk, policy-independent
        cost), so SWAP prices only the async reload, which frequently beats a full
        RECOMPUTE. SWAP wins ties (matching the cost model's swap>recompute order)
        and is only eligible when a CPU-offload connector is available.
        """
        bs = self._block_size
        num_blocks = max(1, (request.num_computed_tokens + bs - 1) // bs)
        w_d = self.cost_model.discard_waste(num_blocks, running_batch, running_blocks)
        if self._swap_available:
            w_s = self.cost_model.swap_waste(num_blocks, running_batch, running_blocks)
            if w_s <= w_d:
                return PauseMode.SWAP, w_s
        return PauseMode.RECOMPUTE, w_d

    def _demote(self, rid, request, mode, waste, usage) -> None:
        # Capture the status BEFORE any branch mutates it: it decides whether the
        # num_computed_tokens reset can be deferred to a future resume-fold or
        # must be applied now (see the already-resumed block below).
        entry_status = request.status
        # Free the preserved-paused or WFRKV KV now.
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # Cancel the in-progress async KV load: tell the connector to drop
            # its in-flight load state (cancels the PCIe copy, releases CPU pin),
            # free the GPU blocks, reset num_computed_tokens=0, and transition back
            # to plain WAITING so the request re-enters normal admission next step.
            # The hardened _update_from_kv_xfer_finished (mars-port vLLM edit)
            # safely ignores any stale completion that arrives after this.
            if self.connector is not None and hasattr(
                self.connector, "cancel_load_for_request"
            ):
                self.connector.cancel_load_for_request(request)
            self.kv_cache_manager.free(request)
            request.num_computed_tokens = 0
            if hasattr(self, "finished_recving_kv_req_ids"):
                self.finished_recving_kv_req_ids.discard(rid)
            # Prevent re-entering WAITING_FOR_REMOTE_KVS on the very next
            # admission: the CPU blocks are still in the connector's LRU cache
            # and would be found immediately without this flag.  The flag is
            # cleared by _update_request_as_session on the next API resume.
            request.skip_reading_prefix_cache = True
            # Transition back to WAITING so the request is schedulable next step.
            request.status = RequestStatus.WAITING
        else:
            self.kv_cache_manager.free(request)
        st = self.mars_state.get(rid)
        if st is None:
            st = _MarsReqState(policy_letter=self._api_policy_for(request))
            self.mars_state[rid] = st
        st.pending_reset = True
        if mode is PauseMode.SWAP:
            st.pending_skip_prefix = False
            self.mars_swap_count += 1
        else:
            st.pending_skip_prefix = self.mars_config.recompute_skip_prefix_cache
            self.mars_recompute_count += 1
        # An already-resumed (plain WAITING) victim -- e.g. one reclaimed from
        # mars_resumed_preserved by _reclaim_blocks_for_admission -- is a
        # last-segment request with no further API pause, so the deferred reset
        # in _update_request_as_session would NEVER fire. Apply it now (mirroring
        # the WFRKV branch): otherwise the freed KV is left with a stale
        # num_computed_tokens and the next admission assumes the prefix is still
        # resident -> corrupted output when prefix caching is off (SWAP is masked
        # by the connector/cache, RECOMPUTE is not). Parked
        # WAITING_FOR_STREAMING_REQ requests keep the deferral: their resume hook
        # applies it after folding in the injected API tokens.
        # WAITING_FOR_REMOTE_KVS already reset inline in its branch above.
        if entry_status == RequestStatus.WAITING:
            request.num_computed_tokens = 0
            request.skip_reading_prefix_cache = st.pending_skip_prefix
            st.pending_reset = False
            st.pending_skip_prefix = False
        self.mars_paused_preserved.pop(rid, None)
        self.mars_resumed_preserved.pop(rid, None)
        st.demotions += 1
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
        rid = session.request_id
        st = self.mars_state.get(rid)
        super()._update_request_as_session(session, update)
        # Resuming -> no longer a preserved-paused candidate.
        self.mars_paused_preserved.pop(rid, None)
        if st is not None and not st.pending_reset:
            # KV blocks were NOT freed at pause (PRESERVE mode). The request
            # resumes holding its GPU blocks; track it so passive_discard can
            # reclaim them if memory is exhausted during admission.
            self.mars_resumed_preserved[rid] = session
        if st is not None and st.pending_reset:
            # Blocks were released at the pause. Recompute from num_computed=0;
            # SWAP lets the prefix cache / CPU-offload host serve the prefix back,
            # while RECOMPUTE additionally bypasses the cache to force a true
            # recompute (prompt + kept output + injected API tokens).
            is_swap = not st.pending_skip_prefix  # SWAP=False/RECOMPUTE=True
            session.num_computed_tokens = 0
            # Explicitly set/clear skip_reading_prefix_cache so any value set
            # by a prior WFRKV reclaim is always overridden at resume time.
            session.skip_reading_prefix_cache = st.pending_skip_prefix
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
        if st is not None:
            # One API call just completed (faithful to the original's per-call
            # api_max_calls decrement) -> the V2 score's api_complete branch can
            # engage for the final segment. Floored at 0.
            if st.remaining_api_calls > 0:
                st.remaining_api_calls -= 1
            # Refresh the KV strategy with the now-current contention, matching
            # the original resume_seq_group re-running classify() each resume.
            self._classify_strategy(session, st)

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
                    "demotions": st.demotions,
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
