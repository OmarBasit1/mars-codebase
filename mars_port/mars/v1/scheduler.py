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

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from mars.config import MarsConfig
from mars.cost_model import CostModel, CostModelCoeffs
from mars.params import MarsApiParams
from mars.policies import PauseMode, decide_pause_mode

# Name under the "vllm" namespace so MARS logs inherit vLLM's log handler
# (vLLM configures a handler on the "vllm" logger only, propagate=False).
logger = init_logger("vllm.mars.scheduler")


@dataclass
class _MarsReqState:
    """Per-request MARS bookkeeping (replaces old SequenceData KV-region fields)."""

    policy_letter: str
    pauses: int = 0
    # Set when blocks were freed at a pause (RECOMPUTE); triggers a
    # num_computed_tokens reset after the next resume-fold.
    pending_recompute_reset: bool = False


class MARSScheduler(Scheduler):
    """vLLM v1 scheduler with MARS per-pause KV-cache policies."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mars_config = MarsConfig.from_vllm_config(self.vllm_config)
        self.mars_state: dict[str, _MarsReqState] = {}
        # Swap is only real once a CPU-offload KV connector is configured (Phase 6).
        self._swap_available = self.connector is not None
        self._block_size = getattr(self.cache_config, "block_size", None) or 16
        # Cost model for the adaptive 'V' (Vulcan) policy.
        self.cost_model = CostModel(
            CostModelCoeffs(
                a=self.mars_config.cost_a,
                c=self.mars_config.cost_c,
                max_ragged_batch=self.mars_config.cost_max_ragged_batch,
                block_size=self._block_size,
            )
        )
        # chunk-fill (simplified): cap the per-step token budget so paused
        # (preserved) KV plus new work fit. Full dynamic chunk-fill is Phase 6.
        if self.mars_config.chunk_fill and self.mars_config.chunk_size > 0:
            self.max_num_scheduled_tokens = min(
                self.max_num_scheduled_tokens, self.mars_config.chunk_size
            )
        # Observability counters (read by tests / logged).
        self.mars_total_pauses = 0
        self.mars_preserve_count = 0
        self.mars_recompute_count = 0
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
        if mode is PauseMode.RECOMPUTE:
            # Free the paused request's KV now (idempotent: req_to_blocks.pop).
            # num_computed_tokens is left intact so the resume-fold can still
            # locate the kept output tokens; it is reset afterwards.
            self.kv_cache_manager.free(request)
            st.pending_recompute_reset = True
            self.mars_recompute_count += 1
            logger.info(
                "[MARS] pause req=%s policy=%s mode=recompute (freed KV)%s",
                request.request_id,
                st.policy_letter,
                detail,
            )
        else:
            # PRESERVE: keep blocks resident; nothing to do.
            self.mars_preserve_count += 1
            logger.info(
                "[MARS] pause req=%s policy=%s mode=preserve (kept KV)%s",
                request.request_id,
                st.policy_letter,
                detail,
            )

    def _decide_pause_mode(self, request: Request, policy_letter: str):
        """Resolve the concrete pause mode (and a log detail string).

        Direct policies (P/D/S) map straight through; the adaptive 'V' (Vulcan)
        policy chooses Preserve vs Recompute via the 2-way cost model.
        """
        if policy_letter == "V":
            return self._vulcan_decision(request)
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
        mode, w_p, w_d = self.cost_model.choose_2way(
            api_exec_time=api_exec_time,
            before_api_tokens=before_api_tokens,
            num_blocks=num_blocks,
            running_batch=running_batch,
            running_blocks=running_blocks,
        )
        return mode, f" w_p={w_p:.4g} w_d={w_d:.4g}"

    # --- resume hook -------------------------------------------------------

    def _update_request_as_session(self, session: Request, update) -> None:
        super()._update_request_as_session(session, update)
        st = self.mars_state.get(session.request_id)
        if st is not None and st.pending_recompute_reset:
            # Blocks were released at the pause -> recompute the whole folded
            # prompt (prompt + kept output + injected API tokens).
            session.num_computed_tokens = 0
            if self.mars_config.recompute_skip_prefix_cache:
                # Force a true recompute rather than silently re-reading the KV
                # that may still be sitting in the prefix-cache pool.
                session.skip_reading_prefix_cache = True
            st.pending_recompute_reset = False

    # --- cleanup -----------------------------------------------------------

    def _free_request(self, request: Request, *args, **kwargs):
        self.mars_state.pop(request.request_id, None)
        return super()._free_request(request, *args, **kwargs)
