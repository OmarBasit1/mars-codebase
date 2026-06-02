"""MARS KV-cache pause policies and the policy -> mode mapping.

When a request pauses for an API call, MARS decides what to do with its KV cache
during the wait. On modern vLLM (v1) the concrete behaviors are:

  * PRESERVE  -- keep the request's KV blocks resident on GPU (default native
                 behavior for a parked resumable request).
  * RECOMPUTE -- free the KV blocks now (memory available during the API wait)
                 and recompute them on resume.
  * SWAP      -- offload the KV to host memory via a CPU-offload KV connector
                 (Phase 6). Until that connector is configured, SWAP degrades to
                 RECOMPUTE or PRESERVE per ``MarsConfig.swap_fallback``.
"""

from __future__ import annotations

import enum


class PauseMode(enum.Enum):
    PRESERVE = "preserve"
    RECOMPUTE = "recompute"
    SWAP = "swap"


# Direct (non-adaptive) MARS policy letters.
_DIRECT: dict[str, PauseMode] = {
    "P": PauseMode.PRESERVE,
    "D": PauseMode.RECOMPUTE,
    "S": PauseMode.SWAP,
}

# Adaptive / heuristic policies whose per-pause choice is made by the cost model
# (Phase 4) or heuristics (Phase 5). Listed here for validation/reference.
ADAPTIVE_POLICIES = frozenset({"V", "G", "H", "H-S", "H-D", "H-B", "I"})


def decide_pause_mode(
    api_policy: str,
    *,
    swap_available: bool,
    swap_fallback: str = "recompute",
) -> PauseMode:
    """Resolve a MARS ``api_policy`` letter to a concrete :class:`PauseMode`.

    Phase 3 resolves the direct policies ``P``/``D`` (and ``S`` degraded). The
    adaptive policies default to PRESERVE here so the engine runs end-to-end;
    Phase 4+ overrides this decision with the cost model / solver.

    Args:
        api_policy: The MARS policy letter (per-request or engine default).
        swap_available: Whether a CPU-offload KV connector is configured.
        swap_fallback: ``"recompute"`` or ``"preserve"`` when swap is requested
            but unavailable.

    Returns:
        The concrete pause mode to apply.
    """
    mode = _DIRECT.get(api_policy)
    if mode is None:
        # Adaptive policy: resolved later; safe default keeps generation correct.
        return PauseMode.PRESERVE
    return degrade_swap(mode, swap_available=swap_available, swap_fallback=swap_fallback)


def degrade_swap(
    mode: PauseMode, *, swap_available: bool, swap_fallback: str = "recompute"
) -> PauseMode:
    """Degrade a SWAP decision to the fallback when CPU offload isn't configured."""
    if mode is PauseMode.SWAP and not swap_available:
        return (
            PauseMode.RECOMPUTE if swap_fallback == "recompute" else PauseMode.PRESERVE
        )
    return mode


# Threshold heuristic baselines (decide per-pause on api_exec_time).
THRESHOLD_POLICIES = frozenset({"H", "H-S", "H-D", "H-B"})


def decide_threshold_mode(
    policy_letter: str, *, api_exec_time: float, heuristic_coef: float
) -> PauseMode:
    """The MARS threshold heuristics (ported verbatim from scheduler_v2.py).

    May return SWAP; the caller applies :func:`degrade_swap` when CPU offload is
    unavailable. ``H`` uses the configurable ``heuristic_coef``; ``H-S``/``H-D``/
    ``H-B`` use the original fixed thresholds (4 s and 7 s).
    """
    if policy_letter == "H":
        return PauseMode.SWAP if api_exec_time >= heuristic_coef else PauseMode.PRESERVE
    if policy_letter == "H-S":
        return PauseMode.SWAP if api_exec_time >= 4 else PauseMode.PRESERVE
    if policy_letter == "H-D":
        return PauseMode.RECOMPUTE if api_exec_time >= 7 else PauseMode.PRESERVE
    if policy_letter == "H-B":
        if api_exec_time < 4:
            return PauseMode.PRESERVE
        if api_exec_time < 7:
            return PauseMode.SWAP
        return PauseMode.RECOMPUTE
    raise ValueError(f"not a threshold heuristic policy: {policy_letter}")
