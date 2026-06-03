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

# Adaptive policy whose per-pause choice is made by the cost model (classify()
# at arrival). Listed here for validation/reference.
ADAPTIVE_POLICIES = frozenset({"V"})


def decide_pause_mode(
    api_policy: str,
    *,
    swap_available: bool,
    swap_fallback: str = "recompute",
) -> PauseMode:
    """Resolve a MARS ``api_policy`` letter to a concrete :class:`PauseMode`.

    Resolves the direct policies ``P``/``D`` (and ``S`` degraded). The adaptive
    ``V`` policy defaults to PRESERVE here; the demotion pass applies the
    arrival-classified strategy each step.

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
        # Adaptive policy (V): always preserve at the pause.
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
