"""Engine-wide MARS configuration, passed to ``MARSScheduler``.

The original MARS put these knobs on ``EngineArgs``/``SchedulerConfig``. On
modern vLLM we keep them in a MARS-owned object and ship it to the (custom)
scheduler through vLLM's supported ``VllmConfig.additional_config`` channel —
no edits to vLLM's own config dataclasses.

Usage:
    cfg = MarsConfig(api_policy="D")
    LLM(model=..., scheduler_cls="mars.v1.scheduler.MARSScheduler",
        additional_config=cfg.to_additional_config())
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

ADDITIONAL_CONFIG_KEY = "mars"


@dataclass
class MarsConfig:
    """Engine-wide MARS knobs.

    Per-request ``MarsApiParams.api_policy`` (in ``extra_args``) overrides
    ``api_policy`` for that request.

    Attributes:
        api_policy: Default KV policy letter — ``P`` preserve, ``D`` discard/
            recompute, ``S`` swap, and the adaptive ``V``/``G``/``H``/``H-S``/
            ``H-D``/``H-B``/``I`` (adaptive policies are resolved in Phase 4+).
        policy_config: Waiting-queue ordering: ``fcfs`` / ``sjf`` / ``V2``
            (SJF wiring lands with Phase 4).
        swap_fallback: How swap-ish policies degrade while CPU offload is not
            configured (pre-Phase 6): ``recompute`` or ``preserve``.
        chunk_fill: Enable chunk-fill admission (token_budget shaping; Phase 4).
        chunk_size: Per-step token budget for chunk-fill (0 => engine default).
        heuristic_coef: Threshold for the ``H`` heuristic (Phase 5).
        starvation_avoidance/threshold/quantum: Anti-starvation (Phase 5).
        recompute_skip_prefix_cache: For the Recompute policy, force a true
            recompute on resume by bypassing the prefix cache (so the freed KV
            is not silently served back). Set False to allow opportunistic
            prefix-cache reuse.
    """

    api_policy: str = "P"
    policy_config: str = "fcfs"
    swap_fallback: str = "recompute"
    chunk_fill: bool = False
    chunk_size: int = 0
    heuristic_coef: float = 4.0
    starvation_avoidance: bool = False
    starvation_threshold: int = 0
    starvation_quantum: int = 0
    recompute_skip_prefix_cache: bool = True
    # Cost-model coefficients for the adaptive 'V' (Vulcan) policy. Defaults are
    # the original MARS values; re-tune per model/GPU via examples/calibrate_cost.py
    # (the forward-step time model is ~ (cost_a * batch_tokens + cost_c) ms, and
    # cost_max_ragged_batch is the tokens/step where compute saturates).
    cost_a: float = 0.0463
    cost_c: float = 10.0
    cost_max_ragged_batch: int = 384
    # Swap-waste coefficients (Phase 6, 3-way Vulcan). Original MARS values;
    # re-profile against the CPU-offload transfer path for real experiments.
    cost_swap_a1: float = 0.136
    cost_swap_a2: float = 0.181
    cost_swap_c: float = 22.5

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "MarsConfig":
        """Build from ``vllm_config.additional_config['mars']`` (defaults if absent)."""
        add = getattr(vllm_config, "additional_config", None) or {}
        raw = add.get(ADDITIONAL_CONFIG_KEY, {}) if isinstance(add, dict) else {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown MarsConfig keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_additional_config(
        self, additional_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return an ``additional_config`` dict with this config under ``'mars'``."""
        out = dict(additional_config) if additional_config else {}
        out[ADDITIONAL_CONFIG_KEY] = {f.name: getattr(self, f.name) for f in fields(self)}
        return out
