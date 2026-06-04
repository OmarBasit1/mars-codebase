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

# Generic fallback cost coefficients (original MARS 6B reference values).
_GENERIC_COST_DEFAULTS: dict[str, float] = {
    "cost_a": 0.0463,
    "cost_c": 10.0,
    "cost_swap_a1": 0.136,
    "solver_per_token_swap_latency": 4e-5,
    "solver_poly_a": 1.3e-5,
    "solver_poly_b": 0.328,
    "solver_poly_c": 24.1,
}

# Per-GPU calibrated overrides.  Matched as substrings of torch device name.
_GPU_PROFILES: dict[str, dict[str, float]] = {
    "A40": {
            "cost_a": 0.002395,
            "cost_c": 55.122,
            "cost_swap_a1": 84.9189,
            "solver_per_token_swap_latency": 8.492e-05,  # s/tok
            "solver_poly_a": 1.224e-8,
            "solver_poly_b": 0.0022,
            "solver_poly_c": 55.236,
        },
}


def _detected_cost_profile() -> dict[str, float]:
    """Return the GPU-specific cost profile for device 0, or {} if unknown."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            for key, profile in _GPU_PROFILES.items():
                if key in name:
                    return profile
    except Exception:
        pass
    return {}


@dataclass
class MarsConfig:
    """Engine-wide MARS knobs.

    Per-request ``MarsApiParams.api_policy`` (in ``extra_args``) overrides
    ``api_policy`` for that request.

    Attributes:
        api_policy: Default KV policy letter — ``P`` preserve, ``D`` discard/
            recompute, ``S`` swap, or the adaptive ``V`` (Vulcan: classify at
            arrival, always preserve at pause, demote by arrival strategy each step).
        policy_config: Waiting-queue ordering: ``fcfs`` / ``sjf`` / ``V2``
            (cost-based memory-time, re-ranked every step with live running_batch).
        swap_fallback: How swap-ish policies degrade while CPU offload is not
            configured: ``recompute`` or ``preserve``.
        chunk_fill: Enable chunk-fill admission (token_budget shaping).
        chunk_size: Per-step token budget for chunk-fill (0 => engine default).
        starvation_avoidance/threshold/quantum: Anti-starvation.
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
    starvation_avoidance: bool = False
    starvation_threshold: int = 0
    starvation_quantum: int = 0
    recompute_skip_prefix_cache: bool = True
    # Cost-model coefficients for the adaptive 'V' (Vulcan) policy. Defaults are
    # the original MARS values; re-tune per model/GPU via examples/calibrate_cost.py
    # (the forward-step time model is ~ (cost_a * batch_tokens + cost_c) ms, and
    # cost_max_ragged_batch is the tokens/step where compute saturates).
    # None => auto-filled by __post_init__ from the detected GPU profile (cost_a/c/
    # swap_a1/etc.) or from vllm's scheduler_config.max_num_batched_tokens
    # (cost_max_ragged_batch) at scheduler init time.
    cost_a: float | None = None
    cost_c: float | None = None
    cost_max_ragged_batch: int | None = None
    # Swap-waste coefficients (Phase 6, 3-way Vulcan). Original MARS values;
    # re-profile against the CPU-offload transfer path for real experiments.
    cost_swap_a1: float | None = None
    cost_swap_a2: float = 0.181
    cost_swap_c: float = 22.5
    # Dynamic demotion (Greedy / InferCept, and V). Faithful to the original
    # _schedule_chunk_and_fill: every step, keep only the single highest-waste
    # preserved-paused request pinned and demote the rest (free their KV -> swap
    # / recompute) so the freed memory can admit waiting work. Demoting each
    # request right after it pauses spreads the CPU-offload (PCIe) traffic across
    # the workload instead of bursting it at a memory wall. Pure 'P' is never
    # demoted. Gated only on pending admission demand (requests waiting).
    demote_under_pressure: bool = True
    # Optional KV-usage floor for demotion. 0 (default) => faithful original:
    # demote whenever work is waiting, regardless of usage. >0 => only demote
    # once usage exceeds this fraction (re-enables the old pressure gate).
    demote_pressure_threshold: float = 0.0
    # Gurobi solver (decision-only) for the 'V' policy: per pause, solve the
    # optimal KV split and apply the dominant WHOLE-request mode. Coefficients
    # are re-calibrated via examples/calibrate_cost.py (6B reference values from
    # the original 6B_bench.sh in comments). See COMPARISON.md for the
    # partial-split -> whole-request discrepancy.
    use_solver: bool = False
    solver_target: float = 1500.0  # SLA target throughput (tokens/s)
    solver_per_token_swap_latency: float | None = None
    solver_poly_a: float | None = None  # forward time = (a*x^2 + b*x + c)/1000 ms
    solver_poly_b: float | None = None
    solver_poly_c: float | None = None
    solver_free_swap_tokens: int = 976
    solver_timeout: float = 0.025  # Gurobi TimeLimit (s)
    # Path to a JSONL sidecar file where the scheduler writes one record per
    # finished request (policy, arrival_strategy, swap_reloads).  Empty = disabled.
    # Set by the bench harness so it can enrich the per-request CSV.
    per_req_stats_path: str = ""

    def __post_init__(self) -> None:
        profile = {**_GENERIC_COST_DEFAULTS, **_detected_cost_profile()}
        for k, default in profile.items():
            if getattr(self, k) is None:
                object.__setattr__(self, k, default)

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
