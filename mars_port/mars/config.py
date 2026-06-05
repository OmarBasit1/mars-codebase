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
    "per_token_swap_latency": 4e-5,
}

# Per-GPU calibrated overrides.  Matched as substrings of torch device name.
_GPU_PROFILES: dict[str, dict[str, float]] = {
    "A40": {
            "cost_a": 0.002395,
            "cost_c": 55.122,
            "cost_swap_a1": 0.084919,  # ms/tok (was 84.9189 us/tok -- 1000x unit bug)
            "per_token_swap_latency": 8.492e-05,  # s/tok
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
    # per_token_swap_latency/etc.) or from vllm's scheduler_config.max_num_batched_tokens
    # (cost_max_ragged_batch) at scheduler init time.
    cost_a: float | None = None
    cost_c: float | None = None
    cost_max_ragged_batch: int | None = None
    # Swap-waste coefficients (3-way Vulcan). Original MARS values;
    # re-profile against the CPU-offload transfer path for real experiments.
    cost_swap_a1: float | None = None
    cost_swap_a2: float = 0.181
    cost_swap_c: float = 22.5
    # Host<->GPU KV transfer latency (s/token), profiled by calibrate_cost.py.
    # Drives the v1 async-overlap swap-waste model.
    per_token_swap_latency: float | None = None
    # Dynamic memory-pressure demotion (V policy). Enabled by default.
    # Every step, keep only the single highest-waste preserved-paused request
    # pinned and demote the rest (free their KV -> swap / recompute) so the freed
    # memory can admit waiting work. Pure 'P' is never demoted. Gated only on
    # pending admission demand (requests waiting). Set False to ablate demotion
    # entirely (preserved-paused KV is then only reclaimed by the running==0 safety net).
    demote_paused: bool = True
    # Optional KV-usage floor for demotion. 0 (default) => faithful original:
    # demote whenever work is waiting, regardless of usage. >0 => only demote
    # once usage exceeds this fraction (re-enables the old pressure gate).
    demote_pressure_threshold: float = 0.0
    # Proactive mid-API-wait swap reload (COMPARISON #2). When on, a SWAP-demoted
    # request whose KV is on host has its host->GPU reload STARTED while it is
    # still parked for the API (if GPU memory is spare), so the KV is resident by
    # the time the API returns -- instead of the default admission-gated reload
    # that starts only after resume. Default OFF: the measured post-resume reload
    # latency is ~1% of e2e (the async WFRKV path already overlaps it), so this is
    # opt-in. Only meaningful with a CPU-offload connector (--swap). Gated on spare
    # GPU memory (preload re-occupies the memory swap freed, so it self-limits and
    # a re-demote reclaims it under pressure). See preload_headroom.
    proactive_preload: bool = False
    # Only preload when KV usage is below this fraction (spare GPU memory). Above
    # it, leave demoted-SWAP requests on host (preloading would defeat the swap).
    preload_headroom: float = 0.6
    # Max preloads to START per schedule step (spreads PCIe traffic).
    preload_per_step: int = 2
    # Path to a JSONL sidecar file where the scheduler writes one record per
    # finished request (policy, arrival_strategy, swap_reloads).  Empty = disabled.
    # Set by the bench harness so it can enrich the per-request CSV.
    per_req_stats_path: str = ""
    # V2 re-key amortization: re-rank the V2 heaps every N schedule() calls
    # instead of every step.  Faithful to the original's
    # skip_sorting_for_this_number_of_iterations; default 1 = every step.
    # Higher values reduce per-step CPU at large queue sizes; stale keys between
    # rekeys are acceptable (the original used N≈8 without measurable effect).
    rekey_interval: int = 1

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
