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
    "A100": { # Yunzhao: TP2 Qwen2.5-14B-Instruct on A100-40GB
            "cost_a": 0.001272,
            "cost_c": 24.987,
            "cost_swap_a1": 0.020533,  # ms/tok
            "per_token_swap_latency": 2.053e-05,  # s/tok
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
    # Master switch for dynamic memory-pressure demotion (V policy). On by
    # default. Set False (``--no-demote``) to ablate demotion entirely:
    # preserved-paused KV is then only reclaimed by the running==0 safety net.
    demote_paused: bool = True
    # Demotion strategy when demote_paused is on. DEFAULT (True) = on-demand:
    # preserved API-waiting requests keep their KV until a NEW request fails to
    # allocate GPU blocks, at which point only the MINIMUM number of
    # preserved-paused requests needed to fit that (chunked) admission are freed
    # (lowest-waste first; CPU offload already holds their KV via the write-through
    # cache). Demoted requests reload only when their API wait ends (never
    # proactively on free memory). This is the best-measured policy across qps
    # 3-13 -- dormant at low load (== no-demote) and beats no-demote/Swap at
    # saturation, with no meltdown.
    # False (``--demote-proactive``) = the original per-step pass: every step keep
    # only the single highest-waste preserved-paused request pinned and demote the
    # rest. Faithful to the original _schedule_chunk_and_fill but churns
    # swap-out/reload and degrades badly under load (kept as an ablation).
    demote_ondemand: bool = True
    # Eager-drop (``V`` only): apply the arrival-classified strategy AT THE PAUSE --
    # free recompute/swap-classified KV immediately (like the direct D/S policies)
    # instead of preserving it and freeing lazily under memory pressure.
    # 'preserve'-classified requests still stay pinned. Default False keeps the lazy
    # on-demand/proactive behavior above. Gated on ``demote_paused`` (``--no-demote``
    # forces pure preserve and wins). The eager free is counted as a demotion in the
    # per-request stats so it categorizes as swap/recompute (``--demote-eager``).
    demote_eager: bool = False
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
