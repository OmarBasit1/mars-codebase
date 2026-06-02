"""Per-policy predicted-cost report (uses the calibrated cost model).

Given a workload + cost-model coefficients, estimate the total/mean memory-time
"waste" (token-seconds) each MARS policy would incur across the workload's API
pauses — lower waste ⇒ likely lower latency. This explains *why* policies differ
and complements the empirical ``mars.bench.run``. Pure analysis (no GPU).

Run (from a neutral cwd):
    python -m mars.bench.policy_cost --workload <json> [--cost-a ... --cost-swap-a1 ...]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mars.config import MarsConfig
from mars.cost_model import CostModel, CostModelCoeffs
from mars.policies import (
    THRESHOLD_POLICIES,
    PauseMode,
    decide_pause_mode,
    decide_threshold_mode,
)

POLICIES = ["P", "D", "S", "V", "G", "I", "H", "H-S", "H-D", "H-B"]


def waste_for(policy, *, wastes, api_exec_time, heuristic_coef):
    """Predicted waste for ``policy`` at one pause given the three mode costs."""
    if policy == "V":
        return min(wastes.values())
    if policy in ("G", "I"):
        return wastes[PauseMode.PRESERVE]  # static; demotes under pressure
    if policy in THRESHOLD_POLICIES:
        mode = decide_threshold_mode(
            policy, api_exec_time=api_exec_time, heuristic_coef=heuristic_coef
        )
        return wastes[mode]
    mode = decide_pause_mode(policy, swap_available=True)  # P / D / S
    return wastes[mode]


def main() -> None:
    d = MarsConfig()  # defaults for the cost coefficients
    ap = argparse.ArgumentParser(description="Per-policy predicted-cost report.")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--running-batch", type=int, default=128)
    ap.add_argument("--running-blocks", type=int, default=256)
    ap.add_argument("--heuristic-coef", type=float, default=d.heuristic_coef)
    ap.add_argument("--cost-a", type=float, default=d.cost_a)
    ap.add_argument("--cost-c", type=float, default=d.cost_c)
    ap.add_argument("--cost-swap-a1", type=float, default=d.cost_swap_a1)
    ap.add_argument("--cost-swap-a2", type=float, default=d.cost_swap_a2)
    ap.add_argument("--cost-swap-c", type=float, default=d.cost_swap_c)
    args = ap.parse_args()

    cm = CostModel(
        CostModelCoeffs(
            a=args.cost_a, c=args.cost_c, block_size=args.block_size,
            swap_a1=args.cost_swap_a1, swap_a2=args.cost_swap_a2, swap_c=args.cost_swap_c,
        )
    )
    bs = args.block_size
    workload = json.load(open(args.workload))

    totals = {p: 0.0 for p in POLICIES}
    n_pauses = 0
    for segs in workload.values():
        for i, seg in enumerate(segs[:-1]):  # each non-final segment = one API pause
            api_exec = float(seg.get("api_time", 0.0))
            if api_exec <= 0:
                continue
            before = int(segs[0].get("prompt_tokens", 0)) + int(seg.get("completion_tokens", 0))
            nb = max(1, (before + bs - 1) // bs)
            wastes = {
                PauseMode.PRESERVE: cm.preserve_waste(api_exec, nb * bs),
                PauseMode.RECOMPUTE: cm.discard_waste(nb, args.running_batch, args.running_blocks),
                PauseMode.SWAP: cm.swap_waste(nb, args.running_batch, args.running_blocks),
            }
            n_pauses += 1
            for p in POLICIES:
                totals[p] += waste_for(
                    p, wastes=wastes, api_exec_time=api_exec, heuristic_coef=args.heuristic_coef
                )

    print(f"workload: {len(workload)} requests, {n_pauses} API pauses")
    print(f"context: running_batch={args.running_batch} running_blocks={args.running_blocks}")
    print(f"{'policy':<8}{'total waste':>16}{'mean/pause':>14}")
    for p in sorted(POLICIES, key=lambda k: totals[k]):
        mean = totals[p] / n_pauses if n_pauses else float("nan")
        print(f"{p:<8}{totals[p]:>16.2f}{mean:>14.4f}")
    print(">>> POLICY_COST_DONE")


if __name__ == "__main__":
    main()
