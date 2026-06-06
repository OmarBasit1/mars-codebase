"""Unit test: 3-way cost model with swap (no GPU).

Confirms ``choose`` evaluates SWAP only when available, and that SWAP can be
selected when it is the minimum-waste option (cheaper swap-transfer coefficients
+ long API + heavy recompute contention).
"""

from mars.cost_model import CostModel, CostModelCoeffs
from mars.policies import PauseMode


def main() -> None:
    cm = CostModel(CostModelCoeffs(block_size=16))
    ctx = dict(before_api_tokens=512, num_blocks=32, running_batch=100, running_blocks=200)

    _, w3 = cm.choose(api_exec_time=1.0, swap_available=True, **ctx)
    assert set(w3) == {PauseMode.PRESERVE, PauseMode.RECOMPUTE, PauseMode.SWAP}
    _, w2 = cm.choose(api_exec_time=1.0, swap_available=False, **ctx)
    assert set(w2) == {PauseMode.PRESERVE, PauseMode.RECOMPUTE}
    print("default 3-way:", {m.value: round(w, 3) for m, w in w3.items()})

    # With a cheap swap path, a long API + heavy recompute contention -> SWAP.
    cheap = CostModel(
        CostModelCoeffs(block_size=16, swap_a1=0.001, swap_a2=0.001, swap_c=0.1)
    )
    sctx = dict(
        api_exec_time=3.0,
        before_api_tokens=2048,
        num_blocks=128,
        running_batch=360,
        running_blocks=400,
    )
    smode, sw = cheap.choose(swap_available=True, **sctx)
    print("cheap-swap regime:", {m.value: round(w, 2) for m, w in sw.items()}, "->", smode.value)
    assert smode is PauseMode.SWAP, (smode, sw)
    print(">>> COST_MODEL_3WAY_OK")


if __name__ == "__main__":
    main()
