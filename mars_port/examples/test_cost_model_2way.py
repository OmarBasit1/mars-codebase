"""Unit test: 2-way cost model preserve/recompute crossover (no GPU).

For a paused request competing with a running batch, sweeping the API execution
time must give: short API -> PRESERVE (holding KV is cheap), long API ->
RECOMPUTE (holding wastes more memory-time than recomputing), with a single
monotonic crossover. ``w_d`` must not depend on the API time.
"""

from mars.cost_model import CostModel, CostModelCoeffs
from mars.policies import PauseMode


def main() -> None:
    cm = CostModel(CostModelCoeffs(a=0.0463, c=10.0, max_ragged_batch=384, block_size=16))
    # A 512-token paused request, contending with a loaded running batch.
    ctx = dict(before_api_tokens=512, num_blocks=32, running_batch=100, running_blocks=200)

    rows = []
    for t in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]:
        mode, w_p, w_d = cm.choose_2way(api_exec_time=t, **ctx)
        rows.append((t, mode, w_p, w_d))
        print(f"api_exec_time={t:>6}  w_p={w_p:9.3f}  w_d={w_d:9.3f}  -> {mode.value}")

    modes = [m for _, m, _, _ in rows]
    assert modes[0] is PauseMode.PRESERVE, "shortest API time should PRESERVE"
    assert modes[-1] is PauseMode.RECOMPUTE, "longest API time should RECOMPUTE"

    # Monotonic: once it flips to RECOMPUTE it stays RECOMPUTE.
    flipped = False
    for m in modes:
        if m is PauseMode.RECOMPUTE:
            flipped = True
        if flipped:
            assert m is PauseMode.RECOMPUTE, "crossover is not monotonic"

    # w_d is independent of the API time.
    assert len({round(w_d, 6) for _, _, _, w_d in rows}) == 1, "w_d depends on API time"

    # Crossover should fall strictly inside the swept range.
    assert any(m is PauseMode.PRESERVE for m in modes)
    assert any(m is PauseMode.RECOMPUTE for m in modes)
    print(">>> COST_MODEL_2WAY_OK")


if __name__ == "__main__":
    main()
