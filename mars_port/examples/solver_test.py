"""Solver unit test: the MIP returns a valid split and it maps to a mode."""

from mars.policies import PauseMode
from mars.v1.solver import GUROBI_AVAILABLE, MarsSolver, SolverParams, split_to_mode


def main() -> None:
    print("gurobipy available:", GUROBI_AVAILABLE)

    # split_to_mode mapping (no gurobi needed). c_blocks=60 so the swap/discard
    # share (10*5=50) clearly dominates the preserved remainder (10).
    assert split_to_mode(0, 0, 0, 100, swap_available=True)[0] is PauseMode.PRESERVE
    assert split_to_mode(10, 0, 5, 60, swap_available=True)[0] is PauseMode.SWAP
    assert split_to_mode(0, 10, 5, 60, swap_available=True)[0] is PauseMode.RECOMPUTE
    assert (
        split_to_mode(10, 0, 5, 60, swap_available=False, swap_fallback="recompute")[0]
        is PauseMode.RECOMPUTE
    )

    s = MarsSolver(SolverParams(block_size=16, target=1500))
    res = s.solve_blocks(
        num_tokens=1024, num_active_gpu_blocks=1024, api_exec_time=5.0,
        api_return_length=128, arrival_time=0.0, now=0.0,
        running_query_head=16, running_query_tail=16,
        swap_in_chunks_head=s.free_swap, swap_in_chunks_tail=s.free_swap,
    )
    print("solve_blocks ->", res)
    if GUROBI_AVAILABLE:
        assert res is not None, "solver returned no solution"
        c_s, c_d, n_e = res
        assert c_s >= 0 and c_d >= 0 and n_e >= 0
        mode, shares = split_to_mode(c_s, c_d, n_e, 64, swap_available=True)
        print("mode:", mode.value, "shares:", {k.value: v for k, v in shares.items()})
        assert mode in (PauseMode.PRESERVE, PauseMode.SWAP, PauseMode.RECOMPUTE)
    print(">>> SOLVER_OK")


if __name__ == "__main__":
    main()
