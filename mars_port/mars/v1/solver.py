"""Gurobi solver for the 'V' (Vulcan) policy — decision-only port.

The original MARS solver (core/solver.py) solves a MIP for the optimal way to
split ONE paused request's KV cache into preserved / swapped / recomputed blocks
across ``n_e`` iterations. vLLM v1 has no per-request *partial*-KV-region
execution, so we run the SAME MIP but apply only the **dominant whole-request
mode** (PRESERVE / SWAP / RECOMPUTE). This whole-request decision is the largest
intentional discrepancy vs the original — see COMPARISON.md.

Import-safe without gurobipy (``GUROBI_AVAILABLE`` is then False); callers fall
back to the greedy cost model.
"""

from __future__ import annotations

from dataclasses import dataclass

from mars.policies import PauseMode

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    GUROBI_AVAILABLE = False


@dataclass
class SolverParams:
    block_size: int = 16
    target: float = 1500.0
    timeout: float = 0.025
    free_swap_tokens: int = 976
    per_token_swap_latency: float = 4e-5
    poly_a: float = 1.3e-5
    poly_b: float = 0.328
    poly_c: float = 24.1


class MarsSolver:
    """Decision-only wrapper around the original MARS MIP."""

    def __init__(self, params: SolverParams) -> None:
        self.p = params
        self.block_size = params.block_size
        self.target = params.target / self.block_size
        self.timeout = params.timeout
        self.free_swap = (params.free_swap_tokens + self.block_size - 1) // self.block_size
        self.per_token_swap_latency = params.per_token_swap_latency
        self.a, self.b, self.c = params.poly_a, params.poly_b, params.poly_c

    def solve_blocks(
        self,
        *,
        num_tokens: int,
        num_active_gpu_blocks: int,
        api_exec_time: float,
        api_return_length: int,
        arrival_time: float,
        now: float,
        running_query_head: int,
        running_query_tail: int,
        swap_in_chunks_head: int,
        swap_in_chunks_tail: int,
    ):
        """Return ``(c_s, c_d, n_e)`` blocks/iter split, or ``None`` on failure.

        Mirrors ``Solver.solve`` from the original (re-parameterized off plain
        ints rather than a ``Sequence``).
        """
        if not GUROBI_AVAILABLE:
            return None
        C = num_tokens
        ret_len = api_return_length
        t_arrival = arrival_time
        t_call = now
        t_start = now
        t_resumed = (C + ret_len) / self.target + t_arrival + api_exec_time
        if t_resumed <= t_start:
            t_resumed = t_start + 0.1
        C = (C + self.block_size - 1) // self.block_size
        S_gpu = max(num_active_gpu_blocks, C)
        try:
            return self._solve(
                C, S_gpu, t_call, t_start, t_resumed,
                running_query_head, running_query_tail,
                swap_in_chunks_head, swap_in_chunks_tail,
            )
        except Exception:
            return None

    def _solve(self, C, S_gpu, t_call, t_start, t_resumed, running_query_head,
               running_query_tail, c_sin_head, c_sin_tail):
        m = gp.Model("mars")
        m.setParam("NonConvex", 2)
        m.setParam("TimeLimit", self.timeout)
        m.setParam("LogToConsole", 0)
        m.setParam("OutputFlag", 0)

        f_normal = (self.a * running_query_head**2 + self.b * running_query_head + self.c) / 1000

        c_s = m.addVar(vtype=GRB.INTEGER, name="c_s")
        c_d = m.addVar(vtype=GRB.INTEGER, name="c_d")
        n_e = m.addVar(vtype=GRB.INTEGER, name="n_e")
        b_osout = m.addVar(vtype=GRB.INTEGER, name="b_osout")
        c_osin = m.addVar(vtype=GRB.INTEGER, name="c_osin")
        s_p = m.addVar(vtype=GRB.INTEGER, name="s_p")
        f_swap = m.addVar(vtype=GRB.CONTINUOUS, name="f_swap")
        f_resume = m.addVar(vtype=GRB.CONTINUOUS, name="f_resume")

        for v in (n_e, c_s, b_osout, c_osin, c_d, f_swap, f_resume):
            m.addConstr(v >= 0)
        m.addConstr(n_e * (c_s + c_d) >= 0)
        m.addConstr(n_e * (c_s + c_d) <= C)
        m.addConstr(
            n_e * (f_resume + self.per_token_swap_latency * (c_osin + c_sin_tail - self.free_swap) * self.block_size)
            <= (t_resumed - t_start)
        )
        m.addConstr(s_p == C - n_e * (c_s + c_d))
        m.addConstr(b_osout >= (c_s * n_e + c_sin_head - self.free_swap))
        m.addConstr(c_osin >= (c_s + c_sin_tail) - self.free_swap)
        m.addConstr(f_swap == self.per_token_swap_latency * (b_osout + c_sin_head - self.free_swap) * self.block_size)
        m.addConstr(
            f_resume
            == (self.a * (running_query_tail + c_d * self.block_size) ** 2
                + self.b * (running_query_tail + c_d * self.block_size) + self.c) / 1000
        )

        w_p = s_p * (t_resumed - t_call)
        w_d = (n_e + 1) / 2 * c_d * (t_resumed - t_start)
        w_s = (n_e + 1) / 2 * c_s * (t_resumed - t_start) + f_swap * S_gpu
        w_o = S_gpu * (self.per_token_swap_latency * (c_osin + c_sin_tail - self.free_swap) * self.block_size + f_resume - f_normal) * n_e
        m.setObjective(w_p + w_d + w_s + w_o, GRB.MINIMIZE)

        m.optimize()
        if m.SolCount == 0:
            return None
        return (round(c_s.X), round(c_d.X), round(n_e.X))


def split_to_mode(
    c_s: int, c_d: int, n_e: int, c_blocks: int, *,
    swap_available: bool, swap_fallback: str = "recompute",
) -> tuple[PauseMode, dict[PauseMode, int]]:
    """Map the solver's split to the dominant whole-request :class:`PauseMode`.

    The mode whose blocks cover the largest share of the request wins; SWAP
    degrades to the fallback when no CPU-offload connector is configured.
    """
    swap_blocks = max(0, c_s) * max(0, n_e)
    disc_blocks = max(0, c_d) * max(0, n_e)
    s_p = max(0, c_blocks - (swap_blocks + disc_blocks))
    shares = {
        PauseMode.PRESERVE: s_p,
        PauseMode.SWAP: swap_blocks,
        PauseMode.RECOMPUTE: disc_blocks,
    }
    mode = max(shares, key=lambda k: shares[k])
    if mode is PauseMode.SWAP and not swap_available:
        mode = PauseMode.RECOMPUTE if swap_fallback == "recompute" else PauseMode.PRESERVE
    return mode, shares
