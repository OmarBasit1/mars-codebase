"""MARS waste cost-model (2-way: Preserve vs Recompute).

Ported from the original ``scheduler_v2.py`` (``classify`` / ``discard_waste``).
Both "wastes" are GPU-memory-time (token-seconds), so they are directly
comparable:

  * **w_p (preserve)** = ``api_exec_time * before_api_tokens`` — KV held idle on
    the GPU for the duration of the API call.
  * **w_d (recompute)** = the memory-time the recompute steals from the running
    batch: recomputing the request's ``num_blocks`` of KV takes ``n`` extra
    forward iterations (each bounded by the compute headroom ``c_h``), and the
    quadratic-in-``n`` term captures the cumulative delay imposed on the
    running batch.

Vulcan picks the cheaper. Swap (the original third option) is added in Phase 6.

Coefficients (``a``, ``c``, ``max_ragged_batch``) are model/GPU specific — the
defaults are the original values; re-tune with ``examples/calibrate_cost.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from mars.policies import PauseMode


@dataclass
class CostModelCoeffs:
    """Hardware/model-specific coefficients for the waste model.

    Attributes:
        a: Per-token slope of the forward-step time (ms/token).
        c: Fixed per-step overhead (ms).
        max_ragged_batch: Tokens/step at which the forward becomes compute-bound
            (the original ``384``).
        block_size: KV block size (tokens per block).
    """

    a: float = 0.0463
    c: float = 10.0
    max_ragged_batch: int = 384
    block_size: int = 16


class CostModel:
    """Computes preserve/recompute waste and the 2-way Vulcan decision."""

    def __init__(self, coeffs: CostModelCoeffs) -> None:
        self.coeffs = coeffs

    def preserve_waste(self, api_exec_time: float, before_api_tokens: int) -> float:
        """w_p: GPU-memory-time the KV sits idle during the API call (token-seconds)."""
        return api_exec_time * before_api_tokens

    def discard_waste(
        self, num_blocks: int, running_batch: int, running_blocks: int
    ) -> float:
        """w_d: memory-time the recompute steals from the running batch.

        Args:
            num_blocks: KV blocks the paused request needs recomputed.
            running_batch: Tokens being computed by the running batch this step
                (the original ``inflight_length`` sum).
            running_blocks: Total GPU blocks held by running requests.

        Returns:
            The recompute waste in token-seconds.
        """
        co = self.coeffs
        bs = co.block_size
        c_h = max(co.max_ragged_batch - running_batch, 1)
        # Extra forward iterations to recompute this request's tokens, given the
        # per-step compute headroom c_h.
        n = max((bs * num_blocks + c_h - 1) // c_h - 1, 0)
        f_ch = (co.a * c_h) / 1000.0
        f_s = (co.a * co.max_ragged_batch + co.c) / 1000.0
        # memory * time, quadratic in n (cumulative delay to the running batch).
        w_d = f_s * (1 + n) * n / 2 * c_h + f_ch * n * running_blocks * bs
        last_resume_toks = (bs * num_blocks) % c_h
        f_last = (co.a * max(0, last_resume_toks)) / 1000.0
        w_d += f_last * (running_blocks * bs + last_resume_toks)
        return w_d

    def choose_2way(
        self,
        *,
        api_exec_time: float,
        before_api_tokens: int,
        num_blocks: int,
        running_batch: int,
        running_blocks: int,
    ) -> tuple[PauseMode, float, float]:
        """Return ``(mode, w_p, w_d)`` choosing the cheaper of preserve/recompute."""
        w_p = self.preserve_waste(api_exec_time, before_api_tokens)
        w_d = self.discard_waste(num_blocks, running_batch, running_blocks)
        mode = PauseMode.PRESERVE if w_p <= w_d else PauseMode.RECOMPUTE
        return mode, w_p, w_d
