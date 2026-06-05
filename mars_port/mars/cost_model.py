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
            (the original ``384``; read from vLLM ``max_num_batched_tokens``).
        block_size: KV block size (tokens per block).
        per_token_swap_latency: Host<->GPU KV transfer latency (s/token), profiled
            by ``calibrate_cost.py`` (``per_token_swap_latency``). Drives the
            v1 async-overlap swap-waste model.
    """

    a: float = 0.0463
    c: float = 10.0
    max_ragged_batch: int = 384
    block_size: int = 16
    # Async swap-in (host->GPU reload) latency, s/token. The v1 swap-waste model
    # uses this + the forward coeffs only.
    per_token_swap_latency: float = 4e-5
    # DEPRECATED (unused): the original V0 *blocking*-swap coefficients. v1's
    # CPU-offload connector is a write-through cache (swap-out is a sunk,
    # policy-independent mirror) and the swap-in reload is async/overlapped, so
    # the blocking model these parameterised no longer applies. Kept only so
    # existing configs / calibrate output don't error on load.
    swap_a1: float = 0.136
    swap_a2: float = 0.181
    swap_c: float = 22.5


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

    def swap_waste(
        self, num_blocks: int, running_batch: int, running_blocks: int
    ) -> float:
        """w_s: memory-time of the v1 **async, write-through** swap (token-seconds).

        The v1 CPU-offload connector is a write-through cache: every request's KV
        is mirrored GPU->CPU as it is computed, so the swap-OUT is a sunk,
        policy-independent cost (paid by preserve/recompute/swap alike). Choosing
        SWAP for a request therefore only adds the **swap-IN reload**, which is
        async and overlaps compute. So the original V0 blocking model (a per-iter
        stall, swap-out+in contention ``x2``) does not apply; this models the
        reload instead:

            T       = per_token_swap_latency * req_tokens   # async reload time (s)
            overlap = f_fwd * n                              # compute that hides it
            exposed = max(T - overlap, 0)                    # un-hidden remainder
            w_s     = T * req_tokens                         # own KV held during reload
                    + exposed * running_blocks * bs          # only exposed part contends (x1)

        where ``f_fwd = (a*max_ragged_batch + c)/1000`` is the forward time per
        resume iteration and ``n`` the number of resume iterations. Uses only the
        profiled ``per_token_swap_latency`` + the forward coeffs (the V0
        ``swap_a1/a2/c`` are no longer used). When the reload hides fully
        (``exposed == 0``, the common case) ``w_s`` reduces to
        ``per_token_swap_latency * req_tokens^2`` -- small, so swap is the
        cheapest free-the-memory option exactly where it wins in practice.
        """
        co = self.coeffs
        bs = co.block_size
        req_tokens = bs * num_blocks
        c_h = max(co.max_ragged_batch - running_batch, 1)
        n = max((req_tokens + c_h - 1) // c_h - 1, 1)
        transfer_s = co.per_token_swap_latency * req_tokens
        f_fwd = (co.a * co.max_ragged_batch + co.c) / 1000.0
        exposed_s = max(transfer_s - f_fwd * n, 0.0)
        return transfer_s * req_tokens + exposed_s * running_blocks * bs

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

    def choose(
        self,
        *,
        api_exec_time: float,
        before_api_tokens: int,
        num_blocks: int,
        running_batch: int,
        running_blocks: int,
        swap_available: bool,
    ) -> tuple[PauseMode, dict[PauseMode, float]]:
        """3-way (or 2-way) Vulcan: pick the minimum-waste mode.

        Returns ``(mode, wastes)`` where ``wastes`` maps each evaluated mode to
        its waste (SWAP is included only when ``swap_available``).

        Tie-breaking is **preserve > swap > recompute**, matching the original
        ``classify()``'s comparison order (``w_p <= both`` then ``w_s <= both``
        else recompute) -- so a ``w_s == w_d < w_p`` tie resolves to SWAP, not
        RECOMPUTE (a plain ``min`` over the dict would pick RECOMPUTE).
        """
        w_p = self.preserve_waste(api_exec_time, before_api_tokens)
        w_d = self.discard_waste(num_blocks, running_batch, running_blocks)
        wastes: dict[PauseMode, float] = {
            PauseMode.PRESERVE: w_p,
            PauseMode.RECOMPUTE: w_d,
        }
        w_s = float("inf")
        if swap_available:
            w_s = self.swap_waste(num_blocks, running_batch, running_blocks)
            wastes[PauseMode.SWAP] = w_s
        if w_p <= w_d and w_p <= w_s:
            mode = PauseMode.PRESERVE
        elif w_s <= w_p and w_s <= w_d:
            mode = PauseMode.SWAP
        else:
            mode = PauseMode.RECOMPUTE
        return mode, wastes
