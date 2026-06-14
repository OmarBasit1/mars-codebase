"""Async-safe pause-time KV free: deferred free produces identical output.

Under async scheduling the MARS pause hook (D/S policies) fires from
``update_from_output(N)`` while batch N+1 may be in-flight.  The fix records the
free in ``mars_pending_pause_free`` and applies it in the next ``schedule()`` step
(``_mars_flush_pending_pause_frees``).

This test verifies correctness by forcing the deferred path via the env hook
``MARS_FORCE_DEFER_PAUSE_FREE=2`` (forces the first 2 pause-frees to defer even
when not actually in-flight) and asserting:
  * the request completes with the expected pause count and token count;
  * the token stream is byte-identical to the control run (deferred free changes
    WHEN the memory is freed, not WHAT is computed — D policy recomputes from 0
    regardless).

Run from a neutral cwd:
    cd /tmp && CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/test_async_pause_free.py
"""

import asyncio
import os

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams

GEN, RET, N_PAUSES = 16, 8, 2


async def run_request(engine: AsyncLLM, rid: str, policy: str):
    extra = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=GEN,
        api_return_length=RET,
        api_exec_time=0.0,   # zero API wait: maximises the resume-before-free race
        api_max_calls=N_PAUSES,
        remain_length=GEN,
        api_policy=policy,
    ).to_extra_args()
    segments = [
        ApiSegment(gen_len=GEN, api_exec_time=0.0, api_return_length=RET),
        ApiSegment(gen_len=GEN, api_exec_time=0.0, api_return_length=RET),
        ApiSegment(gen_len=GEN),
    ]
    return await ApiOrchestrator(engine, api_result_token=5000).run_request(
        request_id=rid,
        prompt_token_ids=list(range(20, 40)),
        segments=segments,
        extra_args=extra,
    )


def _flat(result) -> list[int]:
    return [t for seg in result.segment_tokens for t in seg]


def _make_engine() -> AsyncLLM:
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,
            gpu_memory_utilization=0.35,
            enable_prefix_caching=False,
            scheduler_cls="mars.v1.scheduler.MARSScheduler",
            # async_scheduling defaults to True inside AsyncLLM/AsyncEngineArgs.
        )
    )


async def run_arm(force_defer: int, policy: str = "D") -> list[int]:
    """Run one arm; return the flat token list."""
    prev = os.environ.get("MARS_FORCE_DEFER_PAUSE_FREE")
    os.environ["MARS_FORCE_DEFER_PAUSE_FREE"] = str(force_defer)
    try:
        engine = _make_engine()
        result = await run_request(engine, f"req_defer{force_defer}", policy)
        engine.shutdown()
    finally:
        if prev is None:
            os.environ.pop("MARS_FORCE_DEFER_PAUSE_FREE", None)
        else:
            os.environ["MARS_FORCE_DEFER_PAUSE_FREE"] = prev
    assert result.finished, f"request did not finish (force_defer={force_defer})"
    assert result.pauses == N_PAUSES, (
        f"wrong pause count: expected {N_PAUSES}, got {result.pauses} "
        f"(force_defer={force_defer})"
    )
    assert result.total_generated == GEN * (N_PAUSES + 1), (
        f"wrong token count: expected {GEN * (N_PAUSES + 1)}, "
        f"got {result.total_generated} (force_defer={force_defer})"
    )
    return _flat(result)


async def main() -> None:
    print("--- control arm (no forced defer) ---")
    toks_control = await run_arm(force_defer=0, policy="D")
    print(f"control: {len(toks_control)} tokens generated  ✓")

    print("--- forced-defer arm (MARS_FORCE_DEFER_PAUSE_FREE=2) ---")
    toks_forced = await run_arm(force_defer=2, policy="D")
    print(f"forced:  {len(toks_forced)} tokens generated  ✓")

    n_diff = sum(1 for a, b in zip(toks_control, toks_forced) if a != b) + abs(
        len(toks_control) - len(toks_forced)
    )
    assert toks_control == toks_forced, (
        f"deferred pause-free changed the output ({n_diff} tokens differ)!\n"
        f"  control: {toks_control}\n  forced:  {toks_forced}"
    )
    print(f"outputs identical ({len(toks_control)} tokens)  ✓")

    # Also verify S (swap) policy with force-defer — swap+defer is the
    # riskier path (connector state + reload) so worth checking.
    print("--- forced-defer arm S policy (MARS_FORCE_DEFER_PAUSE_FREE=2) ---")
    toks_swap = await run_arm(force_defer=2, policy="S")
    print(f"swap/forced: {len(toks_swap)} tokens generated  ✓")

    print(">>> ASYNC_PAUSE_FREE_OK")


if __name__ == "__main__":
    asyncio.run(main())
