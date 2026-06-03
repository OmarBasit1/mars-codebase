"""Phase 4 end-to-end: adaptive Vulcan ('V') classifies its strategy at arrival.

Faithful to the original, classify() runs at ARRIVAL on the predicted length and
records the KV strategy; the V request then always PAUSES as PRESERVE, and the
classified swap/recompute is applied only later by the every-step demotion (which
needs memory pressure + waiting work). This test runs two V requests with no
contention, so neither is demoted -- both just preserve and finish. The strategy
choice is surfaced by the arrival ``[MARS] classify ... -> strategy=...`` log:
one with a SHORT predicted API time (holding KV is cheap -> preserve) and one
with a LONG predicted API time (holding wastes memory-time -> recompute). The
bash wrapper greps those classify lines.

(``api_exec_time`` -- the orchestrator's simulated sleep -- is kept small for
speed; ``predicted_api_exec_time`` -- what classify reads -- drives the decision.)
"""

import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams

GEN, RET, N_PAUSES = 16, 8, 2


async def run(engine: AsyncLLM, rid: str, predicted_exec: float):
    extra = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=GEN,
        api_return_length=RET,
        api_exec_time=0.05,
        predicted_api_exec_time=predicted_exec,
        api_max_calls=N_PAUSES,
        remain_length=GEN,
        api_policy="V",
    ).to_extra_args()
    segments = [
        ApiSegment(gen_len=GEN, api_exec_time=0.05, api_return_length=RET),
        ApiSegment(gen_len=GEN, api_exec_time=0.05, api_return_length=RET),
        ApiSegment(gen_len=GEN),
    ]
    return await ApiOrchestrator(engine, api_result_token=5000).run_request(
        request_id=rid,
        prompt_token_ids=list(range(20, 40)),
        segments=segments,
        extra_args=extra,
    )


async def main() -> None:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,
            gpu_memory_utilization=0.30,
            enable_prefix_caching=False,
            scheduler_cls="mars.v1.scheduler.MARSScheduler",
        )
    )
    r_short = await run(engine, "V_short", predicted_exec=0.001)  # classify -> preserve
    r_long = await run(engine, "V_long", predicted_exec=1.0)  # classify -> recompute
    print("V_short finished/pauses/gen:", r_short.finished, r_short.pauses, r_short.total_generated)
    print("V_long  finished/pauses/gen:", r_long.finished, r_long.pauses, r_long.total_generated)
    assert r_short.finished and r_long.finished, "request did not finish"
    assert r_short.pauses == N_PAUSES and r_long.pauses == N_PAUSES, "wrong pause count"
    print(">>> PHASE4_VULCAN_E2E_OK")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
