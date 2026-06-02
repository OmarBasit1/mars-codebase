"""End-to-end: 'V' with the Gurobi solver (decision-only) drives the pause mode."""

import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.config import MarsConfig
from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams

GEN, RET, N_PAUSES = 16, 8, 2


async def run(engine: AsyncLLM, rid: str):
    extra = MarsApiParams(
        use_api_simulator=True, api_invoke_interval=GEN, api_return_length=RET,
        api_exec_time=2.0, predicted_api_exec_time=2.0, api_max_calls=N_PAUSES,
        remain_length=GEN, api_policy="V",
    ).to_extra_args()
    segments = [ApiSegment(GEN, 0.05, RET), ApiSegment(GEN, 0.05, RET), ApiSegment(GEN)]
    return await ApiOrchestrator(engine, api_result_token=5000).run_request(
        request_id=rid, prompt_token_ids=list(range(20, 40)), segments=segments,
        extra_args=extra,
    )


async def main() -> None:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.30,
            enable_prefix_caching=False,
            scheduler_cls="mars.v1.scheduler.MARSScheduler",
            additional_config=MarsConfig(
                api_policy="V", use_solver=True
            ).to_additional_config(),
        )
    )
    r = await run(engine, "Vsolver")
    print("finished/pauses/gen:", r.finished, r.pauses, r.total_generated)
    assert r.finished and r.pauses == N_PAUSES and r.total_generated == GEN * 3
    print(">>> SOLVER_E2E_OK")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
