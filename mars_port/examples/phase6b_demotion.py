"""Dynamic memory-pressure demotion (Greedy / InferCept).

Forces a tiny KV cache (num_gpu_blocks_override) and launches many requests
concurrently under api_policy='G'. They pause (PRESERVE) and pile up, pinning
KV, until usage crosses the threshold with requests still waiting -> the
scheduler demotes the cheaper preserved-paused requests (frees their KV ->
recompute) so the waiting work can run. Asserts every request still finishes;
the bash wrapper greps the '[MARS] demote' logs to confirm demotion fired.
"""

import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.config import MarsConfig
from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams

N = 16
PROMPT = 64
GEN = 16
PAUSE_S = 3.0  # long pauses so preserved-paused requests pile up vs waiting work


async def run(engine: AsyncLLM, rid: str, policy: str):
    extra = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=GEN,
        api_return_length=8,
        api_exec_time=PAUSE_S,
        predicted_api_exec_time=PAUSE_S,
        api_max_calls=1,
        remain_length=GEN,
        api_policy=policy,
    ).to_extra_args()
    segments = [ApiSegment(gen_len=GEN, api_exec_time=PAUSE_S, api_return_length=8),
                ApiSegment(gen_len=GEN)]
    return await ApiOrchestrator(engine, api_result_token=5000).run_request(
        request_id=rid,
        prompt_token_ids=list(range(10, 10 + PROMPT)),
        segments=segments,
        extra_args=extra,
    )


async def main() -> None:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            max_model_len=256,  # so a small block budget is still valid
            num_gpu_blocks_override=40,  # force a scarce KV cache (40*16=640 tok)
            enable_prefix_caching=False,
            scheduler_cls="mars.v1.scheduler.MARSScheduler",
            additional_config=MarsConfig(
                api_policy="G", demote_pressure_threshold=0.5
            ).to_additional_config(),
        )
    )
    results = await asyncio.wait_for(
        asyncio.gather(*(run(engine, f"g{i}", "G") for i in range(N))), timeout=240
    )
    n_fin = sum(r.finished for r in results)
    total = sum(r.total_generated for r in results)
    print(f"requests: {N}  finished: {n_fin}  total_gen: {total}")
    assert n_fin == N, f"only {n_fin}/{N} finished (possible demotion deadlock)"
    assert all(r.total_generated == GEN * 2 for r in results), "wrong token counts"
    print(">>> PHASE6B_DEMOTION_E2E_OK")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
