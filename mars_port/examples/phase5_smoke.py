"""Phase 5 end-to-end: swap-free baseline policies + SJF queue wiring.

The engine is built with policy_config='sjf' (so the SJF MARSRequestQueue is
swapped in and must not break scheduling), and per-request api_policy exercises
the threshold heuristics and the static G/I baselines:

  * H-D with a short API time (<7 s)  -> PRESERVE
  * H-D with a long  API time (>=7 s) -> RECOMPUTE
  * G (Greedy, static)                -> PRESERVE
  * I (InferCept, static)             -> PRESERVE

(The MARS api_exec_time that drives the heuristic is decoupled from the
orchestrator's small simulated sleep, so the test stays fast.) The bash wrapper
greps the '[MARS] pause' logs to confirm each decision.
"""

import asyncio

from vllm.config import VllmConfig  # noqa: F401  (ensures vllm import resolves first)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.config import MarsConfig
from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams

GEN, RET, N_PAUSES = 16, 8, 2


async def run(engine: AsyncLLM, rid: str, policy: str, api_exec_time: float):
    extra = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=GEN,
        api_return_length=RET,
        api_exec_time=api_exec_time,  # drives the heuristic threshold
        predicted_api_exec_time=api_exec_time,
        api_max_calls=N_PAUSES,
        remain_length=GEN,
        api_policy=policy,
    ).to_extra_args()
    segments = [
        ApiSegment(gen_len=GEN, api_exec_time=0.05, api_return_length=RET),  # fast sleep
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
            additional_config=MarsConfig(policy_config="sjf").to_additional_config(),
        )
    )
    cases = [
        ("HD_short", "H-D", 1.0),   # -> preserve
        ("HD_long", "H-D", 10.0),   # -> recompute
        ("G_req", "G", 5.0),        # -> preserve (static)
        ("I_req", "I", 5.0),        # -> preserve (static)
    ]
    for rid, policy, api_t in cases:
        r = await run(engine, rid, policy, api_t)
        print(f"{rid:>9} ({policy}): finished={r.finished} pauses={r.pauses} gen={r.total_generated}")
        assert r.finished and r.pauses == N_PAUSES and r.total_generated == GEN * 3, rid
    print(">>> PHASE5_BASELINES_E2E_OK")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
