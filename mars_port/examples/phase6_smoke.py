"""Phase 6 end-to-end: SWAP via the native CPU-offload connector.

Builds an engine with vLLM's SimpleCPUOffloadConnector + prefix caching (so
MARSScheduler enables SWAP), then runs the SAME 2-pause request under Preserve,
Swap, and Recompute and asserts they all produce BYTE-IDENTICAL output -- i.e.
swap correctly reloads the KV from the cache/host on resume, and recompute
correctly rebuilds it; the policy is purely a memory/perf choice. The bash
wrapper greps the '[MARS]' logs to confirm swap_available=True and the three
distinct mechanisms (preserve / swap / recompute).
"""

import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams
from mars.swap import cpu_offload_kv_transfer_config

GEN, RET, N_PAUSES = 16, 8, 2


async def run(engine: AsyncLLM, rid: str, policy: str):
    extra = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=GEN,
        api_return_length=RET,
        api_exec_time=0.05,
        predicted_api_exec_time=0.05,
        api_max_calls=N_PAUSES,
        remain_length=GEN,
        api_policy=policy,
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


def _flat(result) -> list[int]:
    return [t for seg in result.segment_tokens for t in seg]


async def main() -> None:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,
            gpu_memory_utilization=0.30,
            enable_prefix_caching=True,  # required by the CPU-offload connector
            kv_transfer_config=cpu_offload_kv_transfer_config(cpu_gb=2.0),
            scheduler_cls="mars.v1.scheduler.MARSScheduler",
        )
    )
    rP = await run(engine, "P_req", "P")
    rS = await run(engine, "S_req", "S")
    rD = await run(engine, "D_req", "D")
    fP, fS, fD = _flat(rP), _flat(rS), _flat(rD)
    print("P:", rP.finished, rP.pauses, rP.total_generated)
    print("S:", rS.finished, rS.pauses, rS.total_generated)
    print("D:", rD.finished, rD.pauses, rD.total_generated)
    print("P==S:", fP == fS, " P==D:", fP == fD)

    assert rP.finished and rS.finished and rD.finished, "request did not finish"
    assert all(r.pauses == N_PAUSES for r in (rP, rS, rD)), "wrong pause count"
    assert fP == fS == fD, "preserve / swap / recompute produced different output!"
    print(">>> PHASE6_SWAP_E2E_OK")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
