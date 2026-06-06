"""Proactive preload correctness: preload ON/OFF produces byte-identical output.

Runs the SAME staggered SWAP workload with proactive preload OFF and ON (separate
engines, fixed seed) and asserts every request's output is BYTE-IDENTICAL between
the two. The feature is opt-in (--proactive-preload, default off).

NOTE: in practice the preload rarely fires. With v1's CPU-offload connector +
prefix caching, swap-freed KV stays in the GPU prefix cache and is re-served
locally (is_async=False); the host-resident path only engages after actual
eviction under sustained pressure. This test's job is the byte-identical guard.

Run from a neutral cwd:
    cd /tmp && CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/test_proactive_preload.py
"""

import asyncio
import sys

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.config import MarsConfig
from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams
from mars.swap import cpu_offload_kv_transfer_config

N, GEN, RET, N_PAUSES, API_T = 6, 16, 8, 2, 0.6
STAGGER = 0.15  # seconds between request launches -> overlap parked vs running


async def run(engine: AsyncLLM, rid: str):
    extra = MarsApiParams(
        use_api_simulator=True, api_invoke_interval=GEN, api_return_length=RET,
        api_exec_time=API_T, predicted_api_exec_time=API_T, api_max_calls=N_PAUSES,
        remain_length=GEN, api_policy="S",
    ).to_extra_args()
    segments = [
        ApiSegment(gen_len=GEN, api_exec_time=API_T, api_return_length=RET),
        ApiSegment(gen_len=GEN, api_exec_time=API_T, api_return_length=RET),
        ApiSegment(gen_len=GEN),
    ]
    return await ApiOrchestrator(engine, api_result_token=5000).run_request(
        request_id=rid, prompt_token_ids=list(range(20, 44)), segments=segments,
        extra_args=extra,
    )


def _flat(r) -> list[int]:
    return [t for seg in r.segment_tokens for t in seg]


async def drive(engine: AsyncLLM) -> dict[str, list[int]]:
    async def one(i: int):
        await asyncio.sleep(i * STAGGER)
        return await run(engine, f"s{i}")
    results = await asyncio.gather(*(one(i) for i in range(N)))
    assert all(r.finished and r.pauses == N_PAUSES for r in results), "a request stalled"
    return {r.request_id: _flat(r) for r in results}


def build(preload: bool) -> AsyncLLM:
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="facebook/opt-125m", enforce_eager=True, seed=0,
            gpu_memory_utilization=0.30, enable_prefix_caching=True,
            kv_transfer_config=cpu_offload_kv_transfer_config(cpu_gb=2.0),
            scheduler_cls="mars.v1.scheduler.MARSScheduler",
            additional_config=MarsConfig(
                proactive_preload=preload, preload_headroom=0.98,
            ).to_additional_config(),
        )
    )


async def main() -> None:
    eng_off = build(preload=False)
    out_off = await drive(eng_off)
    eng_off.shutdown()

    eng_on = build(preload=True)
    out_on = await drive(eng_on)
    eng_on.shutdown()

    ndiff = sum(out_off[k] != out_on.get(k) for k in out_off)
    print(f"requests={len(out_off)}  byte-identical off-vs-on: {ndiff == 0} (differing={ndiff})")
    assert set(out_off) == set(out_on), "request id mismatch"
    assert ndiff == 0, "proactive preload changed the output vs the baseline!"
    print(">>> PROACTIVE_PRELOAD_E2E_OK")


if __name__ == "__main__":
    asyncio.run(main())
