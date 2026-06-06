"""Orchestrator end-to-end: pause / API-inject / resume correctness.

Drives the SAME request twice, differing ONLY in the injected "API result"
tokens, and asserts:
  * the request pauses the expected number of times and finishes;
  * total generated tokens == sum of segment lengths (each segment is exact);
  * segment 0 (before any injection) is identical across the two runs
    (deterministic greedy decode from the same prompt);
  * segment 1 (right after the first injection) DIFFERS across the two runs —
    proving the injected API tokens are fed in at the right position and
    actually condition the resumed generation.

Run from a neutral cwd:
    cd /tmp && CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/test_orchestrator.py
"""

import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams

GEN = 16          # tokens per segment (api_invoke_interval)
RET = 8           # API return length
N_PAUSES = 2      # api_max_calls


async def main() -> None:
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="facebook/opt-125m",
            enforce_eager=True,
            gpu_memory_utilization=0.30,
            scheduler_cls="mars.v1.scheduler.MARSScheduler",
        )
    )

    prompt = list(range(20, 40))  # 20 arbitrary valid prompt token ids
    segments = [
        ApiSegment(gen_len=GEN, api_exec_time=0.1, api_return_length=RET),
        ApiSegment(gen_len=GEN, api_exec_time=0.1, api_return_length=RET),
        ApiSegment(gen_len=GEN),  # final segment, no API call
    ]
    extra = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=GEN,
        api_return_length=RET,
        api_exec_time=0.1,
        api_max_calls=N_PAUSES,
        remain_length=GEN,
    ).to_extra_args()

    # Two runs differing only in the injected API token id.
    rA = await ApiOrchestrator(engine, api_result_token=5000).run_request(
        request_id="mars_A", prompt_token_ids=prompt, segments=segments, extra_args=extra
    )
    rB = await ApiOrchestrator(engine, api_result_token=9000).run_request(
        request_id="mars_B", prompt_token_ids=prompt, segments=segments, extra_args=extra
    )

    print("A:", rA.finished, "pauses=", rA.pauses, "gen=", rA.total_generated)
    print("B:", rB.finished, "pauses=", rB.pauses, "gen=", rB.total_generated)
    print("seg0 A==B:", rA.segment_tokens[0] == rB.segment_tokens[0])
    print("seg1 A!=B:", rA.segment_tokens[1] != rB.segment_tokens[1])

    expected_total = GEN * len(segments)
    assert rA.finished and rB.finished, "request did not finish"
    assert rA.pauses == N_PAUSES and rB.pauses == N_PAUSES, "wrong pause count"
    assert rA.total_generated == expected_total, (rA.total_generated, expected_total)
    assert rB.total_generated == expected_total, (rB.total_generated, expected_total)
    # Pre-injection segment is deterministic & identical; post-injection diverges.
    assert rA.segment_tokens[0] == rB.segment_tokens[0], "seg0 should be identical"
    assert rA.segment_tokens[1] != rB.segment_tokens[1], (
        "seg1 identical => injected API tokens did NOT condition the resume"
    )
    print(">>> ORCHESTRATOR_E2E_OK")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
