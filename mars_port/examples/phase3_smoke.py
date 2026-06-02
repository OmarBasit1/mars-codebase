"""Phase 3 smoke test: per-pause KV policy — Preserve vs Recompute.

Runs the SAME 2-pause request under api_policy='P' (Preserve) and 'D'
(Recompute) on one engine and asserts:
  * both finish with the expected pause count and token count;
  * the generated token streams are IDENTICAL — the KV policy is a
    performance/memory choice, NOT a correctness change (Recompute must
    reproduce exactly what Preserve produced from the cached KV);
  * the scheduler counted 2 preserve-pauses for 'P' and 2 recompute-frees for
    'D' (mechanism actually differed) — surfaced via the '[MARS] pause ...'
    log lines, which the bash wrapper greps.

Prefix caching is disabled so the two requests don't share cached KV, making
the Preserve-vs-Recompute behavior unambiguous.
"""

import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.orchestrator import ApiOrchestrator, ApiSegment
from mars.params import MarsApiParams

GEN, RET, N_PAUSES = 16, 8, 2


async def run(engine: AsyncLLM, rid: str, policy: str):
    extra = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=GEN,
        api_return_length=RET,
        api_exec_time=0.1,
        api_max_calls=N_PAUSES,
        remain_length=GEN,
        api_policy=policy,
    ).to_extra_args()
    segments = [
        ApiSegment(gen_len=GEN, api_exec_time=0.1, api_return_length=RET),
        ApiSegment(gen_len=GEN, api_exec_time=0.1, api_return_length=RET),
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
            enable_prefix_caching=False,
            scheduler_cls="mars.v1.scheduler.MARSScheduler",
        )
    )
    rP = await run(engine, "preserve_req", "P")
    rD = await run(engine, "recompute_req", "D")

    fP, fD = _flat(rP), _flat(rD)
    n_diff = sum(1 for a, b in zip(fP, fD) if a != b) + abs(len(fP) - len(fD))
    print("P finished/pauses/gen:", rP.finished, rP.pauses, rP.total_generated)
    print("D finished/pauses/gen:", rD.finished, rD.pauses, rD.total_generated)
    print("identical outputs:", fP == fD, "(differing tokens:", n_diff, ")")

    assert rP.finished and rD.finished, "request did not finish"
    assert rP.pauses == N_PAUSES and rD.pauses == N_PAUSES, "wrong pause count"
    assert rP.total_generated == rD.total_generated == GEN * 3, "wrong token count"
    assert fP == fD, f"Preserve vs Recompute changed the output ({n_diff} tokens differ)!"
    print(">>> PHASE3_PRESERVE_RECOMPUTE_OK")
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
