"""Async benchmark harness for MARS on vLLM v1.

The async re-implementation of
``mars-codebase/benchmarks/fixed_final_tput_bench_real.py`` (which drove the old
synchronous ``engine.step`` loop). It reads the original workload JSON format
(per request a list of segments ``{prompt_tokens, completion_tokens, api_time,
api_token_length}``), launches the requests concurrently with Poisson arrivals
via ``AsyncLLM`` + ``ApiOrchestrator`` under a chosen MARS policy, and reports
throughput / normalized-latency / TTFT metrics (+ an optional per-request CSV).

Run (from a neutral cwd):
    CUDA_VISIBLE_DEVICES=0 .../python -m mars.bench.run \
        --workload .../diverse_oneapi_merged_exp_uniform.json \
        --model facebook/opt-125m --num-requests 8 --qps 8 --api-policy V
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import tempfile
import time
from dataclasses import dataclass

# Drop cwd from sys.path BEFORE importing vllm: launching from mars-codebase
# (which still contains the old vendored vLLM 0.2.0 under vllm/) would otherwise
# shadow the installed vLLM. Must run before the vllm imports below.
import os
import sys

sys.path = [p for p in sys.path if p not in ("", os.getcwd())]

import numpy as np

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from mars.config import MarsConfig
from mars.orchestrator import ApiOrchestrator, ApiSegment, OrchestratorResult
from mars.params import MarsApiParams
from mars.swap import cpu_offload_kv_transfer_config


@dataclass
class RequestPlan:
    prompt_token_ids: list[int]
    segments: list[ApiSegment]
    mars: MarsApiParams
    api_wait: float  # total simulated API latency (for normalized latency)


def build_request_plan(
    segments_json: list[dict], *, api_policy: str, max_model_len: int
) -> RequestPlan:
    """Map one workload request (list of segments) to a runnable plan."""
    seg0 = segments_json[0]
    api_segments = [
        ApiSegment(
            gen_len=max(1, int(s.get("completion_tokens", 1))),
            api_exec_time=float(s.get("api_time", 0.0)),
            api_return_length=int(s.get("api_token_length", 0)),
        )
        for s in segments_json
    ]
    # The orchestrator pauses after segments[0..n-2]; api params of the last
    # segment are unused. api_wait = simulated latency of the realized pauses.
    api_wait = sum(s.api_exec_time for s in api_segments[:-1] if s.api_exec_time > 0)
    n_calls = sum(1 for s in api_segments[:-1] if s.api_exec_time > 0)

    out_total = sum(s.gen_len for s in api_segments)
    prompt_tokens = int(seg0.get("prompt_tokens", 1))
    prompt_tokens = max(1, min(prompt_tokens, max_model_len - out_total - 1))
    prompt_ids = list(range(10, 10 + prompt_tokens))

    mars = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=api_segments[0].gen_len,
        api_return_length=api_segments[0].api_return_length,
        api_exec_time=api_segments[0].api_exec_time,
        predicted_api_exec_time=api_segments[0].api_exec_time,
        api_max_calls=n_calls,
        remain_length=sum(s.gen_len for s in api_segments[1:]),
        api_policy=api_policy,
    )
    return RequestPlan(prompt_ids, api_segments, mars, api_wait)


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


async def main_async(args: argparse.Namespace) -> None:
    workload = json.load(open(args.workload))
    keys = list(workload.keys())
    random.seed(args.seed)
    # --window W (seconds) derives the request count from the arrival rate, like
    # the original benchmark; otherwise use --num-requests directly.
    num_requests = max(1, int(args.qps * args.window)) if args.window > 0 else args.num_requests
    selected = random.choices(keys, k=num_requests)
    rng = np.random.default_rng(args.seed)
    offsets = np.cumsum(rng.exponential(1.0 / args.qps, size=num_requests))

    # Sidecar JSONL: the MARS scheduler writes one record per finished request
    # (policy, arrival_strategy, swap_reloads).  Read after engine.shutdown().
    _stats_fd, stats_path = tempfile.mkstemp(prefix="mars_req_stats_", suffix=".jsonl")
    os.close(_stats_fd)

    eng_kwargs = dict(
        model=args.model,
        enforce_eager=True,
        seed=args.seed,  # deterministic engine sampling => repeatable per (qps, seed)
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        load_format=args.load_format,
        disable_hybrid_kv_cache_manager=True,
        async_scheduling=not args.sync_scheduling,
        scheduler_cls="mars.v1.scheduler.MARSScheduler",
        additional_config=MarsConfig(
            api_policy=args.api_policy,
            policy_config=args.policy_config,
            chunk_fill=args.chunk_fill,
            chunk_size=args.chunk_size,
            demote_paused=not args.no_demote,
            starvation_avoidance=args.starvation_avoidance,
            starvation_threshold=args.starvation_threshold,
            starvation_quantum=args.starvation_quantum,
            per_req_stats_path=stats_path,
        ).to_additional_config(),
    )
    want_swap = args.swap
    if want_swap:
        # vLLM's SimpleCPUOffloadConnector is unsupported for hybrid (Mamba/SSM)
        # models -- the scheduler asserts "External KV connector is not verified"
        # in _mamba_block_aligned_split. Detect that cheaply (config only, no
        # weights) and run swap-free (swap policies degrade to recompute).
        try:
            probe = AsyncEngineArgs(
                model=args.model, max_model_len=args.max_model_len,
                load_format=args.load_format,
            ).create_engine_config()
            if probe.model_config.is_hybrid:
                print(
                    f"WARNING: '{args.model}' is a hybrid (Mamba) model; the "
                    f"CPU-offload connector is unsupported -> running SWAP-FREE "
                    f"(swap policies degrade to recompute)."
                )
                want_swap = False
        except Exception as e:  # be permissive -- worst case the engine errors
            print(f"WARNING: hybrid-model probe failed ({e}); proceeding with --swap.")
    if want_swap:
        eng_kwargs["enable_prefix_caching"] = True
        eng_kwargs["kv_transfer_config"] = cpu_offload_kv_transfer_config(args.cpu_gb)
    else:
        eng_kwargs["enable_prefix_caching"] = args.prefix_cache
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**eng_kwargs))

    plans = [
        build_request_plan(workload[k], api_policy=args.api_policy, max_model_len=args.max_model_len)
        for k in selected
    ]
    arrivals: dict[str, float] = {}
    t0 = time.perf_counter()

    async def drive(i: int, plan: RequestPlan):
        wait = (t0 + float(offsets[i])) - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        arrivals[str(i)] = time.perf_counter()
        orch = ApiOrchestrator(engine, api_result_token=args.api_token)
        try:
            return await asyncio.wait_for(
                orch.run_request(
                    request_id=str(i),
                    prompt_token_ids=plan.prompt_token_ids,
                    segments=plan.segments,
                    extra_args=plan.mars.to_extra_args(),
                ),
                timeout=float(args.window) * 2,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return OrchestratorResult(
                request_id=str(i), finished=False, end_time=time.perf_counter()
            )

    results = await asyncio.gather(*(drive(i, p) for i, p in enumerate(plans)))
    wall = time.perf_counter() - t0
    engine.shutdown()

    # Read per-request MARS stats written by the scheduler into the sidecar JSONL.
    req_mars: dict[str, dict] = {}
    try:
        with open(stats_path) as _sf:
            for line in _sf:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    req_mars[rec["request_id"]] = rec
    except Exception:
        pass
    finally:
        try:
            os.unlink(stats_path)
        except Exception:
            pass

    # --- metrics ---
    finished = [r for r in results if r.finished]
    total_gen = sum(r.total_generated for r in finished)
    norm_lat, ttft, e2e = [], [], []
    rows = []
    for r, plan in zip(results, plans):
        arr = arrivals.get(r.request_id, r.start_time)
        e2e_i = r.end_time - arr
        ttft_i = (r.first_token_time - arr) if r.first_token_time else float("nan")
        nl_i = (
            (r.end_time - arr - plan.api_wait) / r.total_generated
            if r.total_generated
            else float("nan")
        )
        if r.finished:
            e2e.append(e2e_i)
            if r.first_token_time:
                ttft.append(ttft_i)
            if r.total_generated:
                norm_lat.append(nl_i)
        mars_rec = req_mars.get(r.request_id, {})
        # post_resume_ttft: mean seconds from API-return to first token per pause.
        pr_ttfts = [
            (ft - rt)
            for rt, ft in zip(r.resume_times, r.post_resume_first_token_times)
            if ft > 0 and rt > 0
        ]
        pr_ttft_mean = float(np.mean(pr_ttfts)) if pr_ttfts else float("nan")
        rows.append([r.request_id, r.finished, r.pauses, r.total_generated,
                     f"{plan.api_wait:.4f}", f"{e2e_i:.4f}", f"{ttft_i:.4f}", f"{nl_i:.6f}",
                     mars_rec.get("policy", ""),
                     mars_rec.get("arrival_strategy", ""),
                     mars_rec.get("swap_reloads", ""),
                     mars_rec.get("demotions", ""),
                     f"{pr_ttft_mean:.4f}" if not np.isnan(pr_ttft_mean) else ""])

    print("=" * 56)
    print(f"policy={args.api_policy} policy_config={args.policy_config} "
          f"swap={args.swap} chunk_fill={args.chunk_fill} demote={not args.no_demote}")
    print(f"requests: {len(results)}  finished: {len(finished)}  wall: {wall:.2f}s")
    print(f"output tokens: {total_gen}  throughput: {total_gen / wall:.1f} tok/s, "
          f"{len(finished) / wall:.3f} req/s")
    print(f"normalized latency (ms/tok): mean={np.mean(norm_lat) * 1000:.2f} "
          f"p50={_pct(norm_lat, 50) * 1000:.2f} p99={_pct(norm_lat, 99) * 1000:.2f}")
    print(f"TTFT (ms): mean={np.mean(ttft) * 1000:.1f} p99={_pct(ttft, 99) * 1000:.1f}")
    print(f"E2E (ms):  mean={np.mean(e2e) * 1000:.1f} p99={_pct(e2e, 99) * 1000:.1f}")
    print("=" * 56)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["request_id", "finished", "pauses", "gen_tokens",
                        "api_wait_s", "e2e_s", "ttft_s", "norm_lat_s_per_tok",
                        "kv_policy", "arrival_strategy", "swap_reloads",
                        "demotions", "post_resume_ttft_s"])
            w.writerows(rows)
        print(f"wrote per-request CSV -> {args.csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MARS async benchmark (vLLM v1).")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--window", type=float, default=0.0,
                    help="seconds; if >0, num_requests = qps*window (like the original)")
    ap.add_argument("--qps", type=float, default=8.0)
    ap.add_argument("--api-policy", default="V")
    ap.add_argument("--policy-config", default="fcfs")
    ap.add_argument("--chunk-fill", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument("--no-demote", action="store_true",
                    help="disable dynamic memory-pressure demotion (on by default)")
    ap.add_argument("--starvation-avoidance", action="store_true")
    ap.add_argument("--starvation-threshold", type=int, default=100)
    ap.add_argument("--starvation-quantum", type=int, default=100000)
    ap.add_argument("--sync-scheduling", action="store_true",
                    help="force synchronous scheduling (async_scheduling=False)")
    ap.add_argument("--swap", action="store_true", help="enable SimpleCPUOffloadConnector")
    ap.add_argument("--cpu-gb", type=float, default=4.0)
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--load-format", default="auto", help="e.g. 'dummy' for no download")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="max concurrent sequences (vLLM default auto-sizes to ~128)")
    ap.add_argument("--gpu-mem", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-token", type=int, default=5000)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
