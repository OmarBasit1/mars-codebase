"""Async benchmark harness for MARS on vLLM v1.

The async re-implementation of
``mars-codebase/benchmarks/fixed_final_tput_bench_real.py`` (which drove the old
synchronous ``engine.step`` loop). It reads the collected multi-agent / ReAct
trace format: ``--workload`` is ``{log_filename: agent_invocation}`` where each
invocation is ``{name, api_time, multi_turn:[turn...]}`` and a turn is an LLM task
(``prompt_len``, ``output_len``) whose optional ``tool`` is either a simulated tool
call (``{name, api_time}``) or a nested sub-agent (``{name, api_time, multi_turn}``).
Per-agent shared system-prefix lengths come from a separate ``--agent-prefix``
file (``agent_prefix_len_dict.json``). Convert the old flat workload with
``mars.bench.convert_workload``.

Trace fidelity: each agent invocation runs as its own engine request; a sub-agent
call sleeps the client-side pre-launch delay then parks the parent (KV subject to
the MARS policy) while the child runs concurrently. A turn's generated output is
NOT reused as KV (it is rewritten as ``assistant: ...`` with tool params stripped),
so the scheduler discards the output's KV on resume and the next turn re-prefills
the prompt growth (``prompt_len[i+1]-prompt_len[i]``); ``tool_output_len`` is thus
inferred, not stored. Token ids are built so each agent's shared prefix (and an
invocation's carried-over prompt prefix) reuses the prefix cache (on by default).

Requests launch concurrently with Poisson arrivals; reports throughput /
normalized-latency / TTFT (+ an optional per-request CSV).

Run (from a neutral cwd):
    CUDA_VISIBLE_DEVICES=0 .../python -m mars.bench.run \
        --workload .../processed_log_example.json \
        --agent-prefix .../agent_prefix_len_dict.json \
        --model facebook/opt-125m --num-requests 8 --qps 8 --api-policy V \
        --max-model-len 65536
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import tempfile
import time
import zlib

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
from mars.orchestrator import (
    AgentPlan, ApiOrchestrator, ApiSegment, OrchestratorResult,
)
from mars.params import MarsApiParams
from mars.swap import cpu_offload_kv_transfer_config


def _has_subturns(tool: dict) -> bool:
    """A tool dict that carries its own turns is a sub-agent (vs. a plain tool)."""
    return isinstance(tool, dict) and "multi_turn" in tool


def load_workload(path: str, prefix_path: str = "") -> tuple[dict[str, int], list[dict]]:
    """Load a collected-trace workload into ``(agent_prefix_lens, jobs)``.

    The workload file is a dict ``{log_filename: agent_invocation}`` where each
    value is ``{name, api_time, multi_turn:[turn...]}``; ``jobs`` is its values.
    ``agent_prefix_lens`` maps the (prefixed) agent name -> shared system-prefix
    token length, read from ``prefix_path`` (default: a sibling
    ``agent_prefix_len_dict.json`` next to the workload).
    """
    workload = json.load(open(path))
    jobs = list(workload.values())
    if not prefix_path:
        cand = os.path.join(os.path.dirname(os.path.abspath(path)),
                            "agent_prefix_len_dict.json")
        prefix_path = cand if os.path.exists(cand) else ""
    prefix_lens = json.load(open(prefix_path)) if prefix_path else {}
    return prefix_lens, jobs


def _tok(stream_id: tuple, pos: int, vocab: int) -> int:
    """Deterministic dummy token id in ``[10, vocab)`` for ``(stream_id, pos)``.

    Identical ``(stream_id, pos)`` always maps to the same id (process-independent),
    so two token sequences sharing a prefix produce identical ids -> the engine's
    prefix cache reuses them. ``stream_id`` is ``("agent", name)`` for the shared
    system-prefix region (same across all invocations of that agent type) and
    ``("inv", k)`` for an invocation's unique body (stable across its own turns).
    """
    h = zlib.adler32(repr(stream_id).encode())
    span = max(2, vocab - 10)
    return 10 + ((h * 1103515245 + pos * 12345) & 0x7FFFFFFF) % span


def _agent_token_seq(name: str, prefix_len: int, total_len: int,
                     inv_idx: int, vocab: int) -> list[int]:
    """Token ids for one agent invocation: shared agent prefix + unique body.

    Positions ``[0, prefix_len)`` use the agent-shared stream (so all invocations
    of ``name`` share that prefix); positions ``[prefix_len, total_len)`` use this
    invocation's unique stream. Slicing ``seq[:prompt_len[i]]`` yields each turn's
    prompt, and earlier slices are exact prefixes of later ones.
    """
    out = []
    for pos in range(total_len):
        sid = ("agent", name) if pos < prefix_len else ("inv", inv_idx)
        out.append(_tok(sid, pos, vocab))
    return out


def build_agent_plan(
    job: dict, prefix_lens: dict[str, int], counter: list[int], *,
    api_policy: str, vocab: int, zero_api_time: bool = False,
) -> AgentPlan:
    """Map one agent invocation (``{name, api_time, multi_turn:[...]}``) to a plan.

    One segment per turn (``gen_len == output_len``). A turn's ``tool.api_time`` is
    the pause sleep (plain tool) or the sub-agent pre-launch delay; a tool carrying
    ``multi_turn`` recurses into a nested :class:`AgentPlan`. Since ``tool_output_len``
    is dropped from the trace, the tokens injected on the pause after turn ``i`` are
    the prompt growth ``prompt_len[i+1] - prompt_len[i]`` (rewritten output + tool
    output), taken as the next slice of this invocation's deterministic token
    sequence so the prefix cache reuses the carried-over prompt prefix.
    """
    name = job["name"]
    turns = job["multi_turn"]
    inv_idx = counter[0]
    counter[0] += 1

    prompt_lens = [max(1, int(t.get("prompt_len", 1))) for t in turns]
    prefix_len = int(prefix_lens.get(name, 0))
    total_len = max(prompt_lens)
    seq = _agent_token_seq(name, prefix_len, total_len, inv_idx, vocab)

    segments: list[ApiSegment] = []
    n_calls = 0  # number of tool pauses (API + sub-agent), == api_max_calls
    for i, t in enumerate(turns):
        seg = ApiSegment(gen_len=max(1, int(t["output_len"])))
        tool = t.get("tool")
        if tool is not None:
            seg.api_exec_time = float(tool.get("api_time", 0.0))
            if _has_subturns(tool):  # sub-agent: api_time = pre-launch delay
                seg.sub_agent = build_agent_plan(
                    tool, prefix_lens, counter, api_policy=api_policy,
                    vocab=vocab, zero_api_time=zero_api_time,
                )
            n_calls += 1
        # Tokens injected at the pause after turn i = the prompt growth into turn
        # i+1 (>=1). The final turn has no following turn -> no injection.
        if i + 1 < len(turns):
            delta = max(1, prompt_lens[i + 1] - prompt_lens[i])
            seg.injected_token_ids = seq[prompt_lens[i]:prompt_lens[i] + delta]
            seg.api_return_length = len(seg.injected_token_ids)
        segments.append(seg)
    # The orchestrator pauses after segments[0..n-2] only; if the final turn has a
    # tool (no following turn to inject into), append a minimal terminal segment so
    # its pause/sub-agent is still realized.
    if segments and turns[-1].get("tool") is not None:
        segments.append(ApiSegment(gen_len=1))

    if zero_api_time:
        for s in segments:
            s.api_exec_time = 0.0

    prompt_ids = seq[:prompt_lens[0]]
    mars = MarsApiParams(
        use_api_simulator=True,
        api_invoke_interval=segments[0].gen_len,
        api_return_length=segments[0].api_return_length,
        api_exec_time=segments[0].api_exec_time,
        predicted_api_exec_time=segments[0].api_exec_time,
        api_max_calls=n_calls,
        remain_length=sum(s.gen_len for s in segments[1:]),
        api_policy=api_policy,
    )
    turn_lens = [(prompt_lens[i], max(1, int(turns[i]["output_len"])))
                 for i in range(len(turns))]
    return AgentPlan(prompt_ids, segments, mars.to_extra_args(),
                     name=name, turn_lens=turn_lens)


def _generated_with_children(r: OrchestratorResult) -> int:
    """Total model-generated tokens of ``r`` plus all (recursive) sub-agents."""
    return r.total_generated + sum(_generated_with_children(c) for c in r.children)


def _dump_turns(plan: AgentPlan, result: OrchestratorResult, depth: int = 0) -> list[str]:
    """Per-turn realized-vs-trace report for one request (recurses sub-agents).

    For each turn shows the fed prompt length (initial prompt + injected deltas of
    earlier turns, with the prior decoded output discarded) vs the trace
    ``prompt_len``, and the tokens the engine generated vs the trace ``output_len``.
    """
    pad = "  " * depth
    lines = [f"{pad}{result.request_id}  {plan.name}  ({len(plan.turn_lens)} turns)",
             f"{pad}  turn | prompt_len(fed/trace) |  output(gen/trace)"]
    child_i = 0
    fed = len(plan.prompt_token_ids)  # prompt fed at turn 0
    for i, (tp, to) in enumerate(plan.turn_lens):
        gen = len(result.segment_tokens[i]) if i < len(result.segment_tokens) else 0
        pmark = "" if fed == tp else "  <-- MISMATCH"
        omark = "" if gen == to else ("  (+%d overrun)" % (gen - to) if gen > to else "  <-- SHORT")
        lines.append(f"{pad}  {i:4d} | {fed:7d} / {tp:7d}{'':6}| {gen:6d} / {to:6d}{omark}{pmark}")
        # advance the fed prompt length by this turn's injected delta (next prompt)
        if i < len(plan.segments):
            fed += len(plan.segments[i].injected_token_ids)
        seg = plan.segments[i] if i < len(plan.segments) else None
        if seg is not None and seg.sub_agent is not None and child_i < len(result.children):
            lines += _dump_turns(seg.sub_agent, result.children[child_i], depth + 1)
            child_i += 1
    return lines


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


async def main_async(args: argparse.Namespace) -> None:
    prefix_lens, jobs = load_workload(args.workload, args.agent_prefix)
    random.seed(args.seed)
    # --window W (seconds) derives the request count from the arrival rate, like
    # the original benchmark; otherwise use --num-requests directly.
    num_requests = max(1, int(args.qps * args.window)) if args.window > 0 else args.num_requests
    selected = random.choices(jobs, k=num_requests)
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
        tensor_parallel_size=args.tensor_parallel_size,
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
            demote_ondemand=not args.demote_proactive,
            demote_eager=args.demote_eager,
            starvation_avoidance=args.starvation_avoidance,
            starvation_threshold=args.starvation_threshold,
            starvation_quantum=args.starvation_quantum,
            per_req_stats_path=stats_path,
        ).to_additional_config(),
    )
    if args.hf_overrides:
        # e.g. extend context via YaRN:
        #   --hf-overrides '{"rope_parameters": {"factor": 8.0,
        #     "original_max_position_embeddings": 32768, "rope_type": "yarn"}}'
        eng_kwargs["hf_overrides"] = json.loads(args.hf_overrides)
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
        # Prefix caching is ON by default: the collected traces reuse each agent's
        # shared system prefix across invocations and the carried-over prompt prefix
        # across turns (see build_agent_plan's deterministic token sequences).
        eng_kwargs["enable_prefix_caching"] = not args.no_prefix_cache
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**eng_kwargs))

    vocab = engine.model_config.get_vocab_size()
    counter = [0]  # global invocation index -> unique body token ids per agent run
    plans = [
        build_agent_plan(job, prefix_lens, counter, api_policy=args.api_policy,
                         vocab=vocab, zero_api_time=args.zero_api_time)
        for job in selected
    ]
    # Root agent pre-launch delay (client-side, before its first LLM call).
    root_api_times = [float(job.get("api_time", 0.0)) for job in selected]
    arrivals: dict[str, float] = {}
    t0 = time.perf_counter()

    async def drive(i: int, plan: AgentPlan):
        wait = (t0 + float(offsets[i])) - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        # Client-side pre-launch delay before the root agent's first LLM call.
        if not args.zero_api_time and root_api_times[i] > 0:
            await asyncio.sleep(root_api_times[i])
        arrivals[str(i)] = time.perf_counter()
        orch = ApiOrchestrator(engine, api_result_token=args.api_token)
        # --window>0 caps each request at 2*window; the --num-requests path has no
        # window so run untimed (a 0s timeout would abort every request instantly).
        timeout = float(args.window) * 2 if args.window > 0 else None
        try:
            return await asyncio.wait_for(
                orch.run_request(
                    request_id=str(i),
                    prompt_token_ids=plan.prompt_token_ids,
                    segments=plan.segments,
                    extra_args=plan.extra_args,
                ),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return OrchestratorResult(
                request_id=str(i), finished=False, end_time=time.perf_counter()
            )

    results = await asyncio.gather(*(drive(i, p) for i, p in enumerate(plans)))
    wall = time.perf_counter() - t0
    engine.shutdown()

    if args.dump_turns and results:
        print("=" * 56)
        print("PER-TURN realized-vs-trace (request 0):")
        print("\n".join(_dump_turns(plans[0], results[0])))

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
    # Throughput counts sub-agent tokens too (each agent is a real engine sequence).
    total_gen = sum(_generated_with_children(r) for r in finished)
    norm_lat, ttft, e2e = [], [], []
    total_input = 0  # prefill tokens over finished requests (for input_tokens_per_s)
    rows = []
    for r, plan in zip(results, plans):
        arr = arrivals.get(r.request_id, r.start_time)
        e2e_i = r.end_time - arr
        ttft_i = (r.first_token_time - arr) if r.first_token_time else float("nan")
        nl_i = (
            (r.end_time - arr - r.total_pause_wait) / r.total_generated
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
        pr_ttft_last = pr_ttfts[-1] if pr_ttfts else float("nan")  # last realized resume
        # Input (prefill) tokens: initial prompt + the API-return tokens injected
        # at each realized resume (>=1 per resume, matching the orchestrator's
        # empty-prompt guard). r.pauses == number of realized API resumes.
        injected = sum(max(1, plan.segments[i].api_return_length) for i in range(r.pauses))
        input_toks = len(plan.prompt_token_ids) + injected
        if r.finished:
            total_input += input_toks
        rows.append([r.request_id, r.finished, r.pauses, r.total_generated,
                     f"{r.total_pause_wait:.4f}", f"{e2e_i:.4f}", f"{ttft_i:.4f}", f"{nl_i:.6f}",
                     mars_rec.get("policy", ""),
                     mars_rec.get("arrival_strategy", ""),
                     mars_rec.get("swap_reloads", ""),
                     mars_rec.get("demotions", ""),
                     f"{pr_ttft_mean:.4f}" if not np.isnan(pr_ttft_mean) else "",
                     input_toks,
                     f"{pr_ttft_last:.4f}" if not np.isnan(pr_ttft_last) else ""])

    print("=" * 56)
    print(f"policy={args.api_policy} policy_config={args.policy_config} "
          f"swap={args.swap} chunk_fill={args.chunk_fill} demote={not args.no_demote} "
          f"eager={args.demote_eager}")
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
                        "demotions", "post_resume_ttft_s",
                        "input_tokens", "last_resume_ttft_s"])
            w.writerows(rows)
        print(f"wrote per-request CSV -> {args.csv}")
        # Run-level summary sidecar: the per-request CSV has no wall time or
        # offered rate, so the parser reads those from here. Stem matches the CSV
        # ("{tag}_{qps}.csv" -> "{tag}_{qps}.summary.json"); .json is ignored by
        # the parser's *.csv glob.
        summary_path = (args.csv[:-4] if args.csv.endswith(".csv") else args.csv) + ".summary.json"
        with open(summary_path, "w") as sf:
            json.dump({
                "qps_workload": args.qps,
                "window": args.window,
                "wall_s": wall,
                "requests": len(results),
                "finished": len(finished),
                "output_tokens": total_gen,
                "input_tokens": total_input,
                "policy": args.api_policy,
                "policy_config": args.policy_config,
            }, sf, indent=2)
        print(f"wrote run summary -> {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MARS async benchmark (vLLM v1).")
    ap.add_argument("--workload", required=True,
                    help="trace JSON: {log_filename: {name, api_time, multi_turn:[...]}}")
    ap.add_argument("--agent-prefix", default="",
                    help="agent_prefix_len_dict.json (agent name -> shared system-prefix "
                         "token length); default: sibling file next to --workload")
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
                    help="disable dynamic memory-pressure demotion entirely")
    ap.add_argument("--demote-proactive", action="store_true",
                    help="use the original per-step proactive demotion instead of "
                         "the default on-demand (lazy/minimal) demotion (ablation)")
    ap.add_argument("--demote-eager", action="store_true",
                    help="V policy: free recompute/swap-classified KV AT THE PAUSE "
                         "(eager, like D/S) instead of preserving + demoting lazily")
    ap.add_argument("--starvation-avoidance", action="store_true")
    ap.add_argument("--starvation-threshold", type=int, default=100)
    ap.add_argument("--starvation-quantum", type=int, default=100000)
    ap.add_argument("--sync-scheduling", action="store_true",
                    help="force synchronous scheduling (async_scheduling=False)")
    ap.add_argument("--zero-api-time", action="store_true",
                    help="zero every segment's API wait (keep pause/resume + call "
                         "count; removes only idle wall-clock) -- for max-RPS probing")
    ap.add_argument("--swap", action="store_true", help="enable SimpleCPUOffloadConnector")
    ap.add_argument("--cpu-gb", type=float, default=4.0)
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="disable prefix caching (on by default: the traces reuse each "
                         "agent's shared system prefix and the carried-over prompt prefix)")
    ap.add_argument("--load-format", default="auto", help="e.g. 'dummy' for no download")
    ap.add_argument("--max-model-len", type=int, default=2048,
                    help="raise this for long traces (prompt_len can be tens of thousands)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="shard the model across N GPUs (set CUDA_VISIBLE_DEVICES too)")
    ap.add_argument("--hf-overrides", default="",
                    help="JSON dict passed as AsyncEngineArgs(hf_overrides=...), e.g. YaRN "
                         "rope scaling to extend context beyond the model's native limit")
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="max concurrent sequences (vLLM default auto-sizes to ~128)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api-token", type=int, default=5000)
    ap.add_argument("--csv", default="")
    ap.add_argument("--dump-turns", action="store_true",
                    help="print a per-turn realized-vs-trace table for request 0")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
