"""Async pause/resume orchestration for MARS on vLLM v1.

Replaces the old *synchronous* ``resume_request`` loop
(mars-codebase/benchmarks/fixed_final_tput_bench_real.py) with an async driver
built on vLLM v1's **native resumable streaming**: a request generates a
segment, the engine parks it awaiting more input, the orchestrator performs the
(simulated) API call, then injects the API-return tokens as the next streaming
chunk — the request resumes, conditioned on the injected tokens.

Faithful to the original simulator: the pause is a *token-interval* boundary
(``api_invoke_interval`` == the workload's ``completion_tokens``), and the API
response is ``api_return_length`` dummy tokens delivered after ``api_exec_time``
seconds. This needs **no** vLLM core edits and (in Phase 2) no scheduler
override — pause/inject/resume is entirely native. Later phases add the
KV-cache policy decision at the pause point inside ``MARSScheduler``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from vllm import SamplingParams, TokensPrompt
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind


@dataclass
class ApiSegment:
    """One generation segment of an API-augmented request.

    Attributes:
        gen_len: #tokens to generate in this segment before pausing (the MARS
            ``api_invoke_interval`` / workload ``completion_tokens``).
        api_exec_time: seconds to wait for the API after this segment
            (``0`` => no API call follows; used for the final segment).
        api_return_length: #tokens the API injects before the next segment.
    """

    gen_len: int
    api_exec_time: float = 0.0
    api_return_length: int = 0


@dataclass
class OrchestratorResult:
    """Outcome of driving one API-augmented request.

    Attributes:
        request_id: The driven request's id.
        finished: True once the request completed (input stream closed).
        pauses: Number of API pauses observed (== ``len(segments) - 1``).
        total_generated: Total model-generated tokens across all segments
            (excludes injected API tokens, which are input).
        segment_tokens: Per-segment lists of generated token ids.
    """

    request_id: str
    finished: bool = False
    pauses: int = 0
    total_generated: int = 0
    segment_tokens: list[list[int]] = field(default_factory=list)
    # Timing (perf_counter seconds), for the benchmark harness.
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    # Per-pause timing (measures post-resume reload latency on the critical path):
    #   resume_times[i]  = perf_counter() right after API inject for pause i
    #   post_resume_first_token_times[i] = perf_counter() when first token
    #       of segment i+1 arrives after the resume.
    # post_resume_ttft[i] = post_resume_first_token_times[i] - resume_times[i]
    # is the per-pause latency from API-return to first generated token.
    resume_times: list[float] = field(default_factory=list)
    post_resume_first_token_times: list[float] = field(default_factory=list)


class ApiOrchestrator:
    """Drives API-augmented requests on an ``AsyncLLM`` via resumable streaming."""

    def __init__(self, engine: Any, *, api_result_token: int = 0) -> None:
        """
        Args:
            engine: A vLLM v1 ``AsyncLLM`` instance.
            api_result_token: Token id used to fill simulated API responses
                (the original MARS simulator injects token id 0).
        """
        self.engine = engine
        self.api_result_token = api_result_token

    def _segment_params(
        self, gen_len: int, temperature: float, extra_args: dict[str, Any] | None
    ) -> SamplingParams:
        """Sampling params that make a segment generate exactly ``gen_len`` tokens.

        ``ignore_eos`` + ``max_tokens=gen_len`` + no stop strings means the only
        stop condition is length, so each segment is exactly ``gen_len`` tokens —
        which makes the token-counting segment-boundary detection deterministic.
        ``RequestOutputKind.DELTA`` so each output carries only the new tokens.
        """
        return SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=gen_len,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
            extra_args=extra_args,
        )

    async def run_request(
        self,
        *,
        request_id: str,
        prompt_token_ids: list[int],
        segments: list[ApiSegment],
        temperature: float = 0.0,
        extra_args: dict[str, Any] | None = None,
        simulate: bool = True,
    ) -> OrchestratorResult:
        """Drive one request through its API segments and return the outcome.

        Args:
            request_id: Unique id for this request.
            prompt_token_ids: Initial prompt token ids.
            segments: The segment plan; ``segments[-1]`` is the final segment
                (typically ``api_exec_time == 0``).
            temperature: Decode temperature (0 => greedy/deterministic).
            extra_args: ``SamplingParams.extra_args`` to attach to every segment
                (e.g. ``MarsApiParams.to_extra_args()``).
            simulate: When True, sleep ``api_exec_time`` to emulate API latency.

        Returns:
            An :class:`OrchestratorResult`.

        Raises:
            ValueError: If ``segments`` is empty.
        """
        if not segments:
            raise ValueError("segments must be non-empty")
        n = len(segments)
        # Output loop -> input generator handshake: signalled when a segment
        # completes (and the engine has parked the request awaiting more input).
        proceed: asyncio.Queue[bool] = asyncio.Queue()
        result = OrchestratorResult(
            request_id=request_id, segment_tokens=[[] for _ in range(n)]
        )
        result.start_time = time.perf_counter()

        async def input_stream():
            # Initial segment: the real prompt.
            yield StreamingInput(
                prompt=TokensPrompt(prompt_token_ids=list(prompt_token_ids)),
                sampling_params=self._segment_params(
                    segments[0].gen_len, temperature, extra_args
                ),
            )
            for i in range(n - 1):
                await proceed.get()  # segment i finished + request parked
                seg = segments[i]
                if simulate and seg.api_exec_time > 0:
                    await asyncio.sleep(seg.api_exec_time)  # simulated API latency
                # At least one token: the engine rejects empty prompts, so an
                # API that returns nothing still injects a single resume marker.
                api_tokens = [self.api_result_token] * max(1, seg.api_return_length)
                result.resume_times.append(time.perf_counter())
                result.post_resume_first_token_times.append(0.0)
                # Inject the API result + params for the next segment; the engine
                # folds prior output into the prompt, appends these, and resumes.
                yield StreamingInput(
                    prompt=TokensPrompt(prompt_token_ids=api_tokens),
                    sampling_params=self._segment_params(
                        segments[i + 1].gen_len, temperature, extra_args
                    ),
                )
            # Returning closes the stream -> engine finalizes the request.

        base_sp = self._segment_params(segments[0].gen_len, temperature, extra_args)
        seg_idx = 0
        seg_count = 0
        async for out in self.engine.generate(input_stream(), base_sp, request_id):
            if out.outputs:
                toks = list(out.outputs[0].token_ids)  # DELTA: new tokens only
                now = time.perf_counter()
                if toks and result.first_token_time == 0.0:
                    result.first_token_time = now
                # Capture first token after each resume (seg_idx > 0 and the list
                # slot is still 0.0 means we just resumed this segment).
                if (toks and seg_idx > 0
                        and seg_idx <= len(result.post_resume_first_token_times)
                        and result.post_resume_first_token_times[seg_idx - 1] == 0.0):
                    result.post_resume_first_token_times[seg_idx - 1] = now
                result.segment_tokens[seg_idx].extend(toks)
                seg_count += len(toks)
                result.total_generated += len(toks)
            # A segment is done once we've seen its full gen_len of tokens; the
            # engine has stopped it (length cap) and parked it for more input.
            while seg_idx < n - 1 and seg_count >= segments[seg_idx].gen_len:
                seg_count -= segments[seg_idx].gen_len
                seg_idx += 1
                result.pauses += 1
                proceed.put_nowait(True)
            if out.finished:
                result.finished = True
        result.end_time = time.perf_counter()
        return result
