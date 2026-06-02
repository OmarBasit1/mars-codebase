"""Minimal async API-server skeleton for MARS over vLLM v1 ``AsyncLLM``.

Phase-1 skeleton. Exposes:
  * ``GET  /health``
  * ``POST /generate`` -- run one API-augmented request through the built-in
    simulator via :class:`mars.orchestrator.ApiOrchestrator`, returning the
    generated token ids per segment.

The external-tool path (a real tool POSTing its result to resume a paused
request) is intentionally left for a later phase; the benchmark harness
(Phase 7) drives ``ApiOrchestrator`` directly rather than over HTTP.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from mars.orchestrator import ApiOrchestrator, ApiSegment


class SegmentSpec(BaseModel):
    gen_len: int
    api_exec_time: float = 0.0
    api_return_length: int = 0


class GenerateRequest(BaseModel):
    request_id: str
    prompt_token_ids: list[int]
    segments: list[SegmentSpec]
    api_result_token: int = 0
    temperature: float = 0.0


def build_app(engine: Any) -> Any:
    """Build a FastAPI app bound to an ``AsyncLLM`` engine.

    Args:
        engine: A vLLM v1 ``AsyncLLM`` instance.

    Returns:
        A configured ``fastapi.FastAPI`` application.
    """
    from fastapi import FastAPI

    app = FastAPI(title="MARS (vLLM v1) API server", version="0.0.1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> dict[str, Any]:
        orch = ApiOrchestrator(engine, api_result_token=req.api_result_token)
        segments = [
            ApiSegment(
                gen_len=s.gen_len,
                api_exec_time=s.api_exec_time,
                api_return_length=s.api_return_length,
            )
            for s in req.segments
        ]
        res = await orch.run_request(
            request_id=req.request_id,
            prompt_token_ids=req.prompt_token_ids,
            segments=segments,
            temperature=req.temperature,
        )
        return {
            "request_id": res.request_id,
            "finished": res.finished,
            "pauses": res.pauses,
            "total_generated": res.total_generated,
            "segment_tokens": res.segment_tokens,
        }

    return app
