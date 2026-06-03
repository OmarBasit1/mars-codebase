"""Per-request MARS metadata, carried through vLLM via ``SamplingParams.extra_args``.

In the original MARS (a fork of vLLM 0.2.0) these were bespoke fields added
directly to ``SamplingParams`` and ``SequenceData``. On modern vLLM we attach
them as an opaque dict under ``extra_args["mars"]`` — a supported passthrough
that needs no core edit (``Request.__init__`` already reads ``extra_args``).

The scheduler / orchestrator read these back with :func:`from_sampling_params`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EXTRA_ARGS_KEY = "mars"


@dataclass
class MarsApiParams:
    """API-augmentation metadata for a single request.

    Mirrors the MARS-specific ``SamplingParams`` fields from the vLLM-0.2.0 fork
    (``mars-codebase/vllm/sampling_params.py``), plus ``api_stop_token_ids`` which
    replaces the old multi-token stop-*string* detection with token-id detection
    (the Phase-2 ``MARSScheduler.update_from_output`` hook works on token ids).

    Attributes:
        use_api_simulator: Enable the benchmark API simulator (vs. a real tool).
        api_stop_token_ids: Token id(s) that signal "pause for an API call".
        api_invoke_interval: Actual #tokens generated before the API call.
        predicted_api_invoke_interval: Predicted value used at admission time.
        api_return_length: #tokens the API response injects on resume.
        api_exec_time: Actual API/tool latency in seconds.
        predicted_api_exec_time: Predicted latency used by the cost model.
        api_max_calls: Max #API calls for this request (0 = unlimited).
        api_policy: Per-request KV policy override (``P``/``D``/``S``/``V``);
            ``None`` falls back to the engine-wide MARS config.
        strategy: Chosen KV strategy this pause (``preserve``/``swap``/``recompute``).
        remain_length: Remaining output tokens after the final API call.
        waste: Cost-model "waste" recorded for the chosen strategy.
        arrival_time: Request arrival timestamp (set by the orchestrator).
        api_call_time: Timestamp of the most recent pause (runtime bookkeeping).
    """

    use_api_simulator: bool = False
    api_stop_token_ids: list[int] = field(default_factory=list)
    api_invoke_interval: int = 128
    predicted_api_invoke_interval: int = 128
    api_return_length: int = 32
    api_exec_time: float = 1.0
    predicted_api_exec_time: float = 1.0
    api_max_calls: int = 0
    api_policy: str | None = None
    strategy: str = "recompute"
    remain_length: int = 0
    waste: float = 0.0
    arrival_time: float | None = None
    api_call_time: float | None = None

    def to_extra_args(self, extra_args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return ``extra_args`` with this object packed under ``"mars"``.

        Args:
            extra_args: An existing ``SamplingParams.extra_args`` to extend
                (e.g. one already carrying ``kv_transfer_params``). A new dict is
                created when ``None``.

        Returns:
            The (possibly newly created) ``extra_args`` dict, with key
            :data:`EXTRA_ARGS_KEY` set to this object's fields.
        """
        out = dict(extra_args) if extra_args else {}
        out[EXTRA_ARGS_KEY] = asdict(self)
        return out

    @classmethod
    def from_extra_args(cls, extra_args: dict[str, Any] | None) -> "MarsApiParams | None":
        """Reconstruct from a request's ``extra_args``.

        Args:
            extra_args: A ``SamplingParams.extra_args`` dict, or ``None``.

        Returns:
            The decoded :class:`MarsApiParams`, or ``None`` when this request
            carries no MARS metadata.
        """
        if not extra_args or EXTRA_ARGS_KEY not in extra_args:
            return None
        return cls(**extra_args[EXTRA_ARGS_KEY])

    @classmethod
    def from_sampling_params(cls, sampling_params: Any) -> "MarsApiParams | None":
        """Reconstruct from a vLLM ``SamplingParams`` (reads its ``extra_args``).

        Args:
            sampling_params: A vLLM ``SamplingParams`` instance (or anything with
                an ``extra_args`` attribute).

        Returns:
            The decoded :class:`MarsApiParams`, or ``None`` when absent.
        """
        return cls.from_extra_args(getattr(sampling_params, "extra_args", None))
