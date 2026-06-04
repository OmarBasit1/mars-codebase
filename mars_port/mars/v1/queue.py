"""MARS waiting-queue orderings (SJF + cost-based V2) with starvation boosting.

The original MARS chose the waiting order via ``policy_config``
(``fcfs`` / ``sjf`` / ``V2``). vLLM v1 exposes a pluggable ``RequestQueue``; this
module provides:
  * :class:`MARSRequestQueue` — Shortest-Job-First on MARS ``remain_length``.
  * :class:`V2RequestQueue` — the cost-based ``V2`` ordering (memory-time waste).
FCFS uses vLLM's native queue. Both MARS queues consult a shared ``starving`` set
(owned by ``MARSScheduler``) so starvation-boosted requests sort to the front
(key ``-inf``) — this supports the paper's ``V + V2 + starvation`` config.

Note: the V2 insertion key uses ``running_batch=0`` as a cheap placeholder; the
scheduler calls ``rekey`` every step with the live ``running_batch`` (read from
``vllm_config.scheduler_config.max_num_batched_tokens``) so the ordering is always
up-to-date. See COMPARISON.md.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Iterable, Iterator

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request

from mars.params import MarsApiParams

_BS = 16  # block-size approximation for queue-level scoring


def _sjf_key(request: Request) -> float:
    """Shortest remaining output length first; unknown -> treated as longest."""
    mp = MarsApiParams.from_sampling_params(request.sampling_params)
    if mp is not None and mp.remain_length:
        return float(mp.remain_length)
    return float("inf")


def _mem_time(num_blocks: int, running_batch: int, max_ragged_batch: int) -> float:
    """Memory-time of computing ``num_blocks`` (ported from policy.py V2).

    ``max_ragged_batch`` is the per-step token budget where compute saturates
    (read from ``vllm_config.scheduler_config.max_num_batched_tokens``).
    """
    c_h = max(max_ragged_batch - running_batch, 1)
    n = max((_BS * num_blocks + c_h - 1) // c_h, 1)
    f_s = (0.1 * max_ragged_batch + 10) / 1000
    return f_s * (1 + n) * n / 2 * c_h


def _make_v2_key(max_ragged_batch: int) -> Callable[[Request], float]:
    """Return a V2 insertion-key function for the given ``max_ragged_batch``.

    Uses the ``preserve``-strategy score with ``running_batch=0`` as a cheap
    placeholder; the scheduler overwrites this with a live rekey each step.
    """
    def _v2_key(request: Request) -> float:
        mp = MarsApiParams.from_sampling_params(request.sampling_params)
        if mp is None:
            return float("inf")
        prompt_len = getattr(request, "num_prompt_tokens", 0) or 0
        before_blocks = (prompt_len + mp.predicted_api_invoke_interval + _BS - 1) // _BS
        after_blocks = (mp.predicted_api_invoke_interval + mp.api_return_length + _BS - 1) // _BS
        before = _mem_time(before_blocks, 0, max_ragged_batch)
        after = _mem_time(after_blocks, 0, max_ragged_batch)
        api_memory = before_blocks * _BS * mp.predicted_api_exec_time
        return before + api_memory + after
    return _v2_key


class _MarsHeapQueue(RequestQueue):
    """A heap queue keyed by ``key_fn`` with a shared ``starving`` front set."""

    def __init__(
        self, key_fn: Callable[[Request], float], starving: set[str] | None = None
    ) -> None:
        self._key_fn = key_fn
        self._starving = starving if starving is not None else set()
        self._heap: list[tuple[float, int, Request]] = []
        self._counter = itertools.count()

    def _key(self, request: Request) -> float:
        if request.request_id in self._starving:
            return float("-inf")  # starvation-boosted -> front
        return self._key_fn(request)

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, (self._key(request), next(self._counter), request))

    def rekey(self, key_fn: Callable[[Request], float]) -> None:
        """Recompute every entry's key with ``key_fn`` and re-heapify (O(n)).

        Each insertion ``counter`` is preserved (stable ties) and the starving
        ``-inf`` override is re-applied via :meth:`_key`. The scheduler calls this
        once per step to re-rank V2 with the live ``running_batch`` — faithful to
        the original ``sort_by_priority(running_batch=live)`` each step.
        """
        self._key_fn = key_fn
        self._heap = [(self._key(req), counter, req) for _, counter, req in self._heap]
        heapq.heapify(self._heap)

    def peek_key(self) -> float:
        """Return the smallest key in the heap without popping (O(1)).

        Used by ``_select_waiting_queue_for_scheduling`` to compare head keys
        of ``self.waiting`` and ``self.skipped_waiting`` for combined ordering.
        """
        if not self._heap:
            return float("inf")
        return self._heap[0][0]

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty queue")
        return heapq.heappop(self._heap)[2]

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from empty queue")
        return self._heap[0][2]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)  # ordering is by key (or -inf if starving)

    def prepend_requests(self, requests: "RequestQueue") -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self._heap = [e for e in self._heap if e[2] is not request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = set(requests)
        self._heap = [e for e in self._heap if e[2] not in to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        for _, _, request in sorted(self._heap):
            yield request

    def iter_unsorted(self) -> Iterator[Request]:
        """Yield requests in heap (arbitrary) order — O(n), no sort.

        For hot-path scans that don't need priority order (e.g. the per-step
        starvation pass), this avoids the O(n log n) ``sorted()`` in ``__iter__``.
        Iterates a snapshot so callers may mutate the queue while iterating.
        """
        for _, _, request in list(self._heap):
            yield request


class MARSRequestQueue(_MarsHeapQueue):
    """Shortest-Job-First queue (keyed on MARS ``remain_length``)."""

    def __init__(self, starving: set[str] | None = None) -> None:
        super().__init__(_sjf_key, starving)


class V2RequestQueue(_MarsHeapQueue):
    """Cost-based ``V2`` queue (memory-time waste score; lower first).

    ``max_ragged_batch`` is read from ``vllm_config.scheduler_config.max_num_batched_tokens``
    and passed in by the scheduler; it sets the per-step token saturation point for
    the insertion-time placeholder score (overwritten by rekey each step anyway).
    """

    def __init__(self, starving: set[str] | None = None, max_ragged_batch: int = 384) -> None:
        super().__init__(_make_v2_key(max_ragged_batch), starving)


def make_mars_queue(
    policy_config: str,
    starving: set[str] | None = None,
    max_ragged_batch: int = 384,
) -> RequestQueue | None:
    """Return a MARS queue for ``policy_config``, or ``None`` to keep native FCFS."""
    if policy_config == "sjf":
        return MARSRequestQueue(starving)
    if policy_config == "V2":
        return V2RequestQueue(starving, max_ragged_batch)
    return None
