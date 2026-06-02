"""MARS waiting-queue orderings (Shortest-Job-First).

The original MARS selected the waiting-queue order via ``policy_config``
(``fcfs`` / ``sjf`` / ``V2``). vLLM v1 exposes a pluggable ``RequestQueue``; we
provide SJF here keyed on the MARS ``remain_length`` (predicted remaining output
length). ``MARSScheduler`` swaps it in for ``self.waiting`` when
``MarsConfig.policy_config == 'sjf'``. FCFS uses the native queue; the cost-based
``V2`` ordering lands with the solver in Phase 6.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable, Iterator

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request

from mars.params import MarsApiParams


def _sjf_key(request: Request) -> float:
    """Shortest remaining output length first; unknown -> treated as longest."""
    mp = MarsApiParams.from_sampling_params(request.sampling_params)
    if mp is not None and mp.remain_length:
        return float(mp.remain_length)
    return float("inf")


class MARSRequestQueue(RequestQueue):
    """Shortest-Job-First waiting queue keyed on MARS ``remain_length``.

    Like ``PriorityRequestQueue``, "prepend" has no special meaning (ordering is
    by key, then insertion order for stable ties). Heap entries are
    ``(key, seq, request)`` so requests are never compared directly.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Request]] = []
        self._counter = itertools.count()

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, (_sjf_key(request), next(self._counter), request))

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty queue")
        return heapq.heappop(self._heap)[2]

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from empty queue")
        return self._heap[0][2]

    def prepend_request(self, request: Request) -> None:
        # SJF has no front; insert by key (stable for equal keys).
        self.add_request(request)

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
        # Yield in SJF order without mutating the heap (tuples order by
        # (key, seq); the unique seq prevents comparing Request objects).
        for _, _, request in sorted(self._heap):
            yield request
