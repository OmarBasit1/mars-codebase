"""Unit test: SJF MARSRequestQueue ordering (no GPU).

Requests are popped shortest-`remain_length`-first, with stable ordering for
equal keys, and the queue supports peek/len/bool/remove/iter.
"""

from types import SimpleNamespace

from vllm import SamplingParams

from mars.params import MarsApiParams
from mars.v1.queue import MARSRequestQueue


def mkreq(name: str, remain: int):
    sp = SamplingParams(
        max_tokens=8, extra_args=MarsApiParams(remain_length=remain).to_extra_args()
    )
    # request_id is read by the queue's starvation-aware key (_MarsHeapQueue._key).
    return SimpleNamespace(name=name, request_id=name, sampling_params=sp)


def main() -> None:
    q = MARSRequestQueue()
    assert not q and len(q) == 0
    # Insertion order a,b,c,d; remain_length 300,50,200,50.
    reqs = [mkreq("a", 300), mkreq("b", 50), mkreq("c", 200), mkreq("d", 50)]
    for r in reqs:
        q.add_request(r)
    assert q and len(q) == 4
    assert q.peek_request().name == "b"  # smallest remain_length

    # iter is non-destructive and in SJF order (ties stable: b before d).
    assert [r.name for r in q] == ["b", "d", "c", "a"], [r.name for r in q]

    # remove the 'c' request, then drain.
    q.remove_request(reqs[2])
    order = [q.pop_request().name for _ in range(len(q))]
    assert order == ["b", "d", "a"], order
    assert not q and len(q) == 0
    print("SJF pop order (c removed):", order)
    print(">>> SJF_QUEUE_OK")


if __name__ == "__main__":
    main()
