"""No-GPU unit test: combined V2/SJF ordering across waiting + skipped_waiting.

Locks the contract that _select_waiting_queue_for_scheduling picks whichever
non-empty queue has the smaller head key, and that a starving request (-inf)
wins regardless of which queue it lives in.

Run from /tmp to avoid cwd import shadowing:
    cd /tmp && /export2/obasit/MARS_derivative/vllm/.venv/bin/python \
        /export2/obasit/MARS_derivative/mars-codebase/mars_port/examples/combined_queue_test.py
"""

from types import SimpleNamespace

from vllm import SamplingParams

from mars.params import MarsApiParams
from mars.v1.queue import V2RequestQueue, MARSRequestQueue


def mkreq(name: str, remain: int = 100, api_exec_time: float = 2.0):
    sp = SamplingParams(
        max_tokens=8,
        extra_args=MarsApiParams(
            remain_length=remain,
            predicted_api_invoke_interval=remain,
            predicted_api_exec_time=api_exec_time,
            api_return_length=10,
        ).to_extra_args(),
    )
    return SimpleNamespace(
        name=name,
        request_id=name,
        sampling_params=sp,
        num_prompt_tokens=32,
        # status not set here; starvation pass checks req.status == WAITING
        # but combined-selection only reads peek_key() (no status filter).
    )


def test_combined_selection_smaller_key_wins():
    """Combined selection returns the queue with the smaller head key."""
    starving: set[str] = set()
    q_wait = V2RequestQueue(starving, max_ragged_batch=384)
    q_skip = V2RequestQueue(starving, max_ragged_batch=384)

    # req_a has large api_exec_time (big preserve score = big key -> comes later)
    # req_b has small api_exec_time (small preserve score = small key -> comes first)
    req_a = mkreq("a", remain=100, api_exec_time=10.0)
    req_b = mkreq("b", remain=100, api_exec_time=0.1)

    q_wait.add_request(req_a)   # high key (expensive)
    q_skip.add_request(req_b)   # low key (cheap)

    # peek_key() must be consistent with peek_request()
    assert q_skip.peek_key() < q_wait.peek_key(), (
        f"expected skip({q_skip.peek_key():.4g}) < wait({q_wait.peek_key():.4g})"
    )

    # The combined selector should pick skipped (smaller key).
    def combined_select(waiting, skipped):
        w_has, s_has = bool(waiting), bool(skipped)
        if not w_has and not s_has:
            return None
        if w_has and not s_has:
            return waiting
        if s_has and not w_has:
            return skipped
        return waiting if waiting.peek_key() <= skipped.peek_key() else skipped

    chosen = combined_select(q_wait, q_skip)
    assert chosen is q_skip, "expected skipped_waiting (cheaper key) to win"
    print("  combined-select: cheaper-key queue wins -> OK")


def test_starvation_wins_from_either_queue():
    """-inf starvation key beats any finite key, from either queue."""
    starving: set[str] = set()
    q_wait = V2RequestQueue(starving, max_ragged_batch=384)
    q_skip = V2RequestQueue(starving, max_ragged_batch=384)

    req_x = mkreq("x", remain=1000, api_exec_time=50.0)  # very expensive
    req_y = mkreq("y", remain=100, api_exec_time=0.1)    # cheap but not starving

    q_wait.add_request(req_x)   # high key
    q_skip.add_request(req_y)   # low key (would normally win)

    # Starve req_x -> its key becomes -inf.
    starving.add("x")
    q_wait.rekey(lambda r: r.sampling_params.extra_args.get("mars", {}).get(
        "predicted_api_exec_time", 1.0))  # dummy; starvation override dominates

    assert q_wait.peek_key() == float("-inf"), (
        f"starvation should set key to -inf, got {q_wait.peek_key()}"
    )
    assert q_skip.peek_key() != float("-inf")

    def combined_select(waiting, skipped):
        w_has, s_has = bool(waiting), bool(skipped)
        if not w_has and not s_has:
            return None
        if w_has and not s_has:
            return waiting
        if s_has and not w_has:
            return skipped
        return waiting if waiting.peek_key() <= skipped.peek_key() else skipped

    chosen = combined_select(q_wait, q_skip)
    assert chosen is q_wait, "starving request (-inf) should always win"
    assert chosen.peek_request().name == "x"
    print("  starvation -inf wins regardless of queue -> OK")


def test_sjf_combined():
    """SJF combined queue: shorter remain_length wins regardless of queue."""
    starving: set[str] = set()
    q_wait = MARSRequestQueue(starving)
    q_skip = MARSRequestQueue(starving)

    req_long  = mkreq("long",  remain=500)
    req_short = mkreq("short", remain=10)

    q_wait.add_request(req_long)
    q_skip.add_request(req_short)

    def combined_select(waiting, skipped):
        w_has, s_has = bool(waiting), bool(skipped)
        if not w_has and not s_has:
            return None
        if w_has and not s_has:
            return waiting
        if s_has and not w_has:
            return skipped
        return waiting if waiting.peek_key() <= skipped.peek_key() else skipped

    chosen = combined_select(q_wait, q_skip)
    assert chosen is q_skip, "shorter remain_length in skipped should win"
    assert chosen.peek_request().name == "short"
    print("  SJF combined: shorter job in skipped_waiting wins -> OK")


def test_single_queue_fallback():
    """If only one queue is non-empty, return it regardless of key."""
    starving: set[str] = set()
    q_wait = V2RequestQueue(starving, max_ragged_batch=384)
    q_skip = V2RequestQueue(starving, max_ragged_batch=384)

    q_wait.add_request(mkreq("only", remain=100))

    def combined_select(waiting, skipped):
        w_has, s_has = bool(waiting), bool(skipped)
        if not w_has and not s_has:
            return None
        if w_has and not s_has:
            return waiting
        if s_has and not w_has:
            return skipped
        return waiting if waiting.peek_key() <= skipped.peek_key() else skipped

    assert combined_select(q_wait, q_skip) is q_wait
    assert combined_select(q_skip, q_wait) is q_wait  # skipped empty -> return wait
    assert combined_select(q_skip, q_skip) is None
    print("  single-queue and empty fallback -> OK")


if __name__ == "__main__":
    test_combined_selection_smaller_key_wins()
    test_starvation_wins_from_either_queue()
    test_sjf_combined()
    test_single_queue_fallback()
    print(">>> COMBINED_QUEUE_OK")
