"""Unit test: V2 cost-based ordering + starvation front-boost (no GPU)."""

from types import SimpleNamespace

from vllm import SamplingParams

from mars.params import MarsApiParams
from mars.v1.queue import MARSRequestQueue, V2RequestQueue, make_mars_queue


def mkreq(name, *, remain=0, invoke=128, ret=32, exec_t=1.0, prompt=64):
    sp = SamplingParams(
        max_tokens=8,
        extra_args=MarsApiParams(
            remain_length=remain,
            predicted_api_invoke_interval=invoke,
            api_return_length=ret,
            predicted_api_exec_time=exec_t,
        ).to_extra_args(),
    )
    return SimpleNamespace(request_id=name, sampling_params=sp, num_prompt_tokens=prompt)


def main() -> None:
    assert isinstance(make_mars_queue("sjf"), MARSRequestQueue)
    assert isinstance(make_mars_queue("V2"), V2RequestQueue)
    assert make_mars_queue("fcfs") is None

    # V2: a small / short-API request is cheaper -> scheduled before a large one.
    starving: set[str] = set()
    q = V2RequestQueue(starving)
    small = mkreq("small", invoke=16, ret=8, exec_t=0.1, prompt=16)
    big = mkreq("big", invoke=512, ret=128, exec_t=10.0, prompt=512)
    q.add_request(big)
    q.add_request(small)
    assert q.peek_request().request_id == "small", "cheaper V2 cost should be first"

    # Starvation: mark 'big' starving -> re-add jumps it to the front (key -inf).
    starving.add("big")
    q.remove_request(big)
    q.add_request(big)
    assert q.peek_request().request_id == "big", "starving request should be front"
    print(">>> QUEUE_STARVATION_OK")


if __name__ == "__main__":
    main()
