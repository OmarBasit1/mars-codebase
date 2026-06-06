"""Unit test: arrival-time classify() and live-running_batch V2 ordering (no GPU).

Binds the MARS scheduler-mixin methods to a tiny fake scheduler (no engine) to
check, without a GPU:
  * classify() picks the argmin strategy at arrival from the *predicted* length
    (a zero-API-wait request -> preserve; a huge-API-wait request -> not
    preserve);
  * the V2 score re-keyed with a live ``running_batch`` reorders the queue
    (compute-heavy vs API-heavy requests swap rank as load rises) — the
    faithful per-step re-rank the static ``running_batch=0`` key could not do.
"""

from types import SimpleNamespace

from vllm import SamplingParams

from mars.config import MarsConfig
from mars.cost_model import CostModel, CostModelCoeffs
from mars.params import MarsApiParams
from mars.v1.queue import V2RequestQueue
from mars.v1.scheduler import _MARSSchedulerMixin, _MarsReqState

_BOUND = (
    "_api_policy_for", "_running_contention", "_mars_classify",
    "_classify_strategy", "_v2_score", "_mem_time",
)


def make_fake_sched(swap_available: bool = True) -> SimpleNamespace:
    fs = SimpleNamespace(
        _block_size=16,
        cost_model=CostModel(CostModelCoeffs()),
        _swap_available=swap_available,
        solver=None,
        mars_state={},
        running=[],
        mars_config=MarsConfig(api_policy="V"),
    )
    for name in _BOUND:  # bind unbound mixin methods to the fake instance
        setattr(fs, name, getattr(_MARSSchedulerMixin, name).__get__(fs))
    return fs


def mkreq(rid, prompt, invoke, ret, api_t, max_calls=1):
    sp = SamplingParams(
        max_tokens=8,
        extra_args=MarsApiParams(
            predicted_api_invoke_interval=invoke, api_invoke_interval=invoke,
            api_return_length=ret, predicted_api_exec_time=api_t, api_exec_time=api_t,
            api_max_calls=max_calls, api_policy="V",
        ).to_extra_args(),
    )
    return SimpleNamespace(
        request_id=rid, num_prompt_tokens=prompt, num_tokens=prompt,
        arrival_time=0.0, sampling_params=sp,
    )


def test_classify() -> None:
    fs = make_fake_sched()
    r_short = mkreq("s", prompt=32, invoke=16, ret=8, api_t=0.0)    # no API hold
    r_long = mkreq("l", prompt=512, invoke=256, ret=8, api_t=100.0)  # huge hold
    fs._mars_classify(r_short)
    fs._mars_classify(r_long)
    s, ll = fs.mars_state["s"], fs.mars_state["l"]
    print(f"classify short -> {s.arrival_strategy} (waste={s.arrival_waste:.4g})")
    print(f"classify long  -> {ll.arrival_strategy} (waste={ll.arrival_waste:.4g})")
    assert s.arrival_strategy == "preserve", s.arrival_strategy  # w_p=0 wins
    assert ll.arrival_strategy != "preserve", ll.arrival_strategy  # huge hold loses
    print(">>> CLASSIFY_OK")


def test_v2_rekey() -> None:
    fs = make_fake_sched()
    q = V2RequestQueue()
    rA = mkreq("A", prompt=2048, invoke=64, ret=8, api_t=0.1)   # compute-heavy, short API
    rB = mkreq("B", prompt=64, invoke=64, ret=8, api_t=50.0)    # compute-light, long API
    for r in (rA, rB):
        fs.mars_state[r.request_id] = _MarsReqState(
            policy_letter="V", arrival_strategy="preserve")
        q.add_request(r)

    def order(running_batch: int):
        q.rekey(lambda r: fs._v2_score(r, running_batch, 0))
        return [r.request_id for r in q]  # __iter__ = ascending key = admit order

    o_idle = order(0)     # idle: API-hold dominates -> A (short API) first
    o_busy = order(380)   # busy: compute dominates -> B (light) first
    print(f"order running_batch=0   -> {o_idle}")
    print(f"order running_batch=380 -> {o_busy}")
    assert o_idle == ["A", "B"], o_idle
    assert o_busy == ["B", "A"], o_busy
    assert o_idle != o_busy, "live running_batch did not change V2 order"
    print(">>> V2_REKEY_OK")


if __name__ == "__main__":
    test_classify()
    test_v2_rekey()
    print(">>> CLASSIFY_V2_OK")
