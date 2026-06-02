"""MARS scheduler for vLLM v1.

Phase 1 establishes the injection point only: ``MARSScheduler`` is a *pure
pass-through* subclass of vLLM's v1 ``Scheduler``, loaded via
``SchedulerConfig.scheduler_cls`` (e.g. ``--scheduler-cls
mars.v1.scheduler.MARSScheduler``). With no overrides it must behave identically
to the stock scheduler — this is verified by a golden-equivalence test.

Later phases override ``add_request``/``schedule``/``update_from_output`` to add:
  * API-pause detection (stop-token / token-interval) and the pause/resume
    lifecycle on top of native resumable streaming;
  * per-pause KV-cache policies (Preserve / Recompute, then LMCache swap);
  * chunk-fill admission expressed as ``token_budget`` shaping;
  * the Vulcan cost model + solver.

Every override is designed to *wrap* (call ``super()``) rather than replace the
base logic, so the scheduler-side KV-connector hooks keep firing — see plan
risk "MARSScheduler <-> connector cooperation".
"""

from vllm.v1.core.sched.scheduler import Scheduler


class MARSScheduler(Scheduler):
    """MARS scheduler.

    Phase 1: no behavior change. Subclassing the public ``SchedulerInterface``
    implementation is the supported way to inject custom scheduling without
    editing vLLM core.
    """

    # No overrides yet — intentional. Behavior is validated to be byte-identical
    # to ``vllm.v1.core.sched.scheduler.Scheduler`` on a no-API workload.
    pass
