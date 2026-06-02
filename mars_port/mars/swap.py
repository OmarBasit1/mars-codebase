"""Enable vLLM's native CPU KV-offload connector as the MARS swap data-path.

The original MARS swapped a paused request's KV to host memory. On modern vLLM
(v1, no in-core swap) we get the same effect from the supported
``SimpleCPUOffloadConnector``: when a paused request's GPU blocks are freed, the
connector keeps the KV in the CPU/host tier (backed by the prefix cache), and on
resume the re-prefill reloads it instead of recomputing.

Pass the returned config to ``(Async)EngineArgs(kv_transfer_config=...)`` together
with ``enable_prefix_caching=True`` (the connector requires prefix caching).
``MARSScheduler`` then detects the connector and enables the SWAP policy.
"""

from __future__ import annotations

from vllm.config import KVTransferConfig

DEFAULT_CPU_GB = 4.0


def cpu_offload_kv_transfer_config(cpu_gb: float = DEFAULT_CPU_GB) -> KVTransferConfig:
    """Return a KVTransferConfig wiring ``SimpleCPUOffloadConnector``.

    Args:
        cpu_gb: Host memory budget (GiB) for offloaded KV.

    Returns:
        A ``KVTransferConfig`` for ``(Async)EngineArgs(kv_transfer_config=...)``.
    """
    return KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": int(cpu_gb * 1024**3)},
    )
