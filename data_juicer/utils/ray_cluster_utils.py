"""Shared Ray cluster topology detection.

Partition sizing, driver concurrency, and any other cluster-aware decision
must resolve the cluster topology through this single helper so that every
consumer acts on the same view of the cluster.

Key points:
- The node count is derived from ``ray.nodes()`` (alive nodes only) instead
  of guessing it from total CPU resources.
- Resource totals come from ``ray.cluster_resources()`` and therefore
  describe the whole cluster, not the driver machine.
- Any detection failure degrades to a single-node fallback instead of
  raising, so callers never need special error handling.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from loguru import logger


@dataclass(frozen=True)
class ClusterTopology:
    """Cluster-wide resource view used for partitioning decisions."""

    num_nodes: int
    total_cpus: float
    total_gpus: float
    available_cpus: float
    available_gpus: float
    # Exact node capacities are needed for a multi-GPU actor: Ray cannot
    # combine devices from different nodes for one actor's reservation.
    gpu_devices_per_node: tuple = ()

    @property
    def gpus_per_node(self) -> float:
        """Average GPUs per node (0.0 for CPU-only clusters)."""
        if self.num_nodes <= 0:
            return 0.0
        return self.total_gpus / self.num_nodes


def detect_cluster_topology() -> ClusterTopology:
    """Detect the Ray cluster topology.

    Falls back to a conservative single-node view when Ray is unavailable,
    not initialized, or inspection fails, so callers can always rely on a
    usable result.
    """
    fallback = ClusterTopology(
        num_nodes=1,
        total_cpus=0.0,
        total_gpus=0.0,
        available_cpus=0.0,
        available_gpus=0.0,
    )
    try:
        import ray

        if not ray.is_initialized():
            return fallback

        nodes = ray.nodes()
        num_nodes = max(1, sum(1 for node in nodes if node.get("Alive")))

        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        return ClusterTopology(
            num_nodes=num_nodes,
            total_cpus=float(cluster_resources.get("CPU", 0)),
            total_gpus=float(cluster_resources.get("GPU", 0)),
            available_cpus=float(available_resources.get("CPU", 0)),
            available_gpus=float(available_resources.get("GPU", 0)),
            gpu_devices_per_node=tuple(
                int(float(node.get("Resources", {}).get("GPU", 0))) for node in nodes if node.get("Alive")
            ),
        )
    except Exception as e:
        logger.warning(f"Could not detect Ray cluster topology, using single-node fallback: {e}")
        return fallback


def gpu_actor_requests_fit(
    stage_specs: List[Dict[str, Any]], counts: List[int], total_gpus: float, gpu_devices_per_node=None
) -> bool:
    """Check scheduling and measured-memory fractions on individual GPUs.

    Requests are processed largest first and placed with a best-fit policy,
    preserving the largest remaining contiguous shares for later requests.
    """
    device_count = int(math.floor(total_gpus + 1e-9))
    if device_count < 1:
        return False
    requests = []
    for spec, count in zip(stage_specs, counts):
        requests.extend([(spec["num_gpus"], spec["memory_fraction"], spec["max_per_device"])] * count)
    requests.sort(key=lambda item: (max(item[0], item[1]), item[1]), reverse=True)
    node_sizes = gpu_devices_per_node if gpu_devices_per_node is not None else (device_count,)
    devices = [
        {"gpu": 0.0, "memory": 0.0, "workers": 0, "node": node}
        for node, size in enumerate(node_sizes)
        for _ in range(size)
    ]
    for gpu_fraction, memory_fraction, max_per_device in requests:
        if gpu_fraction > 1:
            if not float(gpu_fraction).is_integer():
                return False
            # Ray reserves whole, same-node devices for integer requests.
            for node in range(len(node_sizes)):
                free = [device for device in devices if device["node"] == node and device["workers"] == 0]
                if len(free) >= int(gpu_fraction):
                    for device in free[: int(gpu_fraction)]:
                        device.update(gpu=1.0, memory=1.0, workers=1)
                    break
            else:
                return False
            continue
        eligible = [
            device
            for device in devices
            if (
                device["gpu"] + gpu_fraction <= 1.0 + 1e-9
                and device["memory"] + memory_fraction <= 1.0 + 1e-9
                and device["workers"] < max_per_device
            )
        ]
        if not eligible:
            return False

        # Best fit avoids fragmenting devices by putting the request in the
        # eligible device with the least remaining dominant resource share.
        device = min(
            eligible,
            key=lambda item: (
                max(1.0 - item["gpu"] - gpu_fraction, 1.0 - item["memory"] - memory_fraction),
                2.0 - item["gpu"] - gpu_fraction - item["memory"] - memory_fraction,
                max_per_device - item["workers"] - 1,
            ),
        )
        device["gpu"] += gpu_fraction
        device["memory"] += memory_fraction
        device["workers"] += 1
    return True
