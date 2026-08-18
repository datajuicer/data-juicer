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

from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class ClusterTopology:
    """Cluster-wide resource view used for partitioning decisions."""

    num_nodes: int
    total_cpus: float
    total_gpus: float
    available_cpus: float
    available_gpus: float

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
        )
    except Exception as e:
        logger.warning(f"Could not detect Ray cluster topology, using single-node fallback: {e}")
        return fallback
