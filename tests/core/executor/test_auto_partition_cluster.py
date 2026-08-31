"""Tests for cluster-aware auto partitioning.

Covers:
- Shared cluster topology detection (real node count, fallback behavior)
- PartitionSizeOptimizer cluster fixes (node count, cluster-wide capacity)
- Executor sentinel parsing (num_of_partitions: auto / int / invalid)
- Cluster partition bounds formula (floor, target, data-driven ceiling)
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from data_juicer.core.executor.ray_executor_partitioned import (
    PartitionedRayExecutor,
)
from data_juicer.utils.ray_cluster_utils import (
    ClusterTopology,
    detect_cluster_topology,
)

MULTINODE_NODES = [
    {"Alive": True, "NodeID": "n1"},
    {"Alive": True, "NodeID": "n2"},
    {"Alive": True, "NodeID": "n3"},
    {"Alive": True, "NodeID": "n4"},
    {"Alive": False, "NodeID": "dead"},
]
MULTINODE_CLUSTER = {"CPU": 256.0, "GPU": 32.0, "memory": 1024.0 * 1024**3}
MULTINODE_AVAILABLE = {"CPU": 256.0, "GPU": 32.0, "memory": 1024.0 * 1024**3}


def _patch_ray_multinode():
    """Patch ray as a 4-node / 32-GPU cluster."""
    return [
        patch("ray.is_initialized", return_value=True),
        patch("ray.nodes", return_value=MULTINODE_NODES),
        patch("ray.cluster_resources", return_value=MULTINODE_CLUSTER),
        patch("ray.available_resources", return_value=MULTINODE_AVAILABLE),
    ]


class ClusterTopologyTest(unittest.TestCase):
    def test_multinode_topology_counts_alive_nodes_only(self):
        patches = _patch_ray_multinode()
        for p in patches:
            p.start()
        try:
            topology = detect_cluster_topology()
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(topology.num_nodes, 4)
        self.assertEqual(topology.total_cpus, 256.0)
        self.assertEqual(topology.total_gpus, 32.0)
        self.assertAlmostEqual(topology.gpus_per_node, 8.0)

    def test_fallback_when_ray_not_initialized(self):
        with patch("ray.is_initialized", return_value=False):
            topology = detect_cluster_topology()
        self.assertEqual(topology.num_nodes, 1)
        self.assertEqual(topology.total_cpus, 0.0)
        self.assertEqual(topology.total_gpus, 0.0)


class OptimizerClusterFixTest(unittest.TestCase):
    def test_detect_ray_cluster_uses_real_node_count(self):
        from data_juicer.core.executor.partition_size_optimizer import (
            ResourceDetector,
        )

        patches = _patch_ray_multinode()
        for p in patches:
            p.start()
        try:
            cluster = ResourceDetector.detect_ray_cluster()
        finally:
            for p in patches:
                p.stop()
        self.assertIsNotNone(cluster)
        # Old heuristic would have guessed 256 / 8 = 32 nodes.
        self.assertEqual(cluster.num_nodes, 4)
        self.assertEqual(cluster.total_cpu_cores, 256)

    def test_worker_count_uses_cluster_capacity_not_driver_local(self):
        from data_juicer.core.executor.partition_size_optimizer import (
            ClusterResources,
            LocalResources,
            ResourceDetector,
        )

        local = LocalResources(
            cpu_cores=8,
            available_memory_gb=32.0,
            total_memory_gb=64.0,
            gpu_count=0,
        )
        cluster = ClusterResources(
            num_nodes=4,
            total_cpu_cores=256,
            total_memory_gb=2048.0,
            available_cpu_cores=192,
            available_memory_gb=2048.0,
            gpu_resources={},
        )
        workers = ResourceDetector.calculate_optimal_worker_count(local, cluster)
        # Cluster-wide capacity is authoritative: 75% of 192, not of 8.
        self.assertEqual(workers, int(192 * 0.75))


def _fake_executor(**attrs):
    """Build a minimal stand-in carrying only the state the methods need."""
    fake = SimpleNamespace()
    fake.cfg = SimpleNamespace()
    fake.max_concurrent_partitions = "auto"
    fake.num_partitions = 4
    fake.partition_mode = "auto"
    fake._partitions_per_node_cfg = "auto"
    for key, value in attrs.items():
        setattr(fake, key, value)
    return fake


def _gpu_op(num_gpus):
    return SimpleNamespace(num_gpus=num_gpus, _name="gpu_op")


class SentinelParsingTest(unittest.TestCase):
    def _configure(self, partition_cfg):
        fake = _fake_executor()
        fake.cfg = SimpleNamespace(partition=partition_cfg)
        PartitionedRayExecutor._configure_partitioning(fake)
        return fake

    def test_auto_sentinel_keeps_auto_mode(self):
        fake = self._configure({"mode": "manual", "num_of_partitions": "auto"})
        self.assertEqual(fake.partition_mode, "auto")

    def test_numeric_string_normalized(self):
        fake = self._configure({"mode": "manual", "num_of_partitions": "8"})
        self.assertEqual(fake.partition_mode, "manual")
        self.assertEqual(fake.num_partitions, 8)

    def test_invalid_value_falls_back_to_auto(self):
        fake = self._configure({"mode": "manual", "num_of_partitions": "bogus"})
        self.assertEqual(fake.partition_mode, "auto")

    def test_explicit_int_manual_mode_wins(self):
        fake = self._configure({"mode": "manual", "num_of_partitions": 16})
        self.assertEqual(fake.partition_mode, "manual")
        self.assertEqual(fake.num_partitions, 16)


class ClusterPartitionBoundsTest(unittest.TestCase):
    MULTINODE = ClusterTopology(
        num_nodes=2,
        total_cpus=128.0,
        total_gpus=16.0,
        available_cpus=128.0,
        available_gpus=16.0,
    )

    def _apply(self, fake, ops):
        fake._resolve_partitions_per_node = lambda op_list, topology: (
            PartitionedRayExecutor._resolve_partitions_per_node(fake, op_list, topology)
        )
        with patch(
            "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
            return_value=self.MULTINODE,
        ):
            PartitionedRayExecutor._apply_cluster_partition_bounds(fake, ops)
        return fake

    def test_target_reaches_twice_concurrency(self):
        # 2 nodes x 8 GPUs, ops at 0.5 GPU each -> per_node=16, floor=32.
        fake = _fake_executor(num_partitions=200, max_concurrent_partitions=32)
        self._apply(fake, [_gpu_op(0.5)])
        self.assertEqual(fake.num_partitions, 64)

    def test_data_driven_ceiling_is_respected(self):
        fake = _fake_executor(num_partitions=40, max_concurrent_partitions=32)
        self._apply(fake, [_gpu_op(0.5)])
        self.assertEqual(fake.num_partitions, 40)

    def test_count_below_floor_is_raised(self):
        fake = _fake_executor(num_partitions=10, max_concurrent_partitions=32)
        self._apply(fake, [_gpu_op(0.5)])
        self.assertEqual(fake.num_partitions, 32)

    def test_unresolved_concurrency_uses_node_floor(self):
        fake = _fake_executor(num_partitions=4, max_concurrent_partitions="auto")
        self._apply(fake, [_gpu_op(0.5)])
        self.assertEqual(fake.num_partitions, 32)  # 2 nodes x 16 per node

    def test_cpu_only_pipeline_uses_overlap_factor(self):
        cpu_topology = ClusterTopology(
            num_nodes=4,
            total_cpus=256.0,
            total_gpus=0.0,
            available_cpus=256.0,
            available_gpus=0.0,
        )
        fake = _fake_executor(num_partitions=1000, max_concurrent_partitions=8)
        fake._resolve_partitions_per_node = lambda op_list, topology: (
            PartitionedRayExecutor._resolve_partitions_per_node(fake, op_list, topology)
        )
        with patch(
            "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
            return_value=cpu_topology,
        ):
            PartitionedRayExecutor._apply_cluster_partition_bounds(fake, [])
        # per_node=2 -> floor=max(8,4,8)=8, target=16, ceiling=1000.
        self.assertEqual(fake.num_partitions, 16)

    def test_resolved_plan_published_on_cfg(self):
        fake = _fake_executor(num_partitions=200, max_concurrent_partitions=32)
        self._apply(fake, [_gpu_op(0.5)])
        plan = fake.cfg._resolved_partition_plan
        self.assertEqual(plan["num_partitions"], 64)
        self.assertEqual(plan["num_nodes"], 2)
        self.assertEqual(plan["partitions_per_node"], 16)


class PartitionsPerNodeTest(unittest.TestCase):
    TOPOLOGY = ClusterTopology(
        num_nodes=2,
        total_cpus=128.0,
        total_gpus=16.0,
        available_cpus=128.0,
        available_gpus=16.0,
    )

    def test_explicit_multiplier_wins(self):
        fake = _fake_executor(_partitions_per_node_cfg=3)
        value = PartitionedRayExecutor._resolve_partitions_per_node(
            fake, [_gpu_op(0.5)], self.TOPOLOGY
        )
        self.assertEqual(value, 3)

    def test_gpu_slot_derivation(self):
        fake = _fake_executor()
        value = PartitionedRayExecutor._resolve_partitions_per_node(
            fake, [_gpu_op(0.5)], self.TOPOLOGY
        )
        self.assertEqual(value, 16)  # 8 GPUs/node / 0.5 GPU per worker

    def test_tightest_stage_dominates(self):
        fake = _fake_executor()
        value = PartitionedRayExecutor._resolve_partitions_per_node(
            fake, [_gpu_op(0.5), _gpu_op(1.0)], self.TOPOLOGY
        )
        self.assertEqual(value, 8)  # 8 GPUs/node / 1.0 GPU

    def test_cpu_only_fallback(self):
        cpu_topology = ClusterTopology(
            num_nodes=2,
            total_cpus=128.0,
            total_gpus=0.0,
            available_cpus=128.0,
            available_gpus=0.0,
        )
        fake = _fake_executor()
        value = PartitionedRayExecutor._resolve_partitions_per_node(fake, [], cpu_topology)
        self.assertEqual(value, 2)

    def test_cpu_only_pipeline_on_gpu_cluster_uses_overlap_factor(self):
        # GPU cluster must not size CPU-only pipelines by GPU count.
        fake = _fake_executor()
        value = PartitionedRayExecutor._resolve_partitions_per_node(fake, [], self.TOPOLOGY)
        self.assertEqual(value, 2)

    def test_cuda_flag_marks_gpu_pipeline_without_num_gpus(self):
        cuda_op = SimpleNamespace(num_gpus=None, use_cuda=lambda: True, _name="cuda_op")
        fake = _fake_executor()
        value = PartitionedRayExecutor._resolve_partitions_per_node(
            fake, [cuda_op], self.TOPOLOGY
        )
        self.assertEqual(value, 8)  # 8 GPUs/node / 1.0 GPU implicit

    def test_invalid_multiplier_falls_back_to_auto(self):
        fake = _fake_executor(_partitions_per_node_cfg="bogus")
        value = PartitionedRayExecutor._resolve_partitions_per_node(
            fake, [_gpu_op(0.5)], self.TOPOLOGY
        )
        self.assertEqual(value, 16)


class OptimizerFallbackTest(unittest.TestCase):
    """Cluster bounds must still apply when the optimizer fails."""

    def _bound_fake(self, **attrs):
        fake = _fake_executor(**attrs)
        fake._resolve_partitions_per_node = lambda op_list, topology: (
            PartitionedRayExecutor._resolve_partitions_per_node(fake, op_list, topology)
        )
        fake._apply_cluster_partition_bounds = lambda op_list: (
            PartitionedRayExecutor._apply_cluster_partition_bounds(fake, op_list)
        )
        return fake

    def test_cluster_bounds_applied_when_optimizer_raises(self):
        fake = self._bound_fake(num_partitions=4, max_concurrent_partitions=8)
        with patch(
            "data_juicer.core.executor.partition_size_optimizer.auto_configure_resources",
            side_effect=RuntimeError("optimizer unavailable"),
        ), patch(
            "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
            return_value=ClusterPartitionBoundsTest.MULTINODE,
        ):
            PartitionedRayExecutor._configure_auto_partitioning(fake, None, [_gpu_op(0.5)])
        # floor = 2 nodes x 16 per node = 32; placeholder 4 is raised.
        self.assertEqual(fake.num_partitions, 32)

    def test_cluster_bounds_applied_when_optimizer_import_fails(self):
        fake = self._bound_fake(num_partitions=4, max_concurrent_partitions=8)
        with patch(
            "data_juicer.core.executor.partition_size_optimizer.auto_configure_resources",
            side_effect=ImportError("missing dependency"),
        ), patch(
            "data_juicer.utils.ray_cluster_utils.detect_cluster_topology",
            return_value=ClusterPartitionBoundsTest.MULTINODE,
        ):
            PartitionedRayExecutor._configure_auto_partitioning(fake, None, [_gpu_op(0.5)])
        self.assertEqual(fake.num_partitions, 32)


if __name__ == "__main__":
    unittest.main()
