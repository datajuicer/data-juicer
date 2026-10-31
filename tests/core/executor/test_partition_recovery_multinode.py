"""Checkpointing on a simulated two-node Ray cluster: placement and worker-node loss."""

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import ray
from jsonargparse import Namespace
from ray.cluster_utils import Cluster

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor.partition_protocol import COMMIT_DIR, LocalStore
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops.base_op import Mapper
from data_juicer.utils.ckpt_utils import RayCheckpointManager
from data_juicer.utils.unittest_utils import TEST_TAG

ROWS = 600


class NodeTagMapper(Mapper):
    _batched_op = True

    def __init__(self, tag, delay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.tag = tag
        self.delay = delay

    def process_batched(self, samples):
        time.sleep(self.delay)
        samples[self.tag] = [ray.get_runtime_context().get_node_id()] * len(samples["id"])
        return samples


class DropMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        keep = [i for i, value in enumerate(samples["id"]) if value % 3]
        return {key: [values[i] for i in keep] for key, values in samples.items()}


class DupMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        index = [i for i, value in enumerate(samples["id"]) for _ in range(2 if value % 5 == 0 else 1)]
        return {key: [values[i] for i in index] for key, values in samples.items()}


EXPECTED = sorted(i for i in range(ROWS) if i % 3 for _ in range(2 if i % 5 == 0 else 1))


def _run(checkpoint_dir, delay=0.0):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = Namespace(auto_op_parallelism=False)
    executor.ckpt_manager = RayCheckpointManager(str(checkpoint_dir), checkpoint_enabled=True)
    executor.job_id = "multinode"
    executor.partition_size_cfg = 20
    executor._is_resuming = False
    ops = [
        NodeTagMapper("n0", delay, num_proc=2, num_cpus=1, ray_execution_mode="actor", batch_size=5),
        DropMapper(num_proc=2, num_cpus=1, ray_execution_mode="task"),
        DupMapper(num_proc=2, num_cpus=1, ray_execution_mode="actor"),
        NodeTagMapper("n3", delay, num_proc=2, num_cpus=1, ray_execution_mode="task"),
    ]
    rows = [{"id": i, "text": str(i)} for i in range(ROWS)]
    source = RayDataset(ray.data.from_items(rows, override_num_blocks=8), cfg=executor.cfg)
    dataset, recovery = executor._process_with_partition_recovery(source, ops)
    try:
        coordinator_node = ray.get(
            recovery.coordinator.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
        )
        output = dataset.data.take_all()
        recovery.finish()
    finally:
        recovery.close()
    assert sorted(row["id"] for row in output) == EXPECTED
    assert all("__dj_ckpt_seq__" not in row and "__dj_ckpt_idx__" not in row for row in output)
    return output, coordinator_node


class PartitionRecoveryMultinodeTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="dj-recovery-test-")
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)
        # Own this local two-node cluster; never remove a node from the CI cluster.
        ray.shutdown()
        self.cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 0, "include_dashboard": False})
        self.addCleanup(self.cluster.shutdown)
        self.addCleanup(ray.shutdown)
        self.cluster.add_node(num_cpus=8)
        self.cluster.wait_for_nodes()
        ray.init(address=self.cluster.address, log_to_driver=False)

    @TEST_TAG("ray")
    def test_coordinator_stays_on_driver_node_while_workers_commit(self):
        output, coordinator_node = _run(self.tmp_path / "checkpoints")
        head = self.cluster.head_node.node_id
        self.assertTrue(coordinator_node == ray.get_runtime_context().get_node_id() == head)
        self.assertEqual(
            {row["n0"] for row in output} | {row["n3"] for row in output}, {self.cluster.list_all_nodes()[1].node_id}
        )

    @TEST_TAG("ray")
    def test_worker_node_loss_neither_loses_nor_duplicates_rows(self):
        checkpoint_dir = self.tmp_path / "checkpoints"
        store = LocalStore(str(checkpoint_dir / "partition_recovery"))
        result = {}

        def run():
            try:
                result["output"] = _run(checkpoint_dir, delay=0.3)[0]
            except BaseException as error:  # noqa: B902
                result["error"] = error

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.time() + 180
        while len(store.list(COMMIT_DIR)) < 8:
            self.assertTrue(time.time() < deadline and thread.is_alive(), msg=result)
            time.sleep(0.2)
        worker = self.cluster.list_all_nodes()[1]
        self.cluster.remove_node(worker, allow_graceful=False)
        time.sleep(3)
        replacement = self.cluster.add_node(num_cpus=8)
        thread.join(timeout=600)
        self.assertTrue(not thread.is_alive() and "error" not in result, msg=result)
        # Rows after the loss were produced on the replacement node; finish() verified exactly-once delivery.
        self.assertIn(replacement.node_id, {row["n3"] for row in result["output"]})


if __name__ == "__main__":
    unittest.main()
