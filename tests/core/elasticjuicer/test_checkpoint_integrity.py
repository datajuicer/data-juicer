"""Regression tests for checkpoint integrity validation (0803 row-loss defect).

Defect: Ray ``write_parquet`` writes one file per block with no atomic
commit. A driver crash mid-checkpoint leaves a readable but row-incomplete
checkpoint directory; the old resume path trusted file existence and
silently dropped the missing rows (formal matrix kill:p8:ej_on lost
sample_id 784-799 of 1600).

Fix under test: ``_validate_checkpoint_integrity`` compares the loaded
checkpoint row count against the split-time partition membership
(``_partition_expected_rows``) and discards mismatched checkpoints, forcing
a full partition recompute instead of silent loss.

The corruption here is deterministic: we remove one parquet block file from
a completed checkpoint directory, reproducing the exact on-disk state the
SIGKILL race produced.
"""

import os
import shutil
import tempfile
import unittest

from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase

from .test_ray_partitioned_adaptive_e2e import PartitionedThresholdMapper


class CheckpointIntegrityTest(DataJuicerTestCaseBase):

    ROWS = 64

    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="ej_ckpt_integrity_")
        self.ckpt_dir = os.path.join(self.tmp_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def _elastic_cfg(self, name: str):
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            project_name=name,
            op_fusion=False,
            auto_op_parallelism=False,
            elastic_juicer_adaptive_batching=False,
        )
        cfg.get = lambda key, default=None: getattr(cfg, key, default)
        cfg.__setitem__ = lambda key, value: setattr(cfg, key, value)
        cfg.__getitem__ = lambda key: getattr(cfg, key)
        return cfg

    def _build_executor(self, cfg, num_partitions=1):
        from data_juicer.core.executor.ray_executor_partitioned import (
            PartitionedRayExecutor,
        )

        executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
        executor.cfg = cfg
        executor.executor_type = "ray_partitioned"
        executor.num_partitions = num_partitions
        executor.max_concurrent_partitions = num_partitions
        executor.pipeline_dag = None
        executor.event_logger = None
        executor.work_dir = os.path.join(self.tmp_dir, "work")
        os.makedirs(executor.work_dir, exist_ok=True)
        executor.ckpt_manager = RayCheckpointManager(
            ckpt_dir=self.ckpt_dir,
            checkpoint_enabled=True,
            checkpoint_strategy=CheckpointStrategy.EVERY_OP,
        )
        return executor

    def _operators(self):
        ops = []
        for index, increment in enumerate((1, 2)):
            op = PartitionedThresholdMapper(
                oom_above=64,
                increment=increment,
                batch_size=4,
                num_proc=1,
                auto_op_parallelism=False,
                ray_execution_mode="actor",
                skip_op_error=False,
            )
            op._name = f"ckpt_integrity_inc_{index}"
            ops.append(op)
        return ops

    def _source_dataset(self, cfg):
        import ray

        from data_juicer.core.data.ray_dataset import RayDataset

        source = ray.data.from_items(
            [{"value": index % 7} for index in range(self.ROWS)], override_num_blocks=4
        )
        return RayDataset(source, cfg=cfg, auto_op_parallelism=False)

    def _corrupt_latest_checkpoint(self) -> int:
        """Delete one block file from the latest checkpoint; return removed count."""
        latest_dir = None
        highest = -1
        for name in os.listdir(self.ckpt_dir):
            if name.startswith("checkpoint_op_") and name.endswith("_partition_0000.parquet"):
                op_idx = int(name.split("_")[2])
                if op_idx > highest:
                    highest = op_idx
                    latest_dir = os.path.join(self.ckpt_dir, name)
        assert latest_dir is not None and os.path.isdir(latest_dir), "no checkpoint dir found"
        block_files = sorted(f for f in os.listdir(latest_dir) if f.endswith(".parquet"))
        assert len(block_files) >= 2, (
            f"need >=2 parquet block files to simulate a partial write, got {block_files}"
        )
        victim = block_files[len(block_files) // 2]
        os.remove(os.path.join(latest_dir, victim))
        return 1

    @TEST_TAG("ray")
    def test_truncated_checkpoint_is_rejected_and_recomputed(self):
        cfg = self._elastic_cfg("ej-ckpt-integrity")
        ops = self._operators()

        # Run A: full job, writes checkpoints for both ops.
        executor_a = self._build_executor(cfg)
        result_a = executor_a._process_with_simple_partitioning(self._source_dataset(cfg), ops)
        rows_a = sorted(row["value"] for row in result_a.data.take_all())
        self.assertEqual(rows_a, sorted((index % 7) + 3 for index in range(self.ROWS)))

        # Deterministic corruption: drop one parquet block file from the
        # latest checkpoint (reproduces the SIGKILL-mid-write on-disk state).
        self._corrupt_latest_checkpoint()

        # Direct check: validation must reject the truncated checkpoint.
        executor_b = self._build_executor(cfg)
        executor_b._partition_expected_rows = {0: self.ROWS}
        latest = executor_b.ckpt_manager.find_latest_checkpoint(0)
        self.assertIsNotNone(latest)
        self.assertIsNone(executor_b._validate_checkpoint_integrity(latest, 0))

        # End-to-end: resume must recompute the partition and stay lossless.
        result_b = executor_b._process_with_simple_partitioning(self._source_dataset(cfg), self._operators())
        rows_b = sorted(row["value"] for row in result_b.data.take_all())
        self.assertEqual(len(rows_b), self.ROWS, "resumed output must not drop rows")
        self.assertEqual(rows_b, sorted((index % 7) + 3 for index in range(self.ROWS)))

    @TEST_TAG("ray")
    def test_complete_checkpoint_still_resumes(self):
        cfg = self._elastic_cfg("ej-ckpt-integrity-ok")
        ops = self._operators()

        executor_a = self._build_executor(cfg)
        executor_a._process_with_simple_partitioning(self._source_dataset(cfg), ops)

        # Untouched checkpoints must pass validation and be reused.
        executor_b = self._build_executor(cfg)
        source = self._source_dataset(cfg)
        executor_b._process_with_simple_partitioning(source, self._operators())
        latest = executor_b.ckpt_manager.find_latest_checkpoint(0)
        self.assertIsNotNone(latest)
        self.assertIsNotNone(executor_b._validate_checkpoint_integrity(latest, 0))


if __name__ == "__main__":
    unittest.main()
