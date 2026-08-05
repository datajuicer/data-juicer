"""Joint E2E for ``ray_partitioned`` + actor-local ElasticJuicer adaptive batching.

PR-RP-EJ-1..4 scope (see
``test/elastic-sbs/docs/elasticjuicer_ray_partitioned_complementarity_analysis_20260721.md``):

- Three sizes must never be conflated: ``partition_rows`` (checkpoint/fault
  granularity) >= Ray Data outer map batch >= ElasticJuicer executable
  micro-batch.
- Authority boundaries: the partitioned executor owns partition boundaries and
  checkpoints; the actor-local controller owns the next micro-batch; Captain
  quotas and stage profiles are advisory.

Validated here on real Ray:

1. per-partition adaptive OOM retry is lossless (0 missing / 0 duplicate,
   value correctness) across partitions;
2. all partitions reuse the same job-scoped metrics sink / control service;
3. checkpoint resume skips completed op groups without re-running operators;
4. after a simulated failure (deleting the newest checkpoint) only the
   un-checkpointed op group is recomputed, still through the adaptive path;
5. PR-RP-EJ-2: the partitioned executor owns one job-scoped Captain lifecycle
   spanning all partitions, closed in ``finally``;
6. PR-RP-EJ-3: the same operator instance keeps one stable stage id across
   partitions;
7. PR-RP-EJ-4: later incarnations inherit learned OOM bounds through stage
   profiles instead of re-probing from scratch.
"""

import glob
import os
import shutil
import time
import unittest
import uuid

from loguru import logger

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops.base_op import Mapper
from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


def _real_ray_available():
    try:
        import ray
        import ray.data
    except ImportError:
        return False
    return bool(getattr(ray, "__version__", None)) and callable(getattr(getattr(ray, "data", None), "from_items", None))


class PartitionedThresholdMapper(Mapper):
    """CPU-safe stand-in for a batched CUDA Mapper with a memory limit."""

    _batched_op = True

    def __init__(self, oom_above=8, increment=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oom_above = oom_above
        self.increment = increment

    def process_batched(self, samples):
        if len(samples["value"]) > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        return {"value": [value + self.increment for value in samples["value"]]}

    def use_ray_actor(self):
        return True


@unittest.skipUnless(_real_ray_available(), "real Ray is required for the joint partitioned E2E")
class ElasticJuicerRayPartitionedAdaptiveE2ETest(DataJuicerTestCaseBase):
    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")

    def setUp(self) -> None:
        super().setUp()
        unique_name = f"test_ray_partitioned_adaptive_{uuid.uuid4().hex[:8]}"
        self.tmp_dir = os.path.join(self.root_path, "tmp", unique_name)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.tmp_dir, "checkpoints")

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _elastic_cfg(self, job_id, **overrides):
        # Shared job-scoped config dict: every partition RayDataset built from it
        # must reuse the same metrics sink / control service handles. Profile
        # seeding defaults off here so each test opts in explicitly.
        cfg = {
            "elastic_juicer_adaptive_batching": True,
            "job_id": job_id,
            "auto_op_parallelism": False,
            "elastic_juicer_profile_seed": False,
        }
        cfg.update(overrides)
        return cfg

    def _build_executor(self, cfg, num_partitions=2, checkpoint_enabled=False, max_concurrent_partitions=None):
        # Instantiate only the joint data path under test; dataset loading,
        # export and event logging are out of PR-RP-EJ-1 scope.
        executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
        executor.cfg = cfg
        executor.executor_type = "ray_partitioned"
        executor.num_partitions = num_partitions
        # Concurrent partition execution (upstream #1022) reads this before
        # submitting partition worker threads. Default lets every partition
        # run at once; sequential-semantics tests pin it to 1 explicitly.
        executor.max_concurrent_partitions = (
            max_concurrent_partitions if max_concurrent_partitions is not None else num_partitions
        )
        executor.pipeline_dag = None
        executor.event_logger = None  # EventLoggingMixin._log_event degrades to a no-op
        executor.ckpt_manager = RayCheckpointManager(
            ckpt_dir=self.ckpt_dir,
            checkpoint_enabled=checkpoint_enabled,
            checkpoint_strategy=CheckpointStrategy.EVERY_OP if checkpoint_enabled else CheckpointStrategy.DISABLED,
        )
        return executor

    def _operator(self, name, increment):
        operator = PartitionedThresholdMapper(
            oom_above=8,
            increment=increment,
            batch_size=32,
            num_proc=2,
            auto_op_parallelism=False,
            ray_execution_mode="actor",
            skip_op_error=True,
        )
        operator._name = name
        return operator

    def _source_dataset(self, values, cfg):
        import ray

        source = ray.data.from_items([{"value": value} for value in values], override_num_blocks=4)
        return RayDataset(source, cfg=cfg, auto_op_parallelism=False)

    def _shutdown_job_services(self, cfg):
        import ray

        for key in ("_elastic_juicer_metrics_sink", "_elastic_juicer_control_service"):
            handle = cfg.get(key)
            if handle is not None:
                ray.kill(handle, no_restart=True)

    def _wait_for_complete_metrics(self, sink_handle, timeout_seconds=15.0):
        import ray

        deadline = time.monotonic() + timeout_seconds
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = ray.get(sink_handle.snapshot.remote())
            events = snapshot["events"]
            if any(not event.snapshot.succeeded for event in events) and any(
                event.snapshot.succeeded for event in events
            ):
                return snapshot
            time.sleep(0.05)
        self.fail(f"timed out waiting for OOM and success metrics; last snapshot={snapshot}")

    @TEST_TAG("ray")
    def test_partitioned_adaptive_is_lossless_and_shares_job_scoped_services(self):
        values = list(range(80))
        cfg = self._elastic_cfg("ej-rp-lossless")
        executor = self._build_executor(
            cfg,
            num_partitions=2,
            checkpoint_enabled=False,
            # Sequential partitions keep the per-partition OOM/retry event
            # ordering this losslessness assertion relies on.
            max_concurrent_partitions=1,
        )
        source = self._source_dataset(values, cfg)
        operator = self._operator("rp_threshold_single", increment=3)

        try:
            started_at = time.perf_counter()
            result = executor._process_with_simple_partitioning(source, [operator])
            rows = result.data.take_all()
            elapsed_seconds = time.perf_counter() - started_at
            logger.info(
                "ElasticJuicer joint partitioned diagnostics: "
                f"rows={len(rows)}, elapsed_seconds={elapsed_seconds:.4f}"
            )

            # Lossless across partitions: 0 missing, 0 duplicate, value correctness.
            self.assertEqual(sorted(row["value"] for row in rows), [value + 3 for value in values])

            metrics = self._wait_for_complete_metrics(cfg["_elastic_juicer_metrics_sink"])
            self.assertGreater(metrics["received_events"], 0)
            # Outer batch (32) > oom_above (8): every partition must have hit the
            # adaptive OOM retry path and then succeeded with smaller micro-batches.
            self.assertTrue(any(not event.snapshot.succeeded for event in metrics["events"]))
            self.assertTrue(any(event.snapshot.succeeded for event in metrics["events"]))
            self.assertTrue(all(event.job_id == "ej-rp-lossless" for event in metrics["events"]))
            self.assertTrue(all(event.op_name == "rp_threshold_single" for event in metrics["events"]))

            # PR-RP-EJ-3: the same operator instance keeps one stable stage id
            # across partitions, so profiles and quotas address one logical stage
            # (previously each partition re-registered a fresh stage, section 5.4).
            stage_ids = {event.stage_id for event in metrics["events"]}
            self.assertEqual(len(stage_ids), 1)
            self.assertTrue(all(stage_id.endswith(":rp_threshold_single") for stage_id in stage_ids))

            # Job-scoped services must be shared, not per-partition.
            self.assertIsNotNone(cfg.get("_elastic_juicer_metrics_sink"))
            self.assertIsNotNone(cfg.get("_elastic_juicer_control_service"))
        finally:
            self._shutdown_job_services(cfg)

    @TEST_TAG("ray")
    def test_checkpoint_resume_skips_completed_groups_and_recomputes_only_missing_group(self):
        import ray

        values = list(range(60))
        expected = [value + 3 + 5 for value in values]

        # ---- Run 1: fresh job writes one checkpoint per op per partition. ----
        first_cfg = self._elastic_cfg("ej-rp-ckpt-first")
        executor = self._build_executor(first_cfg, num_partitions=2, checkpoint_enabled=True)
        try:
            result = executor._process_with_simple_partitioning(
                self._source_dataset(values, first_cfg),
                [
                    self._operator("rp_threshold_stage_one", increment=3),
                    self._operator("rp_threshold_stage_two", increment=5),
                ],
            )
            self.assertEqual(sorted(row["value"] for row in result.data.take_all()), expected)
            self._wait_for_complete_metrics(first_cfg["_elastic_juicer_metrics_sink"])
        finally:
            self._shutdown_job_services(first_cfg)

        checkpoint_files = sorted(
            os.path.basename(path) for path in glob.glob(os.path.join(self.ckpt_dir, "checkpoint_op_*.parquet"))
        )
        self.assertEqual(
            checkpoint_files,
            [
                "checkpoint_op_0000_partition_0000.parquet",
                "checkpoint_op_0000_partition_0001.parquet",
                "checkpoint_op_0001_partition_0000.parquet",
                "checkpoint_op_0001_partition_0001.parquet",
            ],
        )

        # ---- Run 2: full resume must skip every op group (no operator work). ----
        second_cfg = self._elastic_cfg("ej-rp-ckpt-second")
        executor = self._build_executor(second_cfg, num_partitions=2, checkpoint_enabled=True)
        try:
            result = executor._process_with_simple_partitioning(
                self._source_dataset(values, second_cfg),
                [
                    self._operator("rp_threshold_stage_one", increment=3),
                    self._operator("rp_threshold_stage_two", increment=5),
                ],
            )
            self.assertEqual(sorted(row["value"] for row in result.data.take_all()), expected)
            # If any operator had re-run on checkpointed data, values would be
            # over-incremented; additionally no metrics event may be produced.
            snapshot = ray.get(second_cfg["_elastic_juicer_metrics_sink"].snapshot.remote())
            self.assertEqual(snapshot["received_events"], 0)
        finally:
            self._shutdown_job_services(second_cfg)

        # ---- Run 3: drop the newest checkpoint to simulate a failure after
        # op group 1; only the second op group may be recomputed. ----
        for path in glob.glob(os.path.join(self.ckpt_dir, "checkpoint_op_0001_*.parquet")):
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)

        third_cfg = self._elastic_cfg("ej-rp-ckpt-third")
        executor = self._build_executor(third_cfg, num_partitions=2, checkpoint_enabled=True)
        try:
            result = executor._process_with_simple_partitioning(
                self._source_dataset(values, third_cfg),
                [
                    self._operator("rp_threshold_stage_one", increment=3),
                    self._operator("rp_threshold_stage_two", increment=5),
                ],
            )
            self.assertEqual(sorted(row["value"] for row in result.data.take_all()), expected)
            metrics = self._wait_for_complete_metrics(third_cfg["_elastic_juicer_metrics_sink"])
            # Only the un-checkpointed op group runs, and it still goes through
            # the adaptive OOM retry path.
            self.assertTrue(all(event.op_name == "rp_threshold_stage_two" for event in metrics["events"]))
            self.assertTrue(any(not event.snapshot.succeeded for event in metrics["events"]))
            self.assertTrue(any(event.snapshot.succeeded for event in metrics["events"]))
        finally:
            self._shutdown_job_services(third_cfg)

    @TEST_TAG("ray")
    def test_partitioned_executor_owns_one_job_scoped_captain_across_partitions(self):
        import ray

        values = list(range(80))
        # Watermarks are set far above any real usage: the Captain must act on
        # the actors' learned local OOM bounds, not on synthetic memory pressure.
        cfg = self._elastic_cfg(
            "ej-rp-captain",
            elastic_juicer_captain_enabled=True,
            elastic_juicer_captain_process_memory_high_mb=1_000_000_000,
            elastic_juicer_captain_process_memory_low_mb=999_999_999,
            elastic_juicer_captain_poll_interval_sec=0.05,
        )
        executor = self._build_executor(cfg, num_partitions=2, checkpoint_enabled=False)
        source = self._source_dataset(values, cfg)
        operator = self._operator("rp_threshold_captain", increment=2)

        try:
            # Captain ownership lives in ``_run_impl`` so one lifecycle also
            # covers convergence and export; mirror that wrapping here around
            # the internal partitioned data path under test.
            captain_lifecycle = source.start_elastic_juicer_captain(cfg)
            self.assertIsNotNone(captain_lifecycle)
            try:
                result = executor._process_with_simple_partitioning(source, [operator])
                rows = result.data.take_all()
            finally:
                captain_lifecycle.close()

            # The joint run stays lossless with the Captain in the loop.
            self.assertEqual(sorted(row["value"] for row in rows), [value + 2 for value in values])

            # PR-RP-EJ-2: exactly one job-scoped lifecycle spans all partitions;
            # it polled while partitions ran and was closed in ``finally``.
            lifecycle = cfg.get("_elastic_juicer_captain_lifecycle")
            self.assertIs(lifecycle, captain_lifecycle)
            lifecycle_snapshot = lifecycle.snapshot()
            self.assertFalse(lifecycle_snapshot["running"])
            self.assertGreater(lifecycle_snapshot["polls"], 0)

            # The Captain observed actor OOM bounds (outer batch 32 > oom_above 8)
            # and published at least one accepted shrink quota job-wide.
            control_snapshot = ray.get(cfg["_elastic_juicer_control_service"].snapshot.remote())
            self.assertGreaterEqual(control_snapshot["accepted_quotas"], 1)
        finally:
            self._shutdown_job_services(cfg)

    @TEST_TAG("ray")
    def test_later_partitions_inherit_learned_oom_bounds_via_stage_profiles(self):
        import ray

        # 128 rows / 2 partitions = 64 rows each = exactly two 32-row outer
        # batches, so every incarnation's first slice is deterministic: 32 for
        # unseeded actors (always OOMs above 8), the inherited safe size for
        # seeded ones (always succeeds).
        values = list(range(128))
        cfg = self._elastic_cfg("ej-rp-profile-seed", elastic_juicer_profile_seed=True)
        executor = self._build_executor(
            cfg,
            num_partitions=2,
            checkpoint_enabled=False,
            # Profile inheritance requires partition N+1 to start after
            # partition N reported its learned bounds; run sequentially.
            max_concurrent_partitions=1,
        )
        source = self._source_dataset(values, cfg)
        operator = self._operator("rp_threshold_seeded", increment=1)

        try:
            result = executor._process_with_simple_partitioning(source, [operator])
            rows = result.data.take_all()

            # Inheritance must never cost losslessness.
            self.assertEqual(sorted(row["value"] for row in rows), [value + 1 for value in values])

            metrics = self._wait_for_complete_metrics(cfg["_elastic_juicer_metrics_sink"])
            first_events = {}
            for event in metrics["events"]:
                current = first_events.get(event.actor_incarnation_id)
                if current is None or event.sequence < current.sequence:
                    first_events[event.actor_incarnation_id] = event

            # PR-RP-EJ-4: every incarnation either probes from scratch (first
            # slice fails at 32) or starts seeded (first slice succeeds while
            # already holding an inherited OOM upper bound).
            unseeded = [event for event in first_events.values() if not event.snapshot.succeeded]
            seeded = [
                event
                for event in first_events.values()
                if event.snapshot.succeeded
                and event.control is not None
                and event.control.local_oom_upper_bound is not None
            ]
            self.assertEqual(len(unseeded) + len(seeded), len(first_events))
            # Partition 1 actors start before any profile exists: they re-probe.
            self.assertGreaterEqual(len(unseeded), 1)
            # Partition 2 actors inherit the learned bounds instead of re-probing.
            self.assertGreaterEqual(len(seeded), 1)

            # PR-RP-EJ-3 + EJ-4: one stable stage id yields exactly one merged
            # profile. The proven safe size is exactly oom_above (8); the OOM
            # bound is the tightest observed failure, which upward growth
            # probes may tighten anywhere within (8, 16].
            control_snapshot = ray.get(cfg["_elastic_juicer_control_service"].snapshot.remote())
            profiles = control_snapshot["stage_profiles"]
            self.assertEqual(len(profiles), 1)
            self.assertTrue(profiles[0].stage_id.endswith(":rp_threshold_seeded"))
            self.assertEqual(profiles[0].safe_batch_size, 8)
            self.assertIsNotNone(profiles[0].oom_upper_bound)
            self.assertGreater(profiles[0].oom_upper_bound, 8)
            self.assertLessEqual(profiles[0].oom_upper_bound, 16)
        finally:
            self._shutdown_job_services(cfg)

    @TEST_TAG("ray")
    def test_concurrent_partitions_keep_one_stage_identity_and_seed_late_partition(self):
        import ray

        # Upstream #1022 deep-copies the operator graph per concurrent
        # partition. The executor must stamp clone-stable identities before
        # cloning so all partitions share one stage (profile merge target),
        # and a partition admitted after earlier ones reported bounds must
        # start seeded instead of re-probing.
        values = list(range(192))  # 3 partitions x 64 rows = two 32-row batches
        cfg = self._elastic_cfg("ej-rp-concurrent-seed", elastic_juicer_profile_seed=True)
        executor = self._build_executor(
            cfg,
            num_partitions=3,
            checkpoint_enabled=False,
            # Two slots for three partitions force one staggered admission,
            # giving the late partition a profile to inherit.
            max_concurrent_partitions=2,
        )
        source = self._source_dataset(values, cfg)
        operator = self._operator("rp_threshold_concurrent", increment=1)

        try:
            result = executor._process_with_simple_partitioning(source, [operator])
            rows = result.data.take_all()
            self.assertEqual(sorted(row["value"] for row in rows), [value + 1 for value in values])

            metrics = self._wait_for_complete_metrics(cfg["_elastic_juicer_metrics_sink"])
            stage_ids = {event.stage_id for event in metrics["events"]}
            # Clone-stable identity: exactly one stage across all partitions.
            self.assertEqual(len(stage_ids), 1)

            first_events = {}
            for event in metrics["events"]:
                current = first_events.get(event.actor_incarnation_id)
                if current is None or event.sequence < current.sequence:
                    first_events[event.actor_incarnation_id] = event
            seeded = [
                event
                for event in first_events.values()
                if event.snapshot.succeeded
                and event.control is not None
                and event.control.local_oom_upper_bound is not None
            ]
            # The staggered partition inherits bounds instead of re-probing.
            self.assertGreaterEqual(len(seeded), 1)

            control_snapshot = ray.get(cfg["_elastic_juicer_control_service"].snapshot.remote())
            profiles = control_snapshot["stage_profiles"]
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].safe_batch_size, 8)
        finally:
            self._shutdown_job_services(cfg)
