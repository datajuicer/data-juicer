"""Gap 5.6: stage-profile persistence across driver restarts.

Before this bridge, StageProfiles lived only in the driver-memory
ControlService: after a crash + resume every partition re-probed the OOM
boundary from scratch (0803 kill-mode seedB=0). The bridge persists the
merged profiles atomically under the executor work_dir and restores them
into a fresh ControlService of the same pipeline (fingerprint-guarded,
TTL-expired at read time). Restored profiles remain advisory priors per
RFC 4.5: local OOM evidence always overrides them.
"""

import os
import shutil
import tempfile
import time
import unittest

from data_juicer.core.elasticjuicer.control_service import (
    STAGE_PROFILE_SCHEMA_VERSION,
    ControlService,
    StageProfile,
    load_stage_profiles,
    save_stage_profiles,
)
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase

from .test_ray_partitioned_adaptive_e2e import PartitionedThresholdMapper


def _profile(job_id="job-a", stage_id="stage-0000-occ0-abcd1234:op_x", age_ms=0):
    return StageProfile(
        job_id=job_id,
        stage_id=stage_id,
        op_name="op_x",
        safe_batch_size=8,
        oom_upper_bound=16,
        observed_at_ms=int(time.time() * 1000) - age_ms,
        op_fingerprint="abcd1234",
    )


class StageProfilePersistenceUnitTest(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ej_profile_persist_")
        self.path = os.path.join(self.tmp, "stage_profiles.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip_restores_profiles_into_new_service(self):
        service_a = ControlService(
            "job-a", profiles_path=self.path, pipeline_fingerprint="pipe-fp"
        )
        service_a.report_stage_profile(_profile())
        self.assertTrue(os.path.exists(self.path))

        # New driver (different job id), same pipeline + work dir.
        service_b = ControlService(
            "job-b", profiles_path=self.path, pipeline_fingerprint="pipe-fp"
        )
        self.assertEqual(service_b._restored_profiles, 1)
        restored = service_b.get_stage_profile("job-b", _profile().stage_id, op_fingerprint="abcd1234")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.safe_batch_size, 8)
        self.assertEqual(restored.oom_upper_bound, 16)

    def test_fingerprint_mismatch_fails_closed(self):
        service_a = ControlService(
            "job-a", profiles_path=self.path, pipeline_fingerprint="pipe-fp"
        )
        service_a.report_stage_profile(_profile())

        service_b = ControlService(
            "job-b", profiles_path=self.path, pipeline_fingerprint="other-pipeline"
        )
        self.assertEqual(service_b._restored_profiles, 0)
        self.assertIsNone(
            service_b.get_stage_profile("job-b", _profile().stage_id, op_fingerprint="abcd1234")
        )

    def test_corrupt_file_fails_closed(self):
        with open(self.path, "w") as stream:
            stream.write("{not json")
        service = ControlService(
            "job-b", profiles_path=self.path, pipeline_fingerprint="pipe-fp"
        )
        self.assertEqual(service._restored_profiles, 0)

    def test_ttl_expires_restored_profiles(self):
        service_a = ControlService(
            "job-a", profiles_path=self.path, pipeline_fingerprint="pipe-fp"
        )
        service_a.report_stage_profile(_profile(age_ms=60_000))

        service_b = ControlService(
            "job-b",
            profile_ttl_ms=1_000,
            profiles_path=self.path,
            pipeline_fingerprint="pipe-fp",
        )
        self.assertEqual(service_b._restored_profiles, 1)
        self.assertIsNone(
            service_b.get_stage_profile("job-b", _profile().stage_id, op_fingerprint="abcd1234")
        )

    def test_merge_then_persist_keeps_tightest_bounds(self):
        service = ControlService(
            "job-a", profiles_path=self.path, pipeline_fingerprint="pipe-fp"
        )
        service.report_stage_profile(_profile())
        tighter = StageProfile(
            job_id="job-a",
            stage_id=_profile().stage_id,
            op_name="op_x",
            safe_batch_size=12,
            oom_upper_bound=13,
            observed_at_ms=int(time.time() * 1000),
            op_fingerprint="abcd1234",
        )
        service.report_stage_profile(tighter)
        reloaded = load_stage_profiles(self.path, "pipe-fp")
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0].oom_upper_bound, 13)  # tightest OOM wins
        self.assertEqual(reloaded[0].safe_batch_size, 12)  # capped under bound


class StageProfileCrossDriverE2ETest(DataJuicerTestCaseBase):
    """A resumed driver (fresh ControlService) seeds from persisted profiles."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="ej_profile_e2e_")
        self.work_dir = os.path.join(self.tmp, "work")
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def _elastic_cfg(self, job_id):
        return {
            "elastic_juicer_adaptive_batching": True,
            "elastic_juicer_profile_seed": True,
            "job_id": job_id,
            "auto_op_parallelism": False,
        }

    def _build_executor(self, cfg):
        from data_juicer.core.executor.ray_executor_partitioned import (
            PartitionedRayExecutor,
        )
        from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager

        executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
        executor.cfg = cfg
        executor.executor_type = "ray_partitioned"
        executor.num_partitions = 1
        executor.max_concurrent_partitions = 1
        executor.pipeline_dag = None
        executor.event_logger = None
        executor.work_dir = self.work_dir
        executor.ckpt_manager = RayCheckpointManager(
            ckpt_dir=os.path.join(self.tmp, "checkpoints"),
            checkpoint_enabled=False,
            checkpoint_strategy=CheckpointStrategy.DISABLED,
        )
        return executor

    def _operator(self, name):
        op = PartitionedThresholdMapper(
            oom_above=8,
            increment=1,
            batch_size=32,
            num_proc=1,
            auto_op_parallelism=False,
            ray_execution_mode="actor",
            skip_op_error=False,
        )
        op._name = name
        return op

    def _source_dataset(self, cfg):
        import ray

        from data_juicer.core.data.ray_dataset import RayDataset

        source = ray.data.from_items([{"value": v} for v in range(64)], override_num_blocks=2)
        return RayDataset(source, cfg=cfg, auto_op_parallelism=False)

    def _shutdown_services(self, cfg):
        import ray

        for key in ("_elastic_juicer_metrics_sink", "_elastic_juicer_control_service"):
            handle = cfg.get(key)
            if handle is not None:
                try:
                    ray.kill(handle, no_restart=True)
                except Exception:
                    pass

    @TEST_TAG("ray")
    def test_resumed_driver_seeds_from_persisted_profiles(self):
        import ray

        values = list(range(64))

        # Driver 1: learns the OOM boundary and persists profiles.
        cfg1 = self._elastic_cfg("persist-driver-1")
        executor1 = self._build_executor(cfg1)
        result1 = executor1._process_with_simple_partitioning(
            self._source_dataset(cfg1), [self._operator("persist_threshold")]
        )
        rows1 = sorted(row["value"] for row in result1.data.take_all())
        self.assertEqual(rows1, [v + 1 for v in values])
        self._shutdown_services(cfg1)

        profiles_path = os.path.join(self.work_dir, "elastic_juicer_stage_profiles.json")
        self.assertTrue(os.path.exists(profiles_path), "profiles must be persisted after run 1")

        # Driver 2: fresh cfg / job id / ControlService, same pipeline + work dir.
        cfg2 = self._elastic_cfg("persist-driver-2")
        executor2 = self._build_executor(cfg2)
        result2 = executor2._process_with_simple_partitioning(
            self._source_dataset(cfg2), [self._operator("persist_threshold")]
        )
        rows2 = sorted(row["value"] for row in result2.data.take_all())
        self.assertEqual(rows2, [v + 1 for v in values])

        snapshot = ray.get(cfg2["_elastic_juicer_metrics_sink"].snapshot.remote(), timeout=30)
        first_events = {}
        for event in snapshot["events"]:
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
        self.assertGreaterEqual(
            len(seeded),
            1,
            "resumed driver's first incarnation must start seeded from persisted profiles",
        )
        control_snapshot = ray.get(cfg2["_elastic_juicer_control_service"].snapshot.remote())
        self.assertGreaterEqual(control_snapshot.get("restored_profiles", 0), 1)
        self._shutdown_services(cfg2)


if __name__ == "__main__":
    unittest.main()
