"""Gate 0 section 4.7: E2E through the public ``PartitionedRayExecutor.run()``.

Everything the sibling suites validate through internal seams must also hold
when a job enters through the real public path: ``init_configs`` parsing, the
DatasetBuilder, ``_prepare_operators`` (stage identity stamping + persisted
manifest), Captain ownership inside ``_run_impl``, per-partition adaptive
batching, cross-partition profile seeding, and export.

Proven here on real Ray from one public ``run()`` call:

1. the run is lossless end to end (every row exported exactly once, mapped);
2. the stage identity manifest is persisted in ``work_dir`` and every metrics
   event carries exactly that stamped stage id;
3. events from different partitions carry distinct ``partition_id``;
4. Captain quotas take effect in-band: events exist with
   ``control.quota_revision > 0`` and every such executed slice stayed within
   the applied hard limit;
5. partition-2 actors inherit partition-1's learned stage profile instead of
   re-probing from scratch.
"""

import glob
import json
import os
import shutil
import sys
import time
import unittest
import uuid

from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


def _real_ray_available():
    try:
        import ray
        import ray.data
    except ImportError:
        return False
    return bool(getattr(ray, "__version__", None)) and callable(getattr(getattr(ray, "data", None), "from_items", None))


@OPERATORS.register_module("ej_public_run_threshold_mapper")
class EJPublicRunThresholdMapper(Mapper):
    """CPU-safe stand-in for a batched CUDA Mapper with a memory limit."""

    _batched_op = True

    def __init__(self, oom_above: int = 8, slice_latency_sec: float = 0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oom_above = oom_above
        self.slice_latency_sec = slice_latency_sec

    def process_batched(self, samples):
        if len(samples["text"]) > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        # Keep each slice measurably slow so Captain quotas land while the
        # run is still executing (the in-band effect under test).
        time.sleep(self.slice_latency_sec)
        return {"text": [text + "|processed" for text in samples["text"]]}

    def use_ray_actor(self):
        return True


@unittest.skipUnless(_real_ray_available(), "real Ray is required for the public run() E2E")
class ElasticJuicerPublicRunE2ETest(DataJuicerTestCaseBase):
    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")

    NUM_ROWS = 320

    def setUp(self) -> None:
        super().setUp()
        unique_name = f"test_ej_public_run_{uuid.uuid4().hex[:8]}"
        self.tmp_dir = os.path.join(self.root_path, "tmp", unique_name)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.work_dir = os.path.join(self.tmp_dir, "work")
        self.export_path = os.path.join(self.tmp_dir, "res.jsonl")
        self.input_path = os.path.join(self.tmp_dir, "input.jsonl")
        with open(self.input_path, "w") as f:
            for index in range(self.NUM_ROWS):
                f.write(json.dumps({"text": f"sample-{index:04d}"}) + "\n")
        self.config_path = os.path.join(self.tmp_dir, "config.yaml")
        with open(self.config_path, "w") as f:
            f.write(
                f"""
project_name: 'ej-public-run-e2e'
dataset_path: '{self.input_path}'
export_path: '{self.export_path}'
executor_type: 'ray'
ray_address: 'auto'
auto_op_parallelism: false
partition:
  mode: manual
  num_of_partitions: 2
elastic_juicer_adaptive_batching: true
elastic_juicer_profile_seed: true
elastic_juicer_captain_enabled: true
elastic_juicer_captain_process_memory_high_mb: 1000000000
elastic_juicer_captain_process_memory_low_mb: 999999999
elastic_juicer_captain_poll_interval_sec: 0.02
elastic_juicer_control_poll_interval_sec: 0.02
process:
  - ej_public_run_threshold_mapper:
      oom_above: 8
      slice_latency_sec: 0.02
      batch_size: 32
      num_proc: 2
      ray_execution_mode: 'actor'
"""
            )

    def tearDown(self) -> None:
        super().tearDown()
        # init_configs installed loguru file sinks inside tmp_dir; close them
        # before deleting the tree (open files break rmtree on NFS mounts).
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _shutdown_job_services(self, cfg):
        import ray

        for key in ("_elastic_juicer_metrics_sink", "_elastic_juicer_control_service"):
            handle = cfg.get(key)
            if handle is not None:
                ray.kill(handle, no_restart=True)

    def _read_exported_rows(self):
        paths = [self.export_path] if os.path.isfile(self.export_path) else []
        paths += glob.glob(os.path.join(self.export_path, "**", "*"), recursive=True)
        rows = []
        for path in paths:
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        return rows

    def _wait_for_events(self, sink_handle, predicate, timeout_seconds=30.0):
        import ray

        deadline = time.monotonic() + timeout_seconds
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = ray.get(sink_handle.snapshot.remote())
            if predicate(snapshot["events"]):
                return snapshot
            time.sleep(0.1)
        self.fail(f"timed out waiting for expected metrics; last snapshot={snapshot}")

    @TEST_TAG("ray")
    def test_public_run_proves_quota_seed_partition_id_and_manifest(self):
        import ray

        cfg = init_configs(["--config", self.config_path])
        cfg.work_dir = self.work_dir
        executor = PartitionedRayExecutor(cfg)

        try:
            executor.run(skip_return=True)

            # 1. Lossless through the full public path, export included.
            rows = self._read_exported_rows()
            self.assertEqual(
                sorted(row["text"] for row in rows),
                [f"sample-{index:04d}|processed" for index in range(self.NUM_ROWS)],
            )

            # 2. The stage identity manifest was persisted by _prepare_operators.
            manifest_path = os.path.join(self.work_dir, "elastic_juicer_stage_identities.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.assertEqual(len(manifest["stages"]), 1)
            stage = manifest["stages"][0]
            self.assertRegex(
                stage["stage_id"],
                r"^stage-0000-occ0-[0-9a-f]{8}:ej_public_run_threshold_mapper$",
            )

            sink_handle = cfg.get("_elastic_juicer_metrics_sink")
            self.assertIsNotNone(sink_handle)

            def complete(events):
                partitions = {event.partition_id for event in events}
                return (
                    {0, 1} <= partitions
                    and any(not event.snapshot.succeeded for event in events)
                    and any(event.snapshot.succeeded for event in events)
                    and any(event.control is not None and event.control.quota_revision > 0 for event in events)
                )

            metrics = self._wait_for_events(sink_handle, complete)
            events = metrics["events"]

            # Every event addresses exactly the persisted stage identity.
            self.assertEqual({event.stage_id for event in events}, {stage["stage_id"]})

            # 3. Both partitions are distinguishable in-band via partition_id.
            self.assertEqual({event.partition_id for event in events}, {0, 1})

            # 4. Captain quotas took effect while the job was running: events
            # exist under a non-zero quota revision, and every slice executed
            # under a quota stayed within the applied hard limit.
            governed = [event for event in events if event.control is not None and event.control.quota_revision > 0]
            self.assertGreaterEqual(len(governed), 1)
            self.assertTrue(all(event.snapshot.batch_size <= event.control.hard_limit for event in governed))
            self.assertTrue(any(event.snapshot.succeeded for event in governed))
            control_snapshot = ray.get(cfg.get("_elastic_juicer_control_service").snapshot.remote())
            self.assertGreaterEqual(control_snapshot["accepted_quotas"], 1)

            # 5. Cross-partition inheritance: partition 0 probed from scratch,
            # one merged stage profile exists, and partition 1 actors started
            # seeded from it (first slice succeeds while already holding an
            # inherited OOM upper bound).
            first_events = {}
            for event in events:
                current = first_events.get(event.actor_incarnation_id)
                if current is None or event.sequence < current.sequence:
                    first_events[event.actor_incarnation_id] = event
            probed = [event for event in first_events.values() if not event.snapshot.succeeded]
            seeded = [
                event
                for event in first_events.values()
                if event.snapshot.succeeded
                and event.control is not None
                and event.control.local_oom_upper_bound is not None
            ]
            governed_first = [
                event
                for event in first_events.values()
                if event.snapshot.succeeded and event.control is not None and event.control.quota_revision > 0
            ]
            # Every incarnation either probed at the full outer batch, started
            # seeded, or had a Captain cap applied before its first slice.
            covered = {event.actor_incarnation_id for event in probed + seeded + governed_first}
            self.assertEqual(covered, set(first_events))
            self.assertGreaterEqual(len(probed), 1)
            self.assertTrue(all(event.partition_id == 0 for event in probed))
            seeded_partitions = {event.partition_id for event in seeded}
            self.assertIn(1, seeded_partitions)

            profiles = control_snapshot["stage_profiles"]
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].stage_id, stage["stage_id"])
            self.assertEqual(profiles[0].op_fingerprint, stage["op_fingerprint"])
            self.assertEqual(profiles[0].safe_batch_size, 8)
        finally:
            self._shutdown_job_services(cfg)
