import concurrent.futures
import time
import unittest

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer.captain import CaptainDecisionCore, CaptainRuntime
from data_juicer.core.elasticjuicer.quota import QuotaEnvelope, current_time_ms
from data_juicer.ops.base_op import Mapper
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


def _real_ray_available():
    try:
        import ray
        import ray.data
    except ImportError:
        return False
    return bool(getattr(ray, "__version__", None)) and callable(getattr(ray.data, "from_items", None))


class SlowIdentityMapper(Mapper):
    _batched_op = True

    def __init__(self, delay_seconds=0.005, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_seconds = delay_seconds

    def process_batched(self, samples):
        time.sleep(self.delay_seconds)
        return {"value": [value + 1 for value in samples["value"]]}

    def use_ray_actor(self):
        return True


@unittest.skipUnless(_real_ray_available(), "real Ray is required for unittest-dist quota E2E")
class ElasticJuicerRayQuotaE2ETest(DataJuicerTestCaseBase):
    def _make_job(self, job_id, row_count=4096, actors=2, captain_enabled=False, stages=1):
        import ray

        cfg = {
            "elastic_juicer_adaptive_batching": True,
            "elastic_juicer_control_poll_interval_sec": 0.01,
            "elastic_juicer_sample_interval_sec": 0.001,
            "job_id": job_id,
            "auto_op_parallelism": False,
            "elastic_juicer_captain_enabled": captain_enabled,
            "elastic_juicer_captain_process_memory_high_mb": 1,
            "elastic_juicer_captain_process_memory_low_mb": 0.5,
            "elastic_juicer_captain_min_confidence": 0.0,
            "elastic_juicer_captain_poll_interval_sec": 0.02,
        }
        source = ray.data.from_items([{"value": value} for value in range(row_count)], override_num_blocks=128)
        dataset = RayDataset(source, cfg=cfg, auto_op_parallelism=False)
        operators = [
            SlowIdentityMapper(
                batch_size=32,
                num_proc=actors,
                auto_op_parallelism=False,
                ray_execution_mode="actor",
            )
            for _ in range(stages)
        ]
        dataset.process(operators)
        if captain_enabled:
            dataset.start_elastic_juicer_captain(cfg)
        return dataset

    def _wait_for_registrations(self, control, expected=2, timeout=20):
        import ray

        deadline = time.monotonic() + timeout
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = ray.get(control.snapshot.remote())
            if len(snapshot["registrations"]) >= expected:
                return snapshot["registrations"]
            time.sleep(0.02)
        self.fail(f"timed out waiting for actor registrations; last snapshot={snapshot}")

    def _wait_for_quota_metrics(self, sink, revisions, timeout=20):
        import ray

        deadline = time.monotonic() + timeout
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = ray.get(sink.snapshot.remote())
            observed = {
                (event.actor_id, event.actor_incarnation_id)
                for event in snapshot["events"]
                if event.control is not None and event.control.quota_revision >= revisions.get(event.actor_id, 10**9)
            }
            if len(observed) >= len(revisions):
                return snapshot
            time.sleep(0.02)
        self.fail(f"timed out waiting for quota-applied metrics; last snapshot={snapshot}")

    @TEST_TAG("ray")
    def test_real_raydataset_actors_receive_distinct_service_caps(self):
        import ray

        dataset = self._make_job("quota-product-e2e")
        control = dataset.elastic_juicer_control_service
        sink = dataset.elastic_juicer_metrics_sink
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(dataset.data.take_all)
        try:
            registrations = sorted(self._wait_for_registrations(control), key=lambda item: item.actor_id)
            caps = [5, 9]
            revisions = {}
            now_ms = current_time_ms()
            for registration, cap in zip(registrations, caps):
                envelope = QuotaEnvelope(
                    job_id=registration.job_id,
                    actor_id=registration.actor_id,
                    actor_incarnation_id=registration.actor_incarnation_id,
                    revision=1,
                    issued_at_ms=now_ms,
                    expires_at_ms=now_ms + 30_000,
                    max_batch_size=cap,
                    reason="product-e2e",
                )
                result = ray.get(control.publish_quota.remote(envelope))
                self.assertTrue(result.accepted)
                revisions[registration.actor_id] = 1

            rows = future.result(timeout=60)
            metrics = self._wait_for_quota_metrics(sink, revisions)
            self.assertEqual(sorted(row["value"] for row in rows), list(range(1, 4097)))
            self.assertEqual(len(rows), len({row["value"] for row in rows}))
            cap_by_actor = {registration.actor_id: cap for registration, cap in zip(registrations, caps)}
            applied = [
                event for event in metrics["events"] if event.control is not None and event.control.quota_revision >= 1
            ]
            self.assertTrue(applied)
            self.assertTrue(all(event.snapshot.batch_size <= cap_by_actor[event.actor_id] for event in applied))
            self.assertTrue(all(event.control.hard_limit == cap_by_actor[event.actor_id] for event in applied))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            ray.kill(sink, no_restart=True)
            ray.kill(control, no_restart=True)

    @TEST_TAG("ray")
    def test_single_stage_captain_closed_loop_preserves_values(self):
        import ray

        dataset = self._make_job("captain-product-e2e")
        control = dataset.elastic_juicer_control_service
        sink = dataset.elastic_juicer_metrics_sink
        captain = CaptainRuntime(
            CaptainDecisionCore(
                metrics_ttl_ms=5_000,
                min_decision_interval_ms=50,
                process_memory_high_mb=1,
                process_memory_low_mb=0.5,
            ),
            sink,
            control,
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(dataset.data.take_all)
        decisions = []
        try:
            self._wait_for_registrations(control)
            deadline = time.monotonic() + 60
            while not future.done() and time.monotonic() < deadline:
                decisions.extend(captain.poll_once())
                time.sleep(0.02)
            rows = future.result(timeout=10)
            metrics = ray.get(sink.snapshot.remote())

            self.assertTrue(decisions)
            self.assertTrue(all(decision.reason == "memory_high_watermark" for decision in decisions))
            self.assertEqual(sorted(row["value"] for row in rows), list(range(1, 4097)))
            self.assertEqual(len(rows), len({row["value"] for row in rows}))
            self.assertTrue(
                any(event.control is not None and event.control.quota_revision > 0 for event in metrics["events"])
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            ray.kill(sink, no_restart=True)
            ray.kill(control, no_restart=True)

    @TEST_TAG("ray")
    def test_control_service_failure_does_not_interrupt_real_raydataset(self):
        import ray

        dataset = self._make_job("control-failure-e2e", row_count=2048)
        control = dataset.elastic_juicer_control_service
        sink = dataset.elastic_juicer_metrics_sink
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(dataset.data.take_all)
        try:
            self._wait_for_registrations(control)
            ray.kill(control, no_restart=True)
            rows = future.result(timeout=60)
            self.assertEqual(sorted(row["value"] for row in rows), list(range(1, 2049)))
            self.assertEqual(len(rows), len({row["value"] for row in rows}))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            ray.kill(sink, no_restart=True)

    @TEST_TAG("ray")
    def test_product_captain_lifecycle_drives_closed_loop_without_manual_poll(self):
        import ray

        dataset = self._make_job("captain-lifecycle-e2e", row_count=8192, captain_enabled=True)
        control = dataset.elastic_juicer_control_service
        sink = dataset.elastic_juicer_metrics_sink
        lifecycle = dataset.elastic_juicer_captain_lifecycle
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(dataset.data.take_all)
        try:
            self._wait_for_registrations(control)
            rows = future.result(timeout=90)
            deadline = time.monotonic() + 10
            metrics = None
            while time.monotonic() < deadline:
                metrics = ray.get(sink.snapshot.remote())
                if any(event.control and event.control.quota_revision > 0 for event in metrics["events"]):
                    break
                time.sleep(0.02)

            self.assertEqual(sorted(row["value"] for row in rows), list(range(1, 8193)))
            self.assertEqual(len(rows), len({row["value"] for row in rows}))
            self.assertIsNotNone(metrics)
            self.assertTrue(any(event.control and event.control.quota_revision > 0 for event in metrics["events"]))
            self.assertGreater(lifecycle.snapshot()["polls"], 0)
        finally:
            dataset.close_elastic_juicer_captain()
            executor.shutdown(wait=False, cancel_futures=True)
            ray.kill(sink, no_restart=True)
            ray.kill(control, no_restart=True)

    @TEST_TAG("ray")
    def test_repeated_operator_stages_have_distinct_topology_identity(self):
        import ray

        dataset = self._make_job("repeated-stage-e2e", row_count=2048, stages=2)
        control = dataset.elastic_juicer_control_service
        sink = dataset.elastic_juicer_metrics_sink
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(dataset.data.take_all)
        try:
            registrations = self._wait_for_registrations(control, expected=4)
            rows = future.result(timeout=90)
            stage_ids = {registration.stage_id for registration in registrations}

            self.assertEqual(
                stage_ids,
                {
                    "stage-000000:SlowIdentityMapper",
                    "stage-000001:SlowIdentityMapper",
                },
            )
            self.assertEqual(sorted(row["value"] for row in rows), list(range(2, 2050)))
            self.assertEqual(len(rows), len({row["value"] for row in rows}))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            ray.kill(sink, no_restart=True)
            ray.kill(control, no_restart=True)
