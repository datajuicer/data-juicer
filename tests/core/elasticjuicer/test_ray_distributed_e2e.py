import time
import unittest

from loguru import logger

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.base_op import Mapper
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


def _real_ray_available():
    try:
        import ray
        import ray.data
    except ImportError:
        return False
    return bool(getattr(ray, "__version__", None)) and callable(getattr(getattr(ray, "data", None), "from_items", None))


class DistributedThresholdMapper(Mapper):
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


@unittest.skipUnless(_real_ray_available(), "real Ray is required for unittest-dist E2E")
class ElasticJuicerRayDistributedE2ETest(DataJuicerTestCaseBase):
    def _run_job(self, job_id, values):
        import ray

        cfg = {
            "elastic_juicer_adaptive_batching": True,
            "job_id": job_id,
            "auto_op_parallelism": False,
        }
        source = ray.data.from_items([{"value": value} for value in values], override_num_blocks=4)
        dataset = RayDataset(source, cfg=cfg, auto_op_parallelism=False)
        operator = DistributedThresholdMapper(
            oom_above=8,
            increment=3,
            batch_size=32,
            num_proc=2,
            auto_op_parallelism=False,
            ray_execution_mode="actor",
            skip_op_error=True,
        )

        started_at = time.perf_counter()
        dataset.process(operator)
        rows = dataset.data.take_all()
        elapsed_seconds = time.perf_counter() - started_at
        throughput = len(rows) / elapsed_seconds if elapsed_seconds > 0 else 0.0
        alive_nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
        logger.info(
            "ElasticJuicer unittest-dist diagnostics: "
            f"job_id={job_id}, alive_nodes={alive_nodes}, rows={len(rows)}, elapsed_seconds={elapsed_seconds:.4f}, "
            f"throughput={throughput:.2f} rows/s"
        )
        return rows, dataset.elastic_juicer_metrics_sink

    def _wait_for_complete_metrics(self, sink_handle, timeout_seconds=10.0):
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
    def test_adaptive_mapper_is_lossless_and_reports_metrics(self):
        import ray

        values = list(range(100))
        sink = None
        try:
            rows, sink = self._run_job("ej-dist-lossless", values)
            metrics = self._wait_for_complete_metrics(sink)

            self.assertEqual([row["value"] for row in rows], [value + 3 for value in values])
            self.assertGreater(metrics["received_events"], 0)
            self.assertTrue(any(not event.snapshot.succeeded for event in metrics["events"]))
            self.assertTrue(any(event.snapshot.succeeded for event in metrics["events"]))
            self.assertTrue(all(event.job_id == "ej-dist-lossless" for event in metrics["events"]))
        finally:
            if sink is not None:
                ray.kill(sink, no_restart=True)

    @TEST_TAG("ray")
    def test_metrics_sink_isolated_between_distributed_jobs(self):
        import ray

        first_sink = None
        second_sink = None
        try:
            first_rows, first_sink = self._run_job("ej-dist-job-a", list(range(40)))
            second_rows, second_sink = self._run_job("ej-dist-job-b", list(range(40, 80)))
            first_metrics = self._wait_for_complete_metrics(first_sink)
            second_metrics = self._wait_for_complete_metrics(second_sink)

            self.assertIsNot(first_sink, second_sink)
            self.assertEqual([row["value"] for row in first_rows], [value + 3 for value in range(40)])
            self.assertEqual([row["value"] for row in second_rows], [value + 3 for value in range(40, 80)])
            self.assertTrue(all(event.job_id == "ej-dist-job-a" for event in first_metrics["events"]))
            self.assertTrue(all(event.job_id == "ej-dist-job-b" for event in second_metrics["events"]))
        finally:
            if first_sink is not None:
                ray.kill(first_sink, no_restart=True)
            if second_sink is not None:
                ray.kill(second_sink, no_restart=True)
