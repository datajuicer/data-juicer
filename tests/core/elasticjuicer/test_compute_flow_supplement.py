"""Supplemental admission and lifecycle regression; GPU resources are logical."""

from dataclasses import dataclass
from types import SimpleNamespace

import pyarrow as pa
import ray

from data_juicer.core.elasticjuicer import partition_compute_flow as runtime
from data_juicer.core.executor.gpu_memory_probe import GPUMemoryProbe
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from tests.core.elasticjuicer.test_partition_compute_flow import assert_rows, recipe


def test_single_model_switch_preserves_actor_local_controllers(monkeypatch):
    events = []

    @dataclass
    class State:
        current_batch_size: int = 2

    class Runner:
        def __init__(self, stage_id):
            self.actor_id = stage_id
            self.op = SimpleNamespace(
                _name="compute_flow_lifecycle_test",
                elastic_juicer_load_model=lambda: events.append(("load", stage_id)),
                elastic_juicer_release_model=lambda: events.append(("release", stage_id)),
            )
            self.controller = SimpleNamespace(state=State())
            self.mapper = SimpleNamespace(oom_retries=0)

        def __call__(self, table):
            return table

    monkeypatch.setattr(runtime, "RayAdaptiveMapperActor", Runner)
    worker = runtime._ResidentWorker([dict(stage_id="a"), dict(stage_id="b")], 1024, max_resident_models=1)
    table = pa.table({runtime._ROW_COLUMN: [0, 1], "id": [12, 7]})
    a, b = worker.runners["a"], worker.runners["b"]
    controller_a = a.controller
    worker.process("a", table, False)
    a.controller.state.current_batch_size = 1
    a.mapper.oom_retries = 3
    worker.process("a", table, False)
    worker.process("b", table, False)
    out, metadata = worker.process("a", table, True)
    assert events == [("load", "a"), ("release", "a"), ("load", "b"), ("release", "b"), ("load", "a")]
    assert a.controller is controller_a and a.controller.state.current_batch_size == 1
    assert a.mapper.oom_retries == 3 and b.mapper.oom_retries == 0
    assert b.controller.state.current_batch_size == 2 and metadata["oom_retries"] == 0
    assert out["id"].to_pylist() == [12, 7] and runtime._ROW_COLUMN not in out.column_names


def test_two_ray_nodes_require_opt_in_and_share_fractional_pool(tmp_path, monkeypatch):
    from ray.cluster_utils import Cluster

    def measured(self, dataset, ops):
        for op in ops:
            if op.accelerator == "cuda":
                op._gpu_memory_fraction = 0.15

    monkeypatch.setattr(GPUMemoryProbe, "resolve", measured)
    assert not ray.is_initialized()
    cluster = Cluster()
    try:
        cluster.add_node(num_cpus=4, num_gpus=1, object_store_memory=128 * 1024**2, include_dashboard=False)
        cluster.add_node(num_cpus=4, num_gpus=1, object_store_memory=128 * 1024**2)
        ray.init(address=cluster.address, log_to_driver=False)
        for allowed in (False, True):
            directory = tmp_path / str(allowed)
            directory.mkdir()
            cfg = recipe(directory, checkpoint=False)
            cfg.partition.compute_flow = dict(
                enabled=True,
                workers=4,
                actors_per_gpu=2,
                allow_multi_node=allowed,
                batch_size=4,
                max_inflight_batches=8,
                startup_timeout_seconds=60,
            )
            assert_rows(PartitionedRayExecutor(cfg).run())
            summary = cfg._resolved_compute_flow_summary
            if not allowed:
                assert summary["fallbacks"] and not summary["pools"]
            else:
                assert not summary["fallbacks"]
                assert len(summary["pools"]) == 1
                pool = summary["pools"][0]
                assert pool["gpu_per_worker"] == 0.5 and pool["actors_per_gpu"] == 2
                assert len(pool["instances"]) == 4
                assert len({i["node_id"] for i in pool["instances"]}) == 2
    finally:
        ray.shutdown()
        cluster.shutdown()
