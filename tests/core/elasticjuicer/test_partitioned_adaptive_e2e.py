"""Real Ray transport, public executor, grouped checkpoints and resume.

The operator injects OOMs at a row threshold and needs no physical GPU.
"""

import json
from copy import deepcopy

import pytest
import yaml

from data_juicer.config import init_configs
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops.base_op import OPERATORS
from tests.core.elasticjuicer.test_ray_adaptive_mapper import ThresholdMapper


def test_public_partitioned_run_and_checkpoint_reuse(tmp_path, monkeypatch):
    ray = pytest.importorskip("ray")
    if ray.is_initialized():
        pytest.skip("This test needs its own isolated local Ray cluster")
    OPERATORS.register_module(ThresholdMapper._name)(ThresholdMapper)
    rows = [{"id": i, "text": str(i), "meta": {"attempts": 0}} for i in range(36)]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    export = tmp_path / "result.jsonl"
    op_cfg = {
        ThresholdMapper._name: {
            "batch_size": 8,
            "threshold": 2,
            "num_proc": 1,
            "num_cpus": 1,
            "num_gpus": 0.5,
            "ray_execution_mode": "actor",
            "adaptive_batching": True,
        }
    }
    recipe = {
        "project_name": "pr1054-ej-e2e",
        "executor_type": "ray_partitioned",
        "dataset_path": str(source),
        "export_path": str(export),
        "work_dir": str(tmp_path / "work"),
        "strict_preflight": False,
        "auto_op_parallelism": False,
        "elastic_juicer_adaptive_batching": True,
        "partition": {
            "mode": "manual",
            "num_of_partitions": 4,
            "execution_group_size": 2,
            "max_concurrent_partitions": 1,
            "gpu_preflight_enabled": False,
        },
        "checkpoint": {"enabled": True, "strategy": "every_op"},
        "process": [op_cfg, deepcopy(op_cfg)],
    }
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    ray.init(address="local", num_cpus=4, num_gpus=1, include_dashboard=False, object_store_memory=100 * 1024**2)
    try:
        cfg = init_configs(["--config", str(config_path)])
        executor = PartitionedRayExecutor(cfg)
        result = executor.run()
        actual = sorted(result.data.take_all(), key=lambda row: row["id"])
        assert [row["id"] for row in actual] == list(range(36))
        assert [row["value"] for row in actual] == [i * 2 for i in range(36)]
        assert all(row["meta"]["attempts"] == 2 for row in actual)
        assert all("__data_juicer_logical_partition_id__" not in row for row in actual)
        manifest = cfg._resolved_stage_identities
        assert len({entry["stage_id"] for entry in manifest["stages"]}) == 2
        assert cfg._resolved_execution_group_plan["execution_group_size"] == 2

        def forbidden_reprocessing(*args, **kwargs):
            raise AssertionError("Completed checkpoint rows must not run through the Mapper again")

        monkeypatch.setattr(RayDataset, "process", forbidden_reprocessing)
        cfg._resume_requested = True
        executor._is_resuming = True
        ops = executor._prepare_operators()
        assert cfg._resolved_stage_identities == manifest
        resumed = executor._process_with_simple_partitioning(executor.datasetbuilder.load_dataset(), ops)
        assert sorted(resumed.data.take_all(), key=lambda row: row["id"]) == actual
        files = [export] if export.is_file() else list(export.rglob("*.json*"))
        exported = [json.loads(line) for path in files for line in path.read_text().splitlines() if line]
        assert sorted(row["id"] for row in exported) == list(range(36))
    finally:
        ray.shutdown()
