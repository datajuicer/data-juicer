"""Exercise the public partitioned executor dispatch with recovery units."""

import json
import sqlite3

import pytest
import ray
import yaml
from jsonargparse import Namespace

from data_juicer.config import init_configs
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.core.executor.unit_protocol import ProtocolError
from data_juicer.ops.base_op import Mapper
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper
from data_juicer.utils.ckpt_utils import CheckpointStrategy, RayCheckpointManager


class DuplicateMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        return {key: [value for item in values for value in (item, item)] for key, values in samples.items()}


@pytest.fixture
def local_ray():
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


def _executor(tmp_path, *, resume=False):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = Namespace(auto_op_parallelism=False)
    executor.ckpt_manager = RayCheckpointManager(
        str(tmp_path / "checkpoints"),
        checkpoint_enabled=True,
        checkpoint_strategy=CheckpointStrategy.EVERY_OP,
    )
    executor.job_id = "unit-integration"
    executor.recovery_mode = "streaming"
    executor.unit_size_cfg = 2
    executor._is_resuming = resume
    return executor


def _source(executor, *, changed=False):
    rows = [
        {"text": " A\tB ", "id": 0},
        {"text": "!!!", "id": 1},
        {"text": " C ", "id": 2},
        {"text": "???", "id": 3},
    ]
    if changed:
        rows[0]["text"] = "different input"
    return RayDataset(ray.data.from_items(rows, override_num_blocks=1), cfg=executor.cfg)


def test_streaming_dispatch_uses_units_for_mapper_filter_and_resumes(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1), TextLengthFilter(min_len=1, max_len=2, num_proc=1)]
    assert executor._should_use_streaming_recovery(_source(executor), ops)
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    assert sorted(row["id"] for row in result.data.take_all()) == [2]

    database = tmp_path / "checkpoints" / "unit_recovery" / "state" / "units.sqlite3"
    with sqlite3.connect(database) as db:
        committed_before = db.execute(
            "SELECT COUNT(*) FROM stage_state WHERE committed_attempt_id IS NOT NULL"
        ).fetchone()[0]
        attempts_before = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    assert committed_before == 4

    restarted = _executor(tmp_path, resume=True)
    restored = restarted._process_with_simple_partitioning(_source(restarted), ops)
    assert sorted(row["id"] for row in restored.data.take_all()) == [2]
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == attempts_before


def test_streaming_resume_rejects_changed_input(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1)]
    executor._process_with_simple_partitioning(_source(executor), ops)
    restarted = _executor(tmp_path, resume=True)
    with pytest.raises(ProtocolError, match="Input content or read order changed"):
        restarted._process_with_simple_partitioning(_source(restarted, changed=True), ops)


def test_streaming_resume_rejects_missing_snapshot_file(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1)]
    executor._process_with_simple_partitioning(_source(executor), ops)
    snapshot_file = tmp_path / "checkpoints" / "unit_recovery" / "source" / "unit-000000000000.parquet"
    snapshot_file.unlink()
    restarted = _executor(tmp_path, resume=True)
    with pytest.raises(ProtocolError, match="snapshot file is missing or modified"):
        restarted._process_with_simple_partitioning(_source(restarted), ops)


def test_streaming_export_keeps_expansion_and_input_unit_order(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [DuplicateMapper(num_proc=1, batch_size=1)]
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    assert [row["id"] for row in result.data.take_all()] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_streaming_all_filtered_keeps_schema(tmp_path, local_ray):
    executor = _executor(tmp_path)
    ops = [TextLengthFilter(min_len=100, num_proc=1)]
    result = executor._process_with_simple_partitioning(_source(executor), ops)
    assert result.data.take_all() == []
    assert "text" in result.data.schema().names


def test_full_executor_exports_streaming_unit_result(tmp_path, local_ray):
    input_path = tmp_path / "input.jsonl"
    with input_path.open("w") as stream:
        for text in [" A\tB ", "!!!", " C "]:
            stream.write(json.dumps({"text": text}) + "\n")
    config_path = tmp_path / "job.yaml"
    export_path = tmp_path / "output.jsonl"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project_name": "unit-integration",
                "executor_type": "ray_partitioned",
                "dataset_path": str(input_path),
                "export_path": str(export_path),
                "work_dir": str(tmp_path / "work"),
                "checkpoint_dir": str(tmp_path / "work" / "checkpoints"),
                "auto_op_parallelism": False,
                "strict_preflight": False,
                "checkpoint": {"enabled": True, "strategy": "every_op"},
                "partition": {"recovery_mode": "streaming", "unit_size": 2},
                "process": [
                    {"whitespace_normalization_mapper": {}},
                    {"text_length_filter": {"min_len": 1, "max_len": 2}},
                ],
            }
        )
    )
    cfg = init_configs(["--config", str(config_path)])
    executor = PartitionedRayExecutor(cfg)
    executor.run(skip_return=True)
    shards = list(export_path.glob("*.json"))
    assert shards
    assert [json.loads(line)["text"] for shard in shards for line in shard.read_text().splitlines()] == ["C"]
