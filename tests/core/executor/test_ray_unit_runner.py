"""Real Ray actor and Data-Juicer operator recovery-unit tests."""

import hashlib
import os
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import ray

from data_juicer.core.executor.operator_unit_adapter import operator_unit_stage
from data_juicer.core.executor.ray_unit_runner import RayStage, run_ray_units
from data_juicer.core.executor.unit_protocol import LocalUnitStore
from data_juicer.core.executor.unit_runner import UnitStage, run_local_units
from data_juicer.ops.base_op import Filter, Mapper
from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper
from data_juicer.utils.constant import Fields


def same_schema(schema):
    return schema


@dataclass
class ParquetRangeSource:
    path: str

    def __call__(self, start, stop):
        yield pq.read_table(self.path).slice(start, stop - start)


class DuplicateMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        return {key: [value for item in values for value in (item, item)] for key, values in samples.items()}


class KeepEvenFilter(Filter):
    _batched_op = True

    def compute_stats_batched(self, samples):
        for value, stats in zip(samples["value"], samples[Fields.stats]):
            stats["even"] = value % 2 == 0
        return samples

    def process_batched(self, samples):
        return [stats["even"] for stats in samples[Fields.stats]]


def _planned(tmp_path, rows, stages, unit_size=1):
    source_path = tmp_path / "source.parquet"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    store = LocalUnitStore(str(tmp_path / "recover"), "job-1")
    units = store.plan(
        input_fingerprint=source_hash,
        config_fingerprint="whitespace-then-alphanumeric-v1",
        input_rows=rows,
        unit_size=unit_size,
        stage_count=stages,
    )
    return store, units, ParquetRangeSource(str(source_path))


def test_actual_mapper_filter_keep_schema_for_empty_unit(tmp_path):
    pq.write_table(pa.table({"text": [" A\tB ", "!!!", " C "]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 3, 2)
    stages = [
        operator_unit_stage(WhitespaceNormalizationMapper(), empty_schema=same_schema),
        operator_unit_stage(AlphanumericFilter(min_ratio=0.5)),
    ]
    result = run_local_units(store, units, source, stages)
    assert [text for block in result for text in block["text"].to_pylist()] == ["A B", "C"]
    assert store.get_commit(units[1].unit_id, 1).rows == 0
    assert Fields.stats in store.get_commit(units[1].unit_id, 1).schema.names
    assert [text for block in run_local_units(store, units, source, stages) for text in block["text"].to_pylist()] == [
        "A B",
        "C",
    ]


def with_worker_pid(table):
    yield table.append_column("worker_pid", pa.array([os.getpid()] * table.num_rows))


def pid_schema(schema):
    return schema.append(pa.field("worker_pid", pa.int64()))


@dataclass
class FailSecondUnitOnce:
    marker: str

    def __call__(self, table):
        if table["value"][0].as_py() == 1 and not os.path.exists(self.marker):
            with open(self.marker, "w") as stream:
                stream.write("attempted")
            raise RuntimeError("worker failed inside unit")
        return (table,)


@pytest.fixture
def local_ray():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    try:
        yield
    finally:
        ray.shutdown()


def test_ray_actor_reused_across_units_and_commits_in_input_order(tmp_path, local_ray):
    pq.write_table(pa.table({"value": [0, 1, 2, 3]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 4, 1)
    stage = RayStage(UnitStage(with_worker_pid, pid_schema), actors=1)
    result = run_ray_units(store, units, source, [stage])
    assert [value for block in result for value in block["value"].to_pylist()] == [0, 1, 2, 3]
    assert len({pid for block in result for pid in block["worker_pid"].to_pylist()}) == 1
    assert all(store.get_commit(unit.unit_id, 0) is not None for unit in units)
    reopened, same_units, same_source = _planned(tmp_path, 4, 1)
    restored = run_ray_units(reopened, same_units, same_source, [stage])
    assert [block.to_pylist() for block in restored] == [block.to_pylist() for block in result]


def test_ray_runner_with_real_mapper_and_filter(tmp_path, local_ray):
    pq.write_table(pa.table({"text": [" A\tB ", "!!!", " C "]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 3, 2)
    stages = [
        RayStage(operator_unit_stage(WhitespaceNormalizationMapper(), empty_schema=same_schema), actors=1),
        RayStage(operator_unit_stage(AlphanumericFilter(min_ratio=0.5)), actors=1),
    ]
    result = run_ray_units(store, units, source, stages)
    assert [text for block in result for text in block["text"].to_pylist()] == ["A B", "C"]
    assert store.get_commit(units[1].unit_id, 1).rows == 0


def test_ray_mapper_expansion_and_filter_zero_output(tmp_path, local_ray):
    pq.write_table(pa.table({"value": [0, 1, 2]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 3, 2)
    stages = [
        RayStage(operator_unit_stage(DuplicateMapper(), empty_schema=same_schema), actors=1),
        RayStage(operator_unit_stage(KeepEvenFilter()), actors=1),
    ]
    result = run_ray_units(store, units, source, stages)
    assert [value for block in result for value in block["value"].to_pylist()] == [0, 0, 2, 2]
    assert [store.get_commit(unit.unit_id, 0).rows for unit in units] == [2, 2, 2]
    assert [store.get_commit(unit.unit_id, 1).rows for unit in units] == [2, 0, 2]


def test_ray_failure_replays_only_uncommitted_unit(tmp_path, local_ray):
    pq.write_table(pa.table({"value": [0, 1]}), tmp_path / "source.parquet")
    store, units, source = _planned(tmp_path, 2, 1)
    stage = RayStage(UnitStage(FailSecondUnitOnce(str(tmp_path / "failed-once")), same_schema), actors=1)
    with pytest.raises(ray.exceptions.RayTaskError, match="worker failed inside unit"):
        run_ray_units(store, units, source, [stage])
    first = store.get_commit(units[0].unit_id, 0)
    assert first is not None
    assert store.get_commit(units[1].unit_id, 0) is None
    reopened, same_units, same_source = _planned(tmp_path, 2, 1)
    result = run_ray_units(reopened, same_units, same_source, [stage])
    assert reopened.get_commit(units[0].unit_id, 0) == first
    assert [value for block in result for value in block["value"].to_pylist()] == [0, 1]
