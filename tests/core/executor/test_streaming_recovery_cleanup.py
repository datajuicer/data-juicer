"""Regression coverage for the opt-in, row-preserving stream recovery path."""

import json
from types import SimpleNamespace

import pyarrow as pa
import pytest
import ray
from jsonargparse import Namespace

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor.ray_executor_partitioned import (
    PartitionedRayExecutor,
    _StreamTeeSink,
)
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.mapper.whitespace_normalization_mapper import (
    WhitespaceNormalizationMapper,
)
from data_juicer.utils.ckpt_utils import (
    STREAM_MANIFEST_SCHEMA_VERSION,
    CheckpointStrategy,
    RayCheckpointManager,
)


def _executor(tmp_path):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = Namespace(auto_op_parallelism=False)
    executor.ckpt_manager = RayCheckpointManager(str(tmp_path))
    executor.job_id = "review"
    executor.recovery_mode = "streaming"
    executor.stream_segments = 2
    executor.stream_fsync = False
    executor.max_concurrent_partitions = 1
    executor._is_resuming = False
    return executor


def _source(executor):
    data = ray.data.from_items([{"text": "a  b", "uid": i} for i in range(10)], override_num_blocks=1)
    return RayDataset(data, cfg=executor.cfg)


def test_streaming_selection_honors_checkpoint_switch_and_accepts_cardinality(tmp_path):
    executor = _executor(tmp_path)
    mapper = WhitespaceNormalizationMapper(num_proc=1)
    dataset = SimpleNamespace(data=SimpleNamespace())
    executor.ckpt_manager.checkpoint_enabled = False
    assert not executor._should_use_streaming_recovery(dataset, [mapper])

    executor._streaming_selected_cache = None
    executor.ckpt_manager.checkpoint_enabled = True
    executor.ckpt_manager.checkpoint_strategy = CheckpointStrategy.EVERY_OP
    assert executor._should_use_streaming_recovery(dataset, [TextLengthFilter(min_len=1, num_proc=1)])

    executor._streaming_selected_cache = None
    mapper._name = "nlpaug_en_mapper"
    assert executor._should_use_streaming_recovery(dataset, [mapper])


def test_streaming_tee_rejects_duplicate_ids(tmp_path):
    sink = _StreamTeeSink(
        str(tmp_path), str(tmp_path / "data"), str(tmp_path / "manifest"), 0, 0, "mapper", "review", fsync=False
    )
    with pytest.raises(RuntimeError, match="duplicate row ids"):
        sink(pa.table({"__data_juicer_row_id__": [0, 0], "text": ["a", "b"]}))
    assert not list((tmp_path / "manifest").glob("*.json"))


def test_streaming_reconcile_rejects_overlapping_blocks(tmp_path):
    manager = RayCheckpointManager(str(tmp_path))
    data_dir = tmp_path / "stream_data" / "segment_0000"
    manifest_dir = tmp_path / "stream_manifest" / "segment_0000"
    data_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    for index, row_range in enumerate(([0, 3], [2, 5])):
        block = data_dir / f"block_{index}.parquet"
        block.touch()
        shard = {
            "schema_version": STREAM_MANIFEST_SCHEMA_VERSION,
            "block_uri": str(block.relative_to(tmp_path)),
            "committed_row_id_ranges": [row_range],
        }
        (manifest_dir / f"block_{index}.json").write_text(json.dumps(shard))

    with pytest.raises(RuntimeError, match="overlapping committed row ids"):
        manager.reconcile_stream_frontier(0)


def test_streaming_plan_build_does_not_commit_probe_output(tmp_path):
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    ray.data.DataContext.get_current().execution_options.preserve_order = True
    executor = _executor(tmp_path)
    ops = [WhitespaceNormalizationMapper(num_proc=1), WhitespaceNormalizationMapper(num_proc=1)]
    segments = executor._split_ops_into_segments(ops)
    stamped = executor._stamp_row_ids(_source(executor))
    stream = executor._thread_segments(stamped.data, segments, 0, 1)

    assert executor.ckpt_manager.reconcile_stream_frontier(0)[2] == 0
    assert executor.ckpt_manager.reconcile_stream_frontier(1)[2] == 0

    assert sorted(row["uid"] for row in stream.materialize().take_all()) == list(range(10))
    assert executor.ckpt_manager.reconcile_stream_frontier(0)[2] == 1
    assert executor.ckpt_manager.reconcile_stream_frontier(1)[2] == 1
