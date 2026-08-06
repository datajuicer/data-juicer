from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow
import pytest

from data_juicer.core.executor.ray_executor_partitioned import (
    PartitionMetadata,
    PartitionedRayExecutor,
    PartitioningInfo,
    _combine_partition_hash_partials,
    _hash_partition_batch,
)


def _metadata(partition_id, row_count, content_hash="hash"):
    return PartitionMetadata(
        partition_id=partition_id,
        row_count=row_count,
        first_row_hash="first",
        last_row_hash="",
        content_hash=content_hash,
    )


def test_partition_content_hash_is_independent_of_batch_boundaries():
    rows = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]
    one_batch = [_hash_partition_batch(pyarrow.Table.from_pylist(rows)).to_pylist()[0]]
    split_batches = [
        _hash_partition_batch(pyarrow.Table.from_pylist(rows[:1])).to_pylist()[0],
        _hash_partition_batch(pyarrow.Table.from_pylist(rows[1:])).to_pylist()[0],
    ]

    one_hash, one_count = _combine_partition_hash_partials(one_batch)
    split_hash, split_count = _combine_partition_hash_partials(split_batches)

    assert one_count == split_count == 3
    assert one_hash == split_hash


def test_partition_content_hash_detects_reordered_rows():
    original = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    reordered = [{"id": "A"}, {"id": "C"}, {"id": "B"}]

    original_hash, _ = _combine_partition_hash_partials(
        _hash_partition_batch(pyarrow.Table.from_pylist(original)).to_pylist()
    )
    reordered_hash, _ = _combine_partition_hash_partials(
        _hash_partition_batch(pyarrow.Table.from_pylist(reordered)).to_pylist()
    )

    assert original_hash != reordered_hash


def test_partition_content_hash_detects_changed_content():
    original = _hash_partition_batch(pyarrow.Table.from_pylist([{"id": 1}])).to_pylist()
    changed = _hash_partition_batch(pyarrow.Table.from_pylist([{"id": 2}])).to_pylist()

    original_hash, _ = _combine_partition_hash_partials(original)
    changed_hash, _ = _combine_partition_hash_partials(changed)

    assert original_hash != changed_hash


def test_partitioning_info_loads_legacy_metadata_without_content_hash():
    info = PartitioningInfo.from_dict(
        {
            "num_partitions": 1,
            "total_rows": 2,
            "partitions": [
                {
                    "partition_id": 0,
                    "row_count": 2,
                    "first_row_hash": "first",
                    "last_row_hash": "",
                }
            ],
        }
    )

    assert info.partitions[0].content_hash == ""


def test_partitioning_info_persists_row_offsets_and_content_hash(tmp_path):
    first = _metadata(0, 4, content_hash="partition-hash")
    first.start_row = 0
    first.end_row = 4
    info = PartitioningInfo(num_partitions=1, total_rows=4, partitions=[first])
    path = tmp_path / "partitioning_info.json"

    info.save(str(path))
    restored = PartitioningInfo.load(str(path))

    assert restored is not None
    assert restored.partitions[0].start_row == 0
    assert restored.partitions[0].end_row == 4
    assert restored.partitions[0].content_hash == "partition-hash"


def test_split_at_saved_boundaries_uses_saved_row_counts():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 3
    data = Mock()
    data.split_at_indices.return_value = ["p0", "p1", "p2"]
    dataset = SimpleNamespace(data=data)
    info = PartitioningInfo(
        num_partitions=3,
        total_rows=10,
        partitions=[_metadata(0, 2), _metadata(1, 3), _metadata(2, 5)],
    )

    partitions = executor._split_at_saved_boundaries(dataset, info)

    assert partitions == ["p0", "p1", "p2"]
    data.split_at_indices.assert_called_once_with([2, 5])


def test_split_at_saved_boundaries_uses_explicit_row_offsets():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 2
    data = Mock()
    data.split_at_indices.return_value = ["p0", "p1"]
    dataset = SimpleNamespace(data=data)
    first = _metadata(0, 4)
    first.start_row = 0
    first.end_row = 4
    second = _metadata(1, 6)
    second.start_row = 4
    second.end_row = 10
    info = PartitioningInfo(num_partitions=2, total_rows=10, partitions=[first, second])

    executor._split_at_saved_boundaries(dataset, info)

    data.split_at_indices.assert_called_once_with([4])


def test_split_at_saved_boundaries_rejects_invalid_row_offsets():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 2
    first = _metadata(0, 4)
    first.start_row = 1
    first.end_row = 5
    info = PartitioningInfo(
        num_partitions=2,
        total_rows=10,
        partitions=[first, _metadata(1, 6)],
    )

    with pytest.raises(RuntimeError, match="Saved row boundaries are invalid"):
        executor._split_at_saved_boundaries(SimpleNamespace(data=Mock()), info)


def test_split_at_saved_boundaries_materializes_valid_single_partition():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 1
    data = Mock()
    data.materialize.return_value = "p0"
    first = _metadata(0, 4)
    first.start_row = 0
    first.end_row = 4
    info = PartitioningInfo(num_partitions=1, total_rows=4, partitions=[first])

    partitions = executor._split_at_saved_boundaries(SimpleNamespace(data=data), info)

    assert partitions == ["p0"]
    data.materialize.assert_called_once_with()
    data.split_at_indices.assert_not_called()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [("start_row", 1), ("end_row", 5), ("total_rows", 5)],
)
def test_split_at_saved_boundaries_rejects_invalid_single_partition_metadata(field, invalid_value):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 1
    data = Mock()
    first = _metadata(0, 4)
    first.start_row = 0
    first.end_row = 4
    info = PartitioningInfo(num_partitions=1, total_rows=4, partitions=[first])
    target = first if field in {"start_row", "end_row"} else info
    setattr(target, field, invalid_value)

    with pytest.raises(RuntimeError, match="Saved row boundaries"):
        executor._split_at_saved_boundaries(SimpleNamespace(data=data), info)

    data.materialize.assert_not_called()


def test_explicit_resume_hash_mismatch_keeps_checkpoints():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace(_resume_requested=True)
    executor.num_partitions = 2
    executor._enable_deterministic_execution = Mock()
    executor._clear_invalid_checkpoints = Mock()
    first = _metadata(0, 1)
    first.start_row = 0
    first.end_row = 1
    second = _metadata(1, 1)
    second.start_row = 1
    second.end_row = 2
    info = PartitioningInfo(num_partitions=2, total_rows=2, partitions=[first, second])
    executor._load_partitioning_info = Mock(return_value=info)
    executor._split_at_saved_boundaries = Mock(return_value=["p0", "p1"])
    executor._validate_partitions = Mock(return_value=False)

    with pytest.raises(RuntimeError, match="Refusing to resume"):
        executor._split_dataset_deterministic(SimpleNamespace(data=Mock()))

    executor._clear_invalid_checkpoints.assert_not_called()


def test_explicit_resume_rejects_legacy_metadata_without_hashes_or_boundaries():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = SimpleNamespace(_resume_requested=True)
    executor.num_partitions = 1
    executor._enable_deterministic_execution = Mock()
    executor._clear_invalid_checkpoints = Mock()
    executor._load_partitioning_info = Mock(
        return_value=PartitioningInfo(
            num_partitions=1,
            total_rows=1,
            partitions=[_metadata(0, 1, content_hash="")],
        )
    )

    with pytest.raises(RuntimeError, match="requires content hashes and row boundaries"):
        executor._split_dataset_deterministic(SimpleNamespace(data=Mock()))

    executor._clear_invalid_checkpoints.assert_not_called()
