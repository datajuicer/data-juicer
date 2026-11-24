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


def _fake_ray_data(total_rows, split_result=None):
    """Return a ``dataset.data`` double plus the materialized view it exposes."""
    materialized = Mock()
    materialized.count.return_value = total_rows
    materialized.split_at_indices.return_value = split_result if split_result is not None else []
    data = Mock()
    data.materialize.return_value = materialized
    return data, materialized


@pytest.mark.parametrize(
    ("total_rows", "num_partitions", "expected"),
    [
        (10, 4, [3, 6, 8]),
        (8, 4, [2, 4, 6]),
        (10, 3, [4, 7]),
        (2, 2, [1]),
    ],
)
def test_balanced_split_indices_spread_the_remainder_over_leading_partitions(total_rows, num_partitions, expected):
    indices = PartitionedRayExecutor._balanced_split_indices(total_rows, num_partitions)

    assert indices == expected
    # No row is dropped and no partition is left empty.
    sizes = [b - a for a, b in zip([0] + indices, indices + [total_rows])]
    assert sum(sizes) == total_rows
    assert min(sizes) >= 1
    assert max(sizes) - min(sizes) <= 1


def test_row_balanced_split_cuts_at_row_boundaries_instead_of_block_boundaries():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 4
    data, materialized = _fake_ray_data(10, ["p0", "p1", "p2", "p3"])

    partitions = executor._split_into_row_balanced_partitions(SimpleNamespace(data=data))

    assert partitions == ["p0", "p1", "p2", "p3"]
    materialized.split_at_indices.assert_called_once_with([3, 6, 8])
    # Ray's block-based split() must not be used for the fresh split.
    data.split.assert_not_called()
    assert executor.num_partitions == 4


def test_row_balanced_split_rejects_more_partitions_than_rows():
    """An empty partition still costs an actor lifecycle and a checkpoint."""
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 8
    data, materialized = _fake_ray_data(3, ["p0", "p1", "p2"])

    with pytest.raises(ValueError, match="exceeds the 3 row"):
        executor._split_into_row_balanced_partitions(SimpleNamespace(data=data))

    materialized.split_at_indices.assert_not_called()
    # The requested count is reported as-is instead of being silently lowered.
    assert executor.num_partitions == 8


def test_row_balanced_split_materializes_a_single_partition():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 1
    data, materialized = _fake_ray_data(5)

    partitions = executor._split_into_row_balanced_partitions(SimpleNamespace(data=data))

    assert partitions == [materialized]
    materialized.split_at_indices.assert_not_called()


def test_row_balanced_split_of_an_empty_dataset_yields_one_partition():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 4
    data, materialized = _fake_ray_data(0)

    partitions = executor._split_into_row_balanced_partitions(SimpleNamespace(data=data))

    assert partitions == [materialized]
    assert executor.num_partitions == 1
    materialized.split_at_indices.assert_not_called()


@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_row_balanced_split_never_produces_empty_partitions_whatever_the_block_layout(num_blocks):
    """Regression: ``Dataset.split(n)`` splits by blocks, not by rows.

    With Ray's own ``split(4)``, ten rows in one block produce ``[10, 0, 0, 0]``
    and in two blocks ``[5, 5, 0, 0]``, so the row-count ceiling alone does not
    prevent empty or badly skewed logical partitions.
    """
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR")

    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 4
    dataset = SimpleNamespace(data=ray.data.range(10, override_num_blocks=num_blocks))

    partitions = executor._split_into_row_balanced_partitions(dataset)
    row_counts = [partition.count() for partition in partitions]

    assert row_counts == [3, 3, 2, 2]
    assert sum(row_counts) == 10


def test_row_balanced_split_keeps_every_row_exactly_once():
    ray = pytest.importorskip("ray")
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR")

    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.num_partitions = 3
    dataset = SimpleNamespace(data=ray.data.range(10, override_num_blocks=1))

    partitions = executor._split_into_row_balanced_partitions(dataset)
    ids = [row["id"] for partition in partitions for row in partition.take_all()]

    # split(n, equal=True) would have dropped rows here; split_at_indices must not.
    assert ids == list(range(10))
