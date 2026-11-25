import heapq

import pytest
from datasets import Dataset

from data_juicer.ops.selector.range_specified_field_selector import (
    RangeSpecifiedFieldSelector,
)
from data_juicer.utils.common_utils import stats_to_number


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        [3, 1, 1, 2, 4, 2, 1, 0],
        [None, "2", "invalid", "1", "2", None, "-1", "0"],
        [[3, 1], [], None, [2, 2], [1], [4, 6], [2], [0]],
        [float("inf"), 2, float("-inf"), 2, -0.0, 0.0, 4, 1],
    ],
)
@pytest.mark.parametrize("bounds", [(0, 4), (2, 7), (5, 20), (3, 3)])
def test_range_selection_matches_stable_reference(values, bounds, nested, indexed):
    rows = [{"id": i, "score": {"quality": value} if nested else value} for i, value in enumerate(values)]
    dataset = Dataset.from_list(rows)
    if indexed:
        dataset = dataset.select([6, 0, 7, 2, 4, 1, 3, 5])
    field_key = "score.quality" if nested else "score"
    scores = [stats_to_number(row["score"]["quality"] if nested else row["score"]) for row in dataset]
    lower, upper = bounds
    upper = max(lower, min(upper, len(dataset)))
    # Preserve the operator's two-stage order: ascending shortlist, then a
    # stable descending selection. In particular, do not reverse tied rows.
    shortlist = heapq.nsmallest(upper, range(len(dataset)), key=scores.__getitem__)
    expected = heapq.nlargest(upper - lower, shortlist, key=scores.__getitem__)
    op = RangeSpecifiedFieldSelector(field_key=field_key, lower_rank=lower, upper_rank=upper)
    assert list(op.process(dataset)["id"]) == list(dataset.select(expected)["id"])


def test_descending_selection_keeps_earlier_tied_rows():
    dataset = Dataset.from_dict({"id": list(range(6)), "score": [1, 2, 2, 2, 3, 3]})
    op = RangeSpecifiedFieldSelector(field_key="score", lower_rank=2, upper_rank=5)
    assert list(op.process(dataset)["id"]) == [4, 1, 2]
