import heapq
from typing import Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.common_utils import iter_column, stats_to_number

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module("range_specified_field_selector")
class RangeSpecifiedFieldSelector(Selector):
    """Selects a range of samples based on the sorted values of a specified field.

    This operator ranks samples by the values of a specified field in ascending order
    and selects samples within a given rank or percentile range.
    Either bound may be omitted: an omitted lower bound defaults to 0, and an omitted
    upper bound defaults to the dataset size. When both percentile and rank constraints
    are provided, the larger lower bound and the smaller upper bound are used.
    If no bounds are provided, the original dataset and its order are preserved.
    Datasets with zero or one sample are returned unchanged. The operator ensures that
    the specified field exists in the dataset and supports nested fields with keys
    separated by dots."""

    def __init__(
        self,
        field_key: str = "",
        lower_percentile: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
        upper_percentile: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
        lower_rank: Optional[PositiveInt] = None,
        upper_rank: Optional[PositiveInt] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param field_key: Selector based on the specified value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param lower_percentile: The lower bound of the percentile to
            be sample, samples will be selected if their specified field
            values are greater than this lower bound. When both
            lower_percentile and lower_rank are set, the value corresponding
            to the larger number of samples will be applied.
        :param upper_percentile: The upper bound of the percentile to
            be sample, samples will be selected if their specified field
            values are less or equal to the upper bound. When both
            upper_percentile and upper_rank are set, the value corresponding
            to the smaller number of samples will be applied.
        :param lower_rank: The lower bound of the rank to be sample,
            samples will be selected if their specified field values are
            greater than this lower bound. When both lower_percentile and
            lower_rank are set, the value corresponding to the larger number
            of samples will be applied.
        :param upper_rank: The upper bound of the rank to be sample,
            samples will be selected if their specified field values are
            less or equal to the upper bound. When both upper_percentile and
            upper_rank are set, the value corresponding to the smaller number
            of samples will be applied.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_key = field_key
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_rank = lower_rank
        self.upper_rank = upper_rank

    def process(self, dataset):
        if len(dataset) <= 1 or not self.field_key:
            return dataset

        if (
            self.lower_percentile is None
            and self.lower_rank is None
            and self.upper_percentile is None
            and self.upper_rank is None
        ):
            return dataset

        lower_bound, upper_bound = 0, len(dataset)
        if self.lower_percentile is not None:
            lower_bound = int(self.lower_percentile * len(dataset))
        if self.lower_rank is not None:
            lower_bound = max(lower_bound, self.lower_rank)
        if self.upper_percentile is not None:
            upper_bound = int(self.upper_percentile * len(dataset))
        if self.upper_rank is not None:
            upper_bound = min(upper_bound, self.upper_rank)
        upper_bound = max(lower_bound, upper_bound)
        if upper_bound <= 0:
            return dataset.select([])

        field_keys = self.field_key.split(".")
        assert field_keys[0] in dataset.features.keys(), "'{}' not in {}".format(field_keys[0], dataset.features.keys())

        # Stream the column and maintain a max-heap (using negated values) of size upper_bound
        # to keep the upper_bound smallest values. This avoids materializing the entire
        # field_value_list (O(n) memory) — peak memory drops to O(upper_bound).
        heap = []  # min-heap of (-value, -idx), simulating a max-heap

        for idx, item in enumerate(iter_column(dataset, field_keys[0])):
            field_value = item
            for key in field_keys[1:]:
                assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
                field_value = field_value[key]
            field_value = stats_to_number(field_value)

            neg_value = -field_value
            # Use -idx as second tuple element so that, on ties, the later row
            # sits on top of the heap and gets evicted first — preserving the
            # original stable selection (earlier rows are kept).
            heap_entry = (neg_value, -idx)

            if len(heap) < upper_bound:
                heapq.heappush(heap, heap_entry)
            elif neg_value > heap[0][0]:
                # heap[0] is the smallest neg_value = largest value in current top.
                # Replace when current value is smaller than the current upper_bound-th smallest.
                heapq.heapreplace(heap, heap_entry)

        # Sort heap entries by (value, idx) ascending to match nsmallest's stable order.
        # The heap itself is not sorted — we must restore ascending value order and
        # original row order for ties before the second-stage nlargest.
        heap_entries = [(-neg_val, -neg_idx) for (neg_val, neg_idx) in heap]
        heap_entries.sort(key=lambda x: (x[0], x[1]))
        select_index = [idx for (val, idx) in heap_entries]
        sub_dataset = dataset.select(select_index)

        # Second pass on sub_dataset (size upper_bound) — already small enough to materialize.
        def get_field_value_list(cur_dataset, field_keys):
            if len(field_keys) == 1:
                field_value_list = list(iter_column(cur_dataset, field_keys[0]))
            else:
                field_value_list = []
                for item in iter_column(cur_dataset, field_keys[0]):
                    field_value = item
                    for key in field_keys[1:]:
                        assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
                        field_value = field_value[key]
                    field_value_list.append(field_value)
            field_value_list = [stats_to_number(s) for s in field_value_list]
            return field_value_list

        field_value_list = get_field_value_list(sub_dataset, field_keys)
        select_index = heapq.nlargest(
            int(upper_bound - lower_bound), range(len(sub_dataset)), field_value_list.__getitem__
        )

        return sub_dataset.select(select_index)
