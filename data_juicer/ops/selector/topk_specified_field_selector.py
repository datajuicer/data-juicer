import heapq
from typing import Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.common_utils import iter_column, stats_to_number

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module("topk_specified_field_selector")
class TopkSpecifiedFieldSelector(Selector):
    """Selects top samples based on the sorted values of a specified field.

    This operator selects the top samples from a dataset based on the values of a specified
    field. The field can be multi-level, with keys separated by dots. The selection is based
    on either a specified ratio of the dataset or a fixed number of top samples. If both
    `top_ratio` and `topk` are provided, the one resulting in fewer samples is used. The
    sorting order can be ascending or descending, controlled by the `reverse` parameter. The
    key metric is the value of the specified field, and the operator uses this to determine
    which samples to keep."""

    def __init__(
        self,
        field_key: str = "",
        top_ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
        topk: Optional[PositiveInt] = None,
        reverse: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param field_key: Selector based on the specified value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param top_ratio: Ratio of selected top samples, samples will be
            selected if their specified field values are within this
            parameter. When both topk and top_ratio are set, the value
            corresponding to the smaller number of samples will be
            applied.
        :param topk: Number of selected top sample, samples will be
            selected if their specified field values are within this
            parameter. When both topk and top_ratio are set, the value
            corresponding to the smaller number of samples will be
            applied.
        :param reverse: Determine the sorting rule, if reverse=True,
            then sort in descending order.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_key = field_key
        self.top_ratio = top_ratio
        self.topk = topk
        self.reverse = reverse

    def process(self, dataset):
        if len(dataset) <= 1 or not self.field_key:
            return dataset

        select_num = 0
        if self.top_ratio is None:
            if self.topk is None:
                return dataset
            else:
                select_num = self.topk
        else:
            select_num = self.top_ratio * len(dataset)
            if self.topk and self.topk < select_num:
                select_num = self.topk

        field_keys = self.field_key.split(".")
        assert field_keys[0] in dataset.features.keys(), "'{}' not in {}".format(field_keys[0], dataset.features.keys())

        select_num_int = int(select_num)
        if select_num_int <= 0:
            return dataset.select([])

        # Stream the field column and maintain a heap of size select_num
        # instead of materializing the entire field_value_list.
        # This reduces peak memory from O(n) to O(select_num).
        heap = []  # min-heap

        for idx, item in enumerate(iter_column(dataset, field_keys[0])):
            field_value = item
            for key in field_keys[1:]:
                assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
                field_value = field_value[key]
            # keep native numbers as-is so large integers are compared exactly
            if not isinstance(field_value, (int, float)):
                field_value = stats_to_number(field_value, self.reverse)

            # For reverse=True (descending, want largest values):
            #   min-heap stores the smallest of the top-k;
            #   replace when current value is larger.
            # For reverse=False (ascending, want smallest values):
            #   negate the value so min-heap behaves as max-heap;
            #   replace when current value is smaller than the current k-th smallest.
            heap_value = field_value if self.reverse else -field_value
            # Use -idx as second tuple element so that, on ties, the later row
            # sits on top of the heap and gets evicted first — preserving the
            # original stable selection (earlier rows are kept).
            heap_entry = (heap_value, -idx)

            if len(heap) < select_num_int:
                heapq.heappush(heap, heap_entry)
            elif heap_value > heap[0][0]:
                heapq.heapreplace(heap, heap_entry)

        # Sort by heap_value descending to match nlargest output order.
        # reverse=True: heap_value=field_value, descending → largest first.
        # reverse=False: heap_value=-field_value, descending → smallest field first.
        heap.sort(reverse=True)
        select_index = [-neg_idx for (val, neg_idx) in heap]
        return dataset.select(select_index)
