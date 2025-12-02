import heapq
from typing import Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.common_utils import stats_to_number

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module("range_specified_field_selector")
class RangeSpecifiedFieldSelector(Selector):
    """Selects a range of samples based on the sorted values of a specified field.

    This operator selects samples whose values for a specified field fall within a given
    range. The range can be defined using percentiles or ranks, and the operator will use
    the more inclusive bounds if both are provided. The field values are first sorted in
    ascending order, and then the samples are selected based on the lower and upper bounds.
    If no bounds are provided, the original dataset is returned. The operator ensures that
    the specified field exists in the dataset and handles multi-level fields by separating
    keys with dots."""

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

        if self.lower_percentile is None and self.lower_rank is None:
            return dataset
        if self.upper_percentile is None and self.upper_rank is None:
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

        field_keys = self.field_key.split(".")
        assert field_keys[0] in dataset.features.keys(), "'{}' not in {}".format(field_keys[0], dataset.features.keys())

        def get_field_value_list(cur_dataset, field_keys):
            if len(field_keys) == 1:
                field_value_list = cur_dataset[field_keys[0]]
            else:
                field_value_list = []
                for item in cur_dataset[field_keys[0]]:
                    field_value = item
                    for key in field_keys[1:]:
                        assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
                        field_value = field_value[key]
                    field_value_list.append(field_value)
            field_value_list = [stats_to_number(s) for s in field_value_list]
            return field_value_list

        field_value_list = get_field_value_list(dataset, field_keys)
        select_index = heapq.nsmallest(int(upper_bound), range(len(dataset)), field_value_list.__getitem__)
        sub_dataset = dataset.select(select_index)

        field_value_list = get_field_value_list(sub_dataset, field_keys)
        select_index = heapq.nlargest(
            int(upper_bound - lower_bound), range(len(sub_dataset)), field_value_list.__getitem__
        )

        return sub_dataset.select(select_index)
