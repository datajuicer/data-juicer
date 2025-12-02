import numbers
from typing import Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from ..base_op import OPERATORS, Selector


@OPERATORS.register_module("frequency_specified_field_selector")
class FrequencySpecifiedFieldSelector(Selector):
    """Selector to filter samples based on the frequency of a specified field.

    This operator selects samples based on the frequency of values in a specified field. The
    field can be multi-level, with keys separated by dots. It supports filtering by either a
    top ratio or a fixed number (topk) of the most frequent values. If both top_ratio and
    topk are provided, the one resulting in fewer samples is used. The sorting order can be
    controlled with the reverse parameter. The operator processes the dataset and returns a
    new dataset containing only the selected samples."""

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
        :param top_ratio: Ratio of selected top specified field value,
            samples will be selected if their specified field values are
            within this parameter. When both topk and top_ratio are set,
            the value corresponding to the smaller number of samples
            will be applied.
        :param topk: Number of selected top specified field value,
            samples will be selected if their specified field values are
            within this parameter. When both topk and top_ratio are set,
            the value corresponding to the smaller number of samples
            will be applied.
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

        field_keys = self.field_key.split(".")
        assert field_keys[0] in dataset.features.keys(), "'{}' not in {}".format(field_keys[0], dataset.features.keys())

        field_value_dict = {}
        for i, item in enumerate(dataset[field_keys[0]]):
            field_value = item
            for key in field_keys[1:]:
                assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
                field_value = field_value[key]
            assert (
                field_value is None or isinstance(field_value, str) or isinstance(field_value, numbers.Number)
            ), "The {} item is not String, Numbers or NoneType".format(i)
            if field_value not in field_value_dict.keys():
                field_value_dict[field_value] = [i]
            else:
                field_value_dict[field_value].append(i)

        select_num = 0
        if not self.top_ratio:
            if not self.topk:
                return dataset
            else:
                select_num = self.topk
        else:
            select_num = self.top_ratio * len(field_value_dict)
            if self.topk and self.topk < select_num:
                select_num = self.topk

        select_index = sum(
            sorted(field_value_dict.values(), key=lambda x: len(x), reverse=self.reverse)[: int(select_num)], []
        )
        return dataset.select(select_index)
