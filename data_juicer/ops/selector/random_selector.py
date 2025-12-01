from typing import Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.ops.base_op import OPERATORS, Selector
from data_juicer.utils.sample import random_sample


@OPERATORS.register_module("random_selector")
class RandomSelector(Selector):
    """Randomly selects a subset of samples from the dataset.

    This operator randomly selects a subset of samples based on either a specified ratio or
    a fixed number. If both `select_ratio` and `select_num` are provided, the one that
    results in fewer samples is used. The selection is skipped if the dataset has only one
    or no samples. The `random_sample` function is used to perform the actual sampling.

    - `select_ratio`: The ratio of samples to select (0 to 1).
    - `select_num`: The exact number of samples to select.
    - If neither `select_ratio` nor `select_num` is set, the dataset remains unchanged."""

    def __init__(
        self,
        select_ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
        select_num: Optional[PositiveInt] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param select_ratio: The ratio to select. When both
            select_ratio and select_num are set, the value corresponding
            to the smaller number of samples will be applied.
        :param select_num: The number of samples to select. When both
            select_ratio and select_num are set, the value corresponding
            to the smaller number of samples will be applied.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.select_ratio = select_ratio
        self.select_num = select_num

    def process(self, dataset):
        if len(dataset) <= 1:
            return dataset

        if self.select_ratio is None and self.select_num is None:
            return dataset

        if not self.select_ratio:
            select_num = self.select_num
        else:
            select_num = int(self.select_ratio * len(dataset))
            if self.select_num and self.select_num < select_num:
                select_num = self.select_num

        return random_sample(dataset, sample_number=select_num)
