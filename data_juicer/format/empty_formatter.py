from typing import List

import pandas as pd
from datasets import Dataset, Features, Value

from data_juicer.utils.lazy_loader import LazyLoader

from .formatter import FORMATTERS, BaseFormatter

ray = LazyLoader("ray")


@FORMATTERS.register_module()
class EmptyFormatter(BaseFormatter):
    """
    The class is used to create empty data.
    """

    SUFFIXES = []

    def __init__(self, length, feature_keys: List[str] = [], *args, **kwargs):
        """
        Initialization method.

        :param length: The empty dataset length.
        :param feature_keys: feature key name list.
        """
        self.length = length
        self.feature_keys = feature_keys
        if isinstance(self.feature_keys, str):
            self.feature_keys = [self.feature_keys]

    @property
    def null_value(self):
        return None

    def load_dataset(self, *args, **kwargs):
        data_dict = {}
        features = Features()

        for key in self.feature_keys:
            features.update({key: Value("string")})
            data_dict.update({key: [self.null_value for _ in range(self.length)]})

        empty_dataset = Dataset.from_dict(data_dict, features=features)

        from data_juicer.core.data import NestedDataset

        empty_dataset = NestedDataset(empty_dataset)

        return empty_dataset


@FORMATTERS.register_module()
class RayEmptyFormatter(BaseFormatter):
    """
    The class is used to create empty data for ray.
    """

    SUFFIXES = []

    def __init__(self, length, feature_keys: List[str] = [], *args, **kwargs):
        """
        Initialization method.

        :param length: The empty dataset length.
        :param feature_keys: feature key name list.
        """
        self.length = length
        self.feature_keys = feature_keys
        if isinstance(self.feature_keys, str):
            self.feature_keys = [self.feature_keys]

    @property
    def null_value(self):
        return {}

    def load_dataset(self, *args, **kwargs):
        if len(self.feature_keys):
            df = pd.DataFrame({col: [self.null_value for _ in range(self.length)] for col in self.feature_keys})
        else:
            df = pd.DataFrame([self.null_value for _ in range(self.length)])

        empty_dataset = ray.data.from_pandas(df)

        return empty_dataset
