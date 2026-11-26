from typing import List, Optional

import pyarrow

from data_juicer.utils.common_utils import dict_to_hash, nested_access

from ..base_op import (
    DEFAULT_BATCH_SIZE,
    OPERATORS,
    Grouper,
    convert_list_dict_to_dict_list,
)
from .naive_grouper import NaiveGrouper


@OPERATORS.register_module("key_value_grouper")
class KeyValueGrouper(Grouper):
    """Groups samples into batches based on values in specified keys.

    This operator groups samples by the values of the given keys, which can be nested. If no
    keys are provided, it defaults to using the text key. It uses a naive grouping strategy
    to batch samples with identical key values. The resulting dataset is a list of batched
    samples, where each batch contains samples that share the same key values. This is
    useful for organizing data by specific attributes or features."""

    _supported_exec_modes = ("default", "ray")

    def __init__(self, group_by_keys: Optional[List[str]] = None, *args, **kwargs):
        """
        Initialization method.

        :param group_by_keys: group samples according values in the keys.
            Support for nested keys such as "__dj__stats__.text_len".
            It is [self.text_key] in default.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.group_by_keys = group_by_keys or [self.text_key]
        self.naive_grouper = NaiveGrouper()

    def _group_key(self, sample):
        return dict_to_hash({key: nested_access(sample, key) for key in self.group_by_keys})

    def process(self, dataset):
        if len(dataset) == 0:
            return dataset

        sample_map = {}
        for sample in dataset:
            sample_key = self._group_key(sample)
            if sample_key in sample_map:
                sample_map[sample_key].append(sample)
            else:
                sample_map[sample_key] = [sample]

        batched_samples = [convert_list_dict_to_dict_list(sample_map[k]) for k in sample_map]

        return batched_samples

    def run_ray(self, dataset):
        """Group across Ray blocks. Each group must fit in memory on one worker."""

        def add_group_keys(batch):
            keys = pyarrow.array([self._group_key(sample) for sample in batch.to_pylist()], type=pyarrow.string())
            return pyarrow.Table.from_arrays([keys, batch.to_struct_array()], names=["group_key", "sample"])

        def pack_group(batch):
            samples = batch["sample"]
            offsets = pyarrow.array([0, len(batch)], type=pyarrow.int32())
            return pyarrow.Table.from_arrays(
                [pyarrow.ListArray.from_arrays(offsets, col.combine_chunks()) for col in samples.flatten()],
                names=[field.name for field in samples.type],
            )

        keyed = dataset.map_batches(
            add_group_keys, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE, runtime_env=self.runtime_env
        )
        return keyed.groupby("group_key").map_groups(pack_group, batch_format="pyarrow", runtime_env=self.runtime_env)
