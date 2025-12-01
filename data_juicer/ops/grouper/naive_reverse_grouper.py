import json
import os

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import create_directory_if_not_exists

from ..base_op import OPERATORS, Grouper, convert_dict_list_to_list_dict


@OPERATORS.register_module("naive_reverse_grouper")
class NaiveReverseGrouper(Grouper):
    """Split batched samples into individual samples.

    This operator processes a dataset by splitting each batched sample into
    individual samples. It also handles and optionally exports batch metadata.
    - If a sample contains 'batch_meta', it is separated and can be exported
    to a specified path.
    - The operator converts the remaining data from a dictionary of lists
    to a list of dictionaries, effectively unbatching the samples.
    - If `batch_meta_export_path` is provided, the batch metadata is written
    to this file in JSON format, one entry per line.
    - If no samples are present in the dataset, the original dataset is returned."""

    def __init__(self, batch_meta_export_path=None, *args, **kwargs):
        """
        Initialization method.

        :param batch_meta_export_path: the path to export the batch meta.
            Just drop the batch meta if it is None.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.batch_meta_export_path = batch_meta_export_path

    def process(self, dataset):
        if len(dataset) == 0:
            return dataset

        samples = []
        batch_metas = []
        for sample in dataset:
            if Fields.batch_meta in sample:
                batch_metas.append(sample[Fields.batch_meta])
                sample = {k: sample[k] for k in sample if k != Fields.batch_meta}
            samples.extend(convert_dict_list_to_list_dict(sample))
        if self.batch_meta_export_path is not None:
            create_directory_if_not_exists(os.path.dirname(self.batch_meta_export_path))
            with open(self.batch_meta_export_path, "w") as f:
                for batch_meta in batch_metas:
                    f.write(json.dumps(batch_meta, ensure_ascii=False) + "\n")

        return samples
