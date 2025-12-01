from ..base_op import OPERATORS, Grouper, convert_list_dict_to_dict_list


@OPERATORS.register_module("naive_grouper")
class NaiveGrouper(Grouper):
    """Group all samples in a dataset into a single batched sample.

    This operator takes a dataset and combines all its samples into one batched sample. If
    the input dataset is empty, it returns an empty dataset. The resulting batched sample is
    a dictionary where each key corresponds to a list of values from all samples in the
    dataset."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def process(self, dataset):
        if len(dataset) == 0:
            return dataset

        batched_sample = convert_list_dict_to_dict_list(dataset)

        return [batched_sample]
