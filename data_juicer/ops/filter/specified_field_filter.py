from data_juicer.utils.constant import Fields

from ..base_op import OPERATORS, Filter

OP_NAME = "specified_field_filter"


@OPERATORS.register_module(OP_NAME)
class SpecifiedFieldFilter(Filter):
    """Filter samples based on the specified field information.

    This operator checks if the value of a specified field in each sample is within a given
    target value range. If the field value is not within the target range, the sample is
    filtered out. The field can be a multi-level key, with levels separated by dots. The
    target value is a list of acceptable values for the field. If the field value is not a
    list or tuple, it is converted to a list for comparison. Samples are retained if all
    values in the field match any of the target values.

    - Uses the 'field_key' and 'target_value' parameters.
    - Supports multi-level field keys, e.g., 'level1.level2'.
    - Converts non-list/tuple field values to a list for comparison."""

    def __init__(self, field_key: str = "", target_value: list = [], *args, **kwargs):
        """
        Initialization method.

        :param field_key: Filter based on the specified value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param target_value: The range of specified field information
            corresponding to the samples that need to be retained.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_key = field_key
        self.target_value = target_value

    def compute_stats_single(self, sample):
        # get the value from the original field
        field_value = sample
        for key in self.field_key.split("."):
            assert key in field_value.keys(), "'{}' not in {}".format(key, field_value.keys())
            field_value = field_value[key]
        # copy it into the stats field
        if self.field_key not in sample[Fields.stats]:
            sample[Fields.stats][self.field_key] = field_value
        return sample

    def process_single(self, sample):
        if not (self.field_key and self.target_value):
            return True

        field_value = sample[Fields.stats][self.field_key]

        if not (isinstance(field_value, list) or isinstance(field_value, tuple)):
            field_value = [field_value]
        res_bool = True
        for value in field_value:
            if value not in self.target_value:
                res_bool = False
        if self.reversed_range:
            res_bool = not res_bool
        return res_bool
