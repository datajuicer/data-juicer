import sys

from data_juicer.utils.constant import Fields

from ..base_op import OPERATORS, Filter


def is_number(s):
    if s:
        try:
            float(s)
            return True
        except ValueError:
            pass
    return False


OP_NAME = "specified_numeric_field_filter"


@OPERATORS.register_module(OP_NAME)
class SpecifiedNumericFieldFilter(Filter):
    """Filter samples based on a specified numeric field value.

    This operator filters out samples if the numeric value in the specified field is not
    within the given range. The field can be multi-level, with keys separated by dots. The
    sample is kept if the numeric value is between the minimum and maximum values,
    inclusive. If the field key is not provided, all samples are retained. The operator
    ensures that the field exists in the sample and that its value is numeric before
    performing the comparison.

    - Uses the 'min_value' and 'max_value' to define the acceptable range.
    - Supports multi-level fields using dot-separated keys.
    - Returns False for non-numeric or out-of-range values, filtering the sample."""

    def __init__(
        self, field_key: str = "", min_value: float = -sys.maxsize, max_value: float = sys.maxsize, *args, **kwargs
    ):
        """
        Initialization method.

        :param field_key: Filter based on the specified numeric value
            corresponding to the target key. The target key
            corresponding to multi-level field information need to be
            separated by '.'.
        :param min_value: The min filter value in SpecifiedNumericField
            op, samples will be filtered if their specified numeric
            field value is below this parameter.
        :param max_value: The max filter value in SpecifiedNumericField
            op, samples will be filtered if their specified numeric
            field value exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.field_key = field_key
        self.min_value = min_value
        self.max_value = max_value

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
        if not self.field_key:
            return True

        field_value = sample[Fields.stats][self.field_key]

        if is_number(field_value):
            field_value = float(field_value)
            return self.get_keep_boolean(field_value, self.min_value, self.max_value)
        else:
            return False
