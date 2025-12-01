import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import get_file_size, size_to_bytes

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module("image_size_filter")
class ImageSizeFilter(Filter):
    """Keep data samples whose image size (in Bytes/KB/MB/...) is within a specific range.

    This operator filters data samples based on the size of their images. It keeps samples
    if the image sizes fall within the specified minimum and maximum size range. The operator
    supports two strategies: 'any'(keep the sample if any image meets the size condition) and
    'all' (keep the sample only if all images meet the size condition). If no images are
    present in the sample, the 'image_sizes' field will be an empty array."""

    _batched_op = True

    def __init__(self, min_size: str = "0", max_size: str = "1TB", any_or_all: str = "any", *args, **kwargs):
        """
        Initialization method.

        :param min_size: The min image size to keep samples.  set to be "0" by
            default for no size constraint
        :param max_size: The max image size to keep samples.  set to be
            "1TB" by default, an approximate for un-limited case
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_size = size_to_bytes(min_size)
        self.max_size = size_to_bytes(max_size)
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.image_sizes in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_sizes] = np.array([], dtype=np.float64)
            return sample

        # for size calculation, no need to load images into memory
        if self.image_bytes_key in sample and len(sample[self.image_bytes_key]) == len(sample[self.image_key]):
            sample[Fields.stats][StatsKeys.image_sizes] = [len(img_bytes) for img_bytes in sample[self.image_bytes_key]]
        else:
            sample[Fields.stats][StatsKeys.image_sizes] = [
                get_file_size(img_path) for img_path in sample[self.image_key]
            ]

        return sample

    def process_single(self, sample):
        image_sizes = sample[Fields.stats][StatsKeys.image_sizes]
        keep_bools = np.array(
            [self.get_keep_boolean(image_size, self.min_size, self.max_size) for image_size in image_sizes]
        )
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
