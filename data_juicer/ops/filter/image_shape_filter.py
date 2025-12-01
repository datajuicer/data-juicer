import sys

import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES


@OPERATORS.register_module("image_shape_filter")
@LOADED_IMAGES.register_module("image_shape_filter")
class ImageShapeFilter(Filter):
    """Filter to keep samples with image shape (width, height) within specific ranges.

    This operator filters samples based on the width and height of images. It keeps samples
    where the image dimensions fall within the specified ranges. The operator supports two
    strategies: 'any' and 'all'. In 'any' mode, a sample is kept if at least one image meets
    the criteria. In 'all' mode, all images in the sample must meet the criteria for the
    sample to be kept. The image width and height are stored in the 'image_width' and
    'image_height' fields of the sample's stats. If no images are present in the sample, the
    corresponding stats fields will be empty arrays."""

    _batched_op = True

    def __init__(
        self,
        min_width: int = 1,
        max_width: int = sys.maxsize,
        min_height: int = 1,
        max_height: int = sys.maxsize,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_width: The min width to keep samples.
        :param max_width: The max width to keep samples.
        :param min_height: The min height to keep samples.
        :param max_height: The max height to keep samples.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.image_width in sample[Fields.stats] and StatsKeys.image_height in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_width] = np.array([], dtype=np.int64)
            sample[Fields.stats][StatsKeys.image_height] = np.array([], dtype=np.int64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # get width and height for each image
        whs = {key: (images[key].width, images[key].height) for key in images}
        sample[Fields.stats][StatsKeys.image_width] = [whs[key][0] for key in loaded_image_keys]
        sample[Fields.stats][StatsKeys.image_height] = [whs[key][1] for key in loaded_image_keys]
        return sample

    def process_single(self, sample):
        ws = sample[Fields.stats][StatsKeys.image_width]
        hs = sample[Fields.stats][StatsKeys.image_height]
        if len(ws) <= 0:
            return True
        keep_bools = np.array(
            [
                self.get_keep_boolean(w, self.min_width, self.max_width)
                and self.get_keep_boolean(h, self.min_height, self.max_height)
                for w, h in zip(ws, hs)
            ]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
