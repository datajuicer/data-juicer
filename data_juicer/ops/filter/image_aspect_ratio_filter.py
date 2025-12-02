import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES


@OPERATORS.register_module("image_aspect_ratio_filter")
@LOADED_IMAGES.register_module("image_aspect_ratio_filter")
class ImageAspectRatioFilter(Filter):
    """Filter to keep samples with image aspect ratio within a specific range.

    The operator computes the aspect ratio for each image in the sample, defined as the
    width divided by the height (W / H). It caches the computed aspect ratios in the
    'aspect_ratios' field. Samples are kept if their images' aspect ratios fall within the
    specified minimum and maximum range. The 'any_or_all' parameter determines the strategy:
    'any' keeps samples if at least one image meets the criteria, while 'all' requires all
    images to meet the criteria. If no images are present in a sample, the sample is not
    filtered out."""

    _batched_op = True

    def __init__(self, min_ratio: float = 0.333, max_ratio: float = 3.0, any_or_all: str = "any", *args, **kwargs):
        """
        Initialization method.

        :param min_ratio: The min aspect ratio to keep samples.
        :param max_ratio: The max aspect ratio to keep samples.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_batched(self, samples, context=False):
        image_list = samples[self.image_key]
        samples_stats = samples[Fields.stats]

        for i, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.aspect_ratios in stat:
                continue

            # there is no image in this sample
            loaded_image_keys = image_list[i]
            if not loaded_image_keys:
                stat[StatsKeys.aspect_ratios] = np.array([], dtype=np.float64)
                continue

            # load images
            samples, images = load_data_with_context(
                samples, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key, sample_idx=i
            )

            # compute aspect ratios for each image with W/H
            aspect_ratios = {key: (images[key].width / images[key].height) for key in images}
            stat[StatsKeys.aspect_ratios] = [aspect_ratios[key] for key in loaded_image_keys]

        return samples

    def process_batched(self, samples):
        def process_single(values):
            keep_bools = np.array([self.get_keep_boolean(value, self.min_ratio, self.max_ratio) for value in values])
            if len(keep_bools) <= 0:
                return True

            # different strategies
            if self.any:
                return keep_bools.any()
            else:
                return keep_bools.all()

        return map(
            lambda stat: process_single(stat[StatsKeys.aspect_ratios]),
            samples[Fields.stats],
        )
