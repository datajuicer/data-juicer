from typing import List, Union

import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    close_video,
    extract_video_frames_uniformly,
    load_data_with_context,
    load_video,
)
from data_juicer.utils.resource_utils import cuda_device_count

from ..base_op import OPERATORS, UNFORKABLE, Filter
from ..op_fusion import INTER_SAMPLED_FRAMES, LOADED_VIDEOS

easyocr = LazyLoader("easyocr")

OP_NAME = "video_ocr_area_ratio_filter"


def triangle_area(p1, p2, p3):
    """
    Compute the triangle area according to its coordinates.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    tri_area = 0.5 * np.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)
    return tri_area


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoOcrAreaRatioFilter(Filter):
    """Keep data samples whose detected text area ratios for specified frames in the video are
    within a specified range.

    This operator filters data based on the ratio of the detected text area to the total
    frame area. It uses EasyOCR to detect text in the specified languages and calculates the
    area ratio for each sampled frame. The operator then determines whether to keep a sample
    based on the `any` or `all` strategy, which checks if any or all of the videos meet the
    specified area ratio range. The key metric, `video_ocr_area_ratio`, is computed as the
    mean of the text area ratios across the sampled frames. The number of sampled frames and
    the specific frames to be sampled can be configured."""

    _accelerator = "cuda"

    def __init__(
        self,
        min_area_ratio: float = 0,
        max_area_ratio: float = 1.0,
        frame_sample_num: PositiveInt = 3,
        languages_to_detect: Union[str, List[str]] = ["ch_sim", "en"],
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_area_ratio: The min ocr area ratio to keep samples. It's 0
            by default.
        :param max_area_ratio: The max ocr area ratio to keep samples. It's 1.0
            by default.
        :param frame_sample_num: The number of sampled frames to calculate the
            ocr area ratio. If it's 1, only middle frame will be selected. If
            it's 2, only the first and the last frames will be selected. If
            it's larger than 2, in addition to the first and the last frames,
            other frames will be sampled evenly within the video duration.
        :param languages_to_detect: texts in which languages should be
            detected. Default: ['ch_sim', 'en']. Full language list can be
            found here: https://www.jaided.ai/easyocr/.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.frame_sample_num = frame_sample_num
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        # initialize easyocr reader
        if isinstance(languages_to_detect, str):
            languages_to_detect = [languages_to_detect]
        self.reader = easyocr.Reader(
            lang_list=languages_to_detect,
            recognizer=False,
            verbose=False,
            gpu=False,
        )

        # only uniformly sampling method is supported in this OP
        self.sampled_frames_key_suffix = f"-uniform-{frame_sample_num}"

    def get_reader(self, rank):
        if self.use_cuda():
            rank = 0 if rank is None else rank
            device = f"cuda:{rank % cuda_device_count()}"
            self.reader.detector = self.reader.detector.to(device)
            self.reader.device = device
        return self.reader

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.video_ocr_area_ratio in sample[Fields.stats]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.stats][StatsKeys.video_ocr_area_ratio] = np.array([], dtype=np.float64)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        reader = self.get_reader(rank)
        # compute ocr area ratios
        video_ocr_area_ratios = {}
        for video_key, container in videos.items():
            sampled_frames_key = video_key + self.sampled_frames_key_suffix
            if context and sampled_frames_key in sample[Fields.context]:
                sampled_frames = sample[Fields.context][sampled_frames_key]
            else:
                sampled_frames = extract_video_frames_uniformly(container, self.frame_sample_num)
                # store the sampled frames in the context
                if context:
                    sample[Fields.context][sampled_frames_key] = sampled_frames
            images = [f.to_image() for f in sampled_frames]
            # collect ocr results for each image
            frame_ocr_area_ratios = []
            for idx, image in enumerate(images):
                # return horizontal detected results and free-form detected
                # results
                horizontal_list, free_list = reader.detect(np.asarray(image))
                total_area = image.width * image.height
                # rectangles
                rect_area = 0
                for xmin, xmax, ymin, ymax in horizontal_list[0]:
                    if xmax < xmin or ymax < ymin:
                        continue
                    rect_area += (xmax - xmin) * (ymax - ymin)
                # free-form
                quad_area = 0
                for points in free_list[0]:
                    triangle1 = points[:3]
                    quad_area += triangle_area(*triangle1)
                    triangle2 = points[2:] + [points[0]]
                    quad_area += triangle_area(*triangle2)
                text_area = rect_area + quad_area
                frame_ocr_area_ratios.append(text_area / total_area)

                # for debug
                # if False:
                #     from PIL import ImageDraw
                #     draw = ImageDraw.Draw(image)
                #     for xmin, xmax, ymin, ymax in horizontal_list[0]:
                #         if xmax < xmin or ymax < ymin:
                #             continue
                #         draw.rectangle((xmin, ymin, xmax, ymax),
                #                        outline='red',
                #                        width=1)
                #     for points in free_list[0]:
                #         points = [(int(item[0]), int(item[1]))
                #                   for item in points]
                #         draw.polygon(points, outline='blue', width=1)
                #     image.save(f'{video_key}-{idx}.jpg')
            video_ocr_area_ratios[video_key] = np.mean(frame_ocr_area_ratios)

            if not context:
                close_video(container)

        # get video durations
        sample[Fields.stats][StatsKeys.video_ocr_area_ratio] = [
            video_ocr_area_ratios[video_key] for video_key in sample[self.video_key]
        ]

        return sample

    def process_single(self, sample):
        video_ocr_area_ratios = sample[Fields.stats][StatsKeys.video_ocr_area_ratio]
        keep_bools = np.array(
            [
                self.get_keep_boolean(ocr_area_ratio, self.min_area_ratio, self.max_area_ratio)
                for ocr_area_ratio in video_ocr_area_ratios
            ]
        )
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
