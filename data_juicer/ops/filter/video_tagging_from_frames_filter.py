from typing import List

import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, MetaKeys

from ..base_op import NON_STATS_FILTERS, OPERATORS, TAGGING_OPS, UNFORKABLE, Filter
from ..mapper.video_tagging_from_frames_mapper import VideoTaggingFromFramesMapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_tagging_from_frames_filter"


@NON_STATS_FILTERS.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoTaggingFromFramesFilter(Filter):
    """Filter to keep samples whose videos contain specified tags.

    This operator filters video samples based on the presence of given tags in the video
    frames. It uses a Hugging Face tokenizer to extract and tag frames. The filtering can be
    configured to require any or all of the specified tags to be present. The operator
    supports two frame sampling methods: "all_keyframes" and "uniform". When using
    "uniform", the number of frames to sample can be specified. The extracted tags are
    stored in the meta field with the key 'video_frame_tags' by default. The decision to
    keep a sample is based on whether any or all of the video frames meet the tag criteria,
    as specified by the 'any_or_all' parameter."""

    _accelerator = "cuda"

    def __init__(
        self,
        tags: List[str] = ["people"],
        contain: str = "any",
        frame_sampling_method: str = "all_keyframes",
        frame_num: PositiveInt = 3,
        tag_field_name: str = MetaKeys.video_frame_tags,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param tags: a tag list to shift the videos, total tags can be found
            in https://github.com/xinyu1205/recognize-anything/blob/main/ram/data/ram_tag_list.txt # noqa: E501
        :param contain: require the videos containing 'any' or 'all' tags.
            When tags equal to [], 'all' keeps all samples, 'any' keeps no
            sample.
        :param frame_sampling_method: sampling method of extracting frame
            images from the videos. Should be one of
            ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number of which depends
            on the duration of the video) and the latter one extract specified
            number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param tag_field_name: the key name to store the tags in the meta
            field. It's "video_frame_tags" in default.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["mem_required"] = "9GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        if contain not in ["any", "all"]:
            raise ValueError(
                f"the containing type [{contain}] is not " f'supported. Can only be one of ["any", "all"].'
            )
        if frame_sampling_method not in ["all_keyframes", "uniform"]:
            raise ValueError(
                f"Frame sampling method [{frame_sampling_method}] is not "
                f'supported. Can only be one of ["all_keyframes", "uniform"].'
            )
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.tags = set([tag.lower() for tag in tags])
        self.contain_any = contain == "any"
        self.any = any_or_all == "any"
        self.tag_field_name = tag_field_name
        self.tagging_producer = VideoTaggingFromFramesMapper(
            frame_sampling_method=frame_sampling_method,
            frame_num=frame_num,
            accelerator=self.accelerator,
            tag_field_name=self.tag_field_name,
        )

    def compute_stats_single(self, sample, rank=None, context=False):
        sample = self.tagging_producer.process(sample, rank, context)

        return sample

    def process_single(self, sample, rank=None):
        video_tags = sample[Fields.meta][self.tag_field_name]
        if len(video_tags) <= 0:
            return True

        keep_bools = []
        for words in video_tags:
            words = set([w.lower() for w in words])
            if self.contain_any:
                keep_bools.append(bool(self.tags & words))
            else:
                keep_bools.append(self.tags.issubset(words))
        keep_bools = np.array(keep_bools)
        if self.reversed_range:
            keep_bools = np.logical_not(keep_bools)

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
