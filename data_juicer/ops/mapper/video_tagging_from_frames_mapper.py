from collections import Counter

import numpy as np
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import (
    close_video,
    extract_key_frames,
    extract_video_frames_uniformly,
    load_data_with_context,
    load_video,
)
from data_juicer.utils.model_utils import get_model, prepare_model, ram, torch

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_tagging_from_frames_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoTaggingFromFramesMapper(Mapper):
    """Generates video tags from frames extracted from videos.

    This operator extracts frames from videos and generates tags based on the
    content of these frames. The frame extraction method can be either
    "all_keyframes" or "uniform". For "all_keyframes", all keyframes are
    extracted, while for "uniform", a specified number of frames are
    extracted uniformly across the video. The tags are generated using a
    pre-trained model and stored in the specified field name. If the tags
    are already present in the sample, the operator skips processing.
    Important notes:
    - Uses a Hugging Face tokenizer and a pre-trained model for tag generation.
    - If no video is present in the sample, an empty tag array is stored.
    - Frame tensors are processed to generate tags, which are then sorted by
    frequency and stored."""

    _accelerator = "cuda"

    def __init__(
        self,
        frame_sampling_method: str = "all_keyframes",
        frame_num: PositiveInt = 3,
        tag_field_name: str = MetaKeys.video_frame_tags,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

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
        :param tag_field_name: the field name to store the tags. It's
            "video_frame_tags" in default.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["mem_required"] = "9GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        if frame_sampling_method not in ["all_keyframes", "uniform"]:
            raise ValueError(
                f"Frame sampling method [{frame_sampling_method}] is not "
                f'supported. Can only be one of ["all_keyframes", "uniform"].'
            )
        self.model_key = prepare_model(
            model_type="recognizeAnything", pretrained_model_name_or_path="ram_plus_swin_large_14m.pth", input_size=384
        )
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        self.transform = ram.get_transform(image_size=384)

        self.tag_field_name = tag_field_name

    def process_single(self, sample, rank=None, context=False):
        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.meta][self.tag_field_name] = np.array([[]], dtype=np.str_)
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        model = get_model(self.model_key, rank, self.use_cuda())
        video_tags = []
        for _, value in enumerate(loaded_video_keys):
            video = videos[value]

            # extract frame images
            if self.frame_sampling_method == "all_keyframes":
                frames = extract_key_frames(video)
            elif self.frame_sampling_method == "uniform":
                frames = extract_video_frames_uniformly(video, self.frame_num)
            else:
                video_tags.append([])
                continue

            frame_tensor = torch.stack([self.transform(frame.to_image()) for frame in frames]).to(
                next(model.parameters()).device
            )
            with torch.no_grad():
                tags, _ = model.generate_tag(frame_tensor)

            words = [word.strip() for tag in tags for word in tag.split("|")]
            word_count = Counter(words)
            sorted_word_list = [item for item, _ in word_count.most_common()]
            video_tags.append(np.array(sorted_word_list, dtype=np.str_))

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])

        sample[Fields.meta][self.tag_field_name] = video_tags
        return sample
