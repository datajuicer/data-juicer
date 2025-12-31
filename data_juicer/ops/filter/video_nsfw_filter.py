from functools import partial
from typing import Optional

import numpy as np
from PIL import Image
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    extract_video_frames_uniformly,
    load_data_with_context,
    load_image,
)
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.video_utils import create_video_reader

from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_SAMPLED_FRAMES, LOADED_VIDEOS

torch = LazyLoader("torch")
cv2 = LazyLoader("cv2", "opencv-python")

OP_NAME = "video_nsfw_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoNSFWFilter(Filter):
    """Filter to keep samples whose videos have nsfw scores in a specified range.

    This operator uses a Hugging Face model to detect NSFW content in video frames. It keeps
    samples where the NSFW score is below a specified threshold. The operator supports two
    frame sampling methods: "all_keyframes" and "uniform". For "uniform", it extracts a
    specified number of frames. The NSFW scores are reduced using one of three modes: "avg",
    "max", or "min". The key metric, 'video_nsfw_score', is computed for each video and
    stored in the sample's stats. The operator can use either an "any" or "all" strategy to
    decide if a sample should be kept based on the NSFW scores of its videos."""

    _accelerator = "cuda"

    def __init__(
        self,
        hf_nsfw_model: str = "Falconsai/nsfw_image_detection",
        trust_remote_code: bool = False,
        min_score: float = 0.0,
        max_score: float = 0.5,
        frame_field: Optional[str] = None,
        frame_sampling_method: str = "all_keyframes",
        frame_num: PositiveInt = 3,
        reduce_mode: str = "avg",
        any_or_all: str = "any",
        video_backend: str = "av",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_nsfw_model: nsfw detection model name on huggingface.
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param min_score: the nsfw score threshold for samples.
            range from 0 to 1. Samples with nsfw score greater than this
            threshold will be kept.
        :param max_score: the nsfw score threshold for samples.
            range from 0 to 1. Samples with nsfw score less than this threshold
            will be kept.
        :param frame_field: the field name of video frames to calculate the nsfw score.
            If frame_field is None, extract frames from the video field.
        :param frame_sampling_method: sampling method of extracting frame
            images from the videos.
            Should be one of ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number of which depends
            on the duration of the video) and the latter one extract specified
            number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform" or "frame_field" is given.
            If it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param reduce_mode: reduce mode for multiple sampled video frames.
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all videos. 'any': keep this sample if any videos meet the
            condition. 'all': keep this sample only if all videos meet the
            condition.
        :param video_backend: video backend, can be `ffmpeg`, `av`.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["memory"] = "1GB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        if frame_sampling_method not in ["all_keyframes", "uniform"]:
            raise ValueError(
                f"Frame sampling method "
                f"[{frame_sampling_method}] is not supported. "
                f'Can only be one of ["all_keyframes", "uniform"].'
            )
        if reduce_mode not in ["avg", "max", "min"]:
            raise ValueError(
                f"Reduce mode [{reduce_mode}] is not supported. " f'Can only be one of ["avg", "max", "min"].'
            )
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_nsfw_model, trust_remote_code=trust_remote_code
        )
        self.reduce_mode = reduce_mode
        self.frame_field = frame_field
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        self.video_backend = video_backend
        assert self.video_backend in ["ffmpeg", "av"]
        if self.frame_sampling_method == "uniform":
            assert self.video_backend == "av", "Only 'av' backend is supported for 'uniform' frame sampling method."

        self.sampled_frames_key_suffix = f"-{frame_sampling_method}" + (
            "" if frame_sampling_method == "all_keyframes" else f"-{frame_num}"
        )

    def _calculate_score(self, frame_images, model, processor):
        if len(frame_images) > 0:
            inputs = processor(images=frame_images, return_tensors="pt")
            inputs = inputs.to(model.device)
            outputs = model(**inputs)
            cur_scores = torch.softmax(outputs.logits, dim=-1)[:, 1]

            if self.reduce_mode == "avg":
                cur_score = cur_scores.mean()
            elif self.reduce_mode == "max":
                cur_score = cur_scores.max()
            else:
                cur_score = cur_scores.min()
        else:
            cur_score = 0.0

        return float(cur_score)

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.video_nsfw_score in sample[Fields.stats]:
            return sample

        # priority to get frames from frame_field
        if self.frame_field and self.frame_field in sample:
            videos_frames_path_list = sample[self.frame_field]
        else:
            # there is no videos in this sample
            if self.video_key not in sample or not sample[self.video_key]:
                sample[Fields.stats][StatsKeys.video_nsfw_score] = np.array([], dtype=np.float64)
                return sample

            # load videos
            loaded_video_keys = sample[self.video_key]
            video_reader = partial(create_video_reader, backend=self.video_backend)
            sample, videos = load_data_with_context(sample, context, loaded_video_keys, video_reader)

        nsfw_scores = []
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        if self.frame_field and self.frame_field in sample:
            for frames_path in videos_frames_path_list:
                if self.frame_num < len(frames_path):
                    indices = np.linspace(0, len(frames_path) - 1, self.frame_num, dtype=int)
                    frames_path = [frames_path[i] for i in indices]
                frame_images = [load_image(frame_path) for frame_path in frames_path]
                cur_score = self._calculate_score(frame_images, model, processor)
                nsfw_scores.append(cur_score)
        else:
            # extract frames from videos
            for video_key, video in videos.items():
                sampled_frames_key = video_key + self.sampled_frames_key_suffix

                # extract frame images
                if context and sampled_frames_key in sample[Fields.context]:
                    # context hit
                    frames = sample[Fields.context][sampled_frames_key]
                else:
                    if self.frame_sampling_method == "all_keyframes":
                        frames = video.extract_keyframes().frames
                        frame_images = [Image.fromarray(img) for img in frames]
                    elif self.frame_sampling_method == "uniform":
                        # only support av backend
                        frames = extract_video_frames_uniformly(video.container, self.frame_num)
                        frame_images = [frame.to_image() for frame in frames]
                    else:
                        frames = []

                    # store the sampled frames in the context
                    if context:
                        sample[Fields.context][sampled_frames_key] = frame_images

                cur_score = self._calculate_score(frame_images, model, processor)
                nsfw_scores.append(cur_score)

            if not context:
                for vid_key in videos:
                    videos[vid_key].close()

        sample[Fields.stats][StatsKeys.video_nsfw_score] = nsfw_scores

        return sample

    def process_single(self, sample, rank=None):
        itm_scores = sample[Fields.stats][StatsKeys.video_nsfw_score]
        if len(itm_scores) <= 0:
            return True

        keep_bools = np.array(
            [self.get_keep_boolean(itm_score, self.min_score, self.max_score) for itm_score in itm_scores]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
