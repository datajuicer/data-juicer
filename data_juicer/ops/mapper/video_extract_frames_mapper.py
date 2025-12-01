import io
import os
import os.path as osp

from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.file_utils import dict_to_hash
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    close_video,
    extract_key_frames,
    extract_key_frames_by_seconds,
    extract_video_frames_uniformly,
    extract_video_frames_uniformly_by_seconds,
    load_data_with_context,
    load_video,
)

from ..base_op import OPERATORS, TAGGING_OPS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_extract_frames_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoExtractFramesMapper(Mapper):
    """Mapper to extract frames from video files according to specified methods.

    Extracts frames from video files using either all keyframes or a uniform sampling
    method.

    Supported output formats are: ["path", "bytes"].
    If format is "path", the output is a list of lists, where each inner
    list contains the path of the frames of a single video.
    e.g.[
            [video1_frame1_path, video1_frame2_path, ...],
            [video2_frame1_path, video2_frame2_path, ...],
            ...
        ] (In the order of the videos).
    If format is "bytes", the output is a list of lists, where each inner
    list contains the bytes of the frames of a single video.
    e.g. [
            [video1_byte1, video1_byte2, ...],
            [video2_byte1, video2_byte2, ...],
            ...
        ] (In the order of the videos).

    - **Frame Sampling Methods**:
    - "all_keyframes": Extracts all keyframes from the video.
    - "uniform": Extracts a specified number of frames uniformly from the video.
    - If `duration` is set, the video is segmented into multiple segments based on the
      duration, and frames are extracted from each segment.
    - The output directory for the frames can be specified if output format is "path", else left to None.
    - The field name in the sample's metadata where the frame information is stored can be
      customized."""

    _batched_op = True

    def __init__(
        self,
        frame_sampling_method: str = "all_keyframes",
        output_format: str = "path",
        frame_num: PositiveInt = 3,
        duration: float = 0,
        frame_dir: str = None,
        frame_key: str = MetaKeys.video_frames,
        *args,
        **kwargs,
    ):
        """
        Initialization method.
        :param frame_sampling_method: sampling method of extracting frame
            videos from the videos. Should be one of
            ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number
            of which depends on the duration of the video) and the latter
            one extract specified number of frames uniformly from the video.
            If "duration" > 0, frame_sampling_method acts on every segment.
            Default: "all_keyframes".
        :param output_format: The output format of the frame videos.
            Supported formats are: ["path", "bytes"].
            If format is "path", the output is a list of lists, where each inner
            list contains the path of the frames of a single video.
            e.g.[
                    [video1_frame1_path, video1_frame2_path, ...],
                    [video2_frame1_path, video2_frame2_path, ...],
                    ...
                ] (In the order of the videos).
            If format is "bytes", the output is a list of lists, where each inner
            list contains the bytes of the frames of a single video.
            e.g. [
                    [video1_byte1, video1_byte2, ...],
                    [video2_byte1, video2_byte2, ...],
                    ...
                ] (In the order of the videos).
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
            If "duration" > 0, frame_num is the number of frames per segment.
        :param duration: The duration of each segment in seconds.
            If 0, frames are extracted from the entire video.
            If duration > 0, the video is segmented into multiple segments
            based on duration, and frames are extracted from each segment.
        :param frame_dir: Output directory to save extracted frames.
            If output_format is "path", must specify a directory.
        :param frame_key: The name of field to save generated frames info.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.output_format = output_format.lower()
        assert self.output_format in [
            "path",
            "bytes",
        ], f"output_format '{output_format}' is not supported. Can only be one of ['path', 'bytes']."

        if frame_sampling_method not in ["all_keyframes", "uniform"]:
            raise ValueError(
                f"Frame sampling method "
                f"[{frame_sampling_method}] is not supported. "
                f'Can only be one of ["all_keyframes", "uniform"].'
            )

        self.frame_dir = frame_dir
        if self.output_format in ["path"]:
            assert frame_dir is not None, "'frame_dir' must be specified when output_format is in ['path']."
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        self.duration = duration
        self.frame_key = frame_key
        self.frame_fname_template = "frame_{}.jpg"

    def _get_default_frame_dir(self, original_filepath):
        original_dir = os.path.dirname(original_filepath)
        dir_token = f"/{Fields.multimodal_data_output_dir}/"
        if dir_token in original_dir:
            original_dir = original_dir.split(dir_token)[0]
        saved_dir = os.path.join(original_dir, f"{Fields.multimodal_data_output_dir}/{OP_NAME}")
        original_filename = osp.splitext(osp.basename(original_filepath))[0]
        hash_val = dict_to_hash(self._init_parameters)

        return osp.join(saved_dir, f"{original_filename}__dj_hash_#{hash_val}#")

    def extract_frames(self, video):
        # extract frame videos
        if self.frame_sampling_method == "all_keyframes":
            if self.duration:
                frames = extract_key_frames_by_seconds(video, self.duration)
            else:
                frames = extract_key_frames(video)
        elif self.frame_sampling_method == "uniform":
            if self.duration:
                frames = extract_video_frames_uniformly_by_seconds(video, self.frame_num, duration=self.duration)
            else:
                frames = extract_video_frames_uniformly(video, self.frame_num)
        else:
            raise ValueError(
                f"Not support sampling method \
                `{self.frame_sampling_method}`."
            )
        frames = [frame.to_image() for frame in frames]
        return frames

    def process_single(self, sample, context=False):
        # check if it's generated already
        if self.frame_key in sample:
            return sample

        # there is no videos in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)
        videos_frames_list = [[] for _ in range(len(loaded_video_keys))]
        text = sample[self.text_key]
        offset = 0
        for chunk in text.split(SpecialTokens.eoc):
            video_count = chunk.count(SpecialTokens.video)
            # no video or no text
            if video_count == 0 or len(chunk) == 0:
                continue
            else:
                for video_key in loaded_video_keys[offset : offset + video_count]:
                    video_idx = loaded_video_keys.index(video_key)
                    video = videos[video_key]
                    # extract frame videos
                    frames = self.extract_frames(video)
                    if self.output_format in ["path"]:
                        frame_dir = osp.join(self.frame_dir, osp.splitext(osp.basename(video_key))[0])
                        os.makedirs(frame_dir, exist_ok=True)
                        cur_frames_path = []

                        for i, frame in enumerate(frames):
                            frame_path = osp.join(frame_dir, self.frame_fname_template.format(i))
                            if not os.path.exists(frame_path):
                                frame.save(frame_path)
                            cur_frames_path.append(frame_path)

                        videos_frames_list[video_idx] = cur_frames_path
                    else:
                        cur_frames_bytes = []
                        for i, frame in enumerate(frames):
                            stream = io.BytesIO()
                            frame.save(stream, format="jpeg")
                            cur_frames_bytes.append(stream.getvalue())

                        videos_frames_list[video_idx] = cur_frames_bytes

                offset += video_count

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])

        sample[self.frame_key] = videos_frames_list

        return sample
