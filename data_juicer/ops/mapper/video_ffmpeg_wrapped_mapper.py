from typing import Dict, List, Optional

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

ffmpeg = LazyLoader("ffmpeg", "ffmpeg-python")
OP_NAME = "video_ffmpeg_wrapped_mapper"


@OPERATORS.register_module(OP_NAME)
class VideoFFmpegWrappedMapper(Mapper):
    """Wraps FFmpeg video filters for processing video files in a dataset.

    This operator applies a specified FFmpeg video filter to each video file in the dataset.
    It supports passing keyword arguments to the filter and global arguments to the FFmpeg
    command line. The processed videos are saved in a specified directory or the same
    directory as the input files. If no filter name is provided, the videos remain
    unmodified. The operator updates the source file paths in the dataset to reflect any
    changes."""

    def __init__(
        self,
        filter_name: Optional[str] = None,
        filter_kwargs: Optional[Dict] = None,
        global_args: Optional[List[str]] = None,
        capture_stderr: bool = True,
        overwrite_output: bool = True,
        save_dir: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param filter_name: ffmpeg video filter name.
        :param filter_kwargs: keyword-arguments passed to ffmpeg filter.
        :param global_args: list-arguments passed to ffmpeg command-line.
        :param capture_stderr: whether to capture stderr.
        :param overwrite_output: whether to overwrite output file.
        :param save_dir: The directory where generated video files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)

        self.filter_name = filter_name
        self.filter_kwargs = filter_kwargs
        self.global_args = global_args
        self.capture_stderr = capture_stderr
        self.overwrite_output = overwrite_output
        self.save_dir = save_dir

    def process_single(self, sample):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.video_key]

        if self.filter_name is None:
            return sample

        loaded_video_keys = sample[self.video_key]
        processed = {}
        for video_key in loaded_video_keys:
            if video_key in processed:
                continue

            output_key = transfer_filename(video_key, OP_NAME, self.save_dir, **self._init_parameters)
            stream = ffmpeg.input(video_key).filter(self.filter_name, **self.filter_kwargs).output(output_key)
            if self.global_args is not None:
                stream = stream.global_args(*self.global_args)
            stream.run(capture_stderr=self.capture_stderr, overwrite_output=self.overwrite_output)
            processed[video_key] = output_key

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(loaded_video_keys):
            if sample[Fields.source_file][i] != value:
                if processed[value] != value:
                    sample[Fields.source_file][i] = value

        sample[self.video_key] = [processed[key] for key in loaded_video_keys]
        return sample
