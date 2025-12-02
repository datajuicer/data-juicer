from typing import Dict, List, Optional

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

ffmpeg = LazyLoader("ffmpeg", "ffmpeg-python")

OP_NAME = "audio_ffmpeg_wrapped_mapper"


@OPERATORS.register_module(OP_NAME)
class AudioFFmpegWrappedMapper(Mapper):
    """Wraps FFmpeg audio filters for processing audio files in a dataset.

    This operator applies specified FFmpeg audio filters to the audio files in the dataset.
    It supports passing custom filter parameters and global arguments to the FFmpeg command
    line. The processed audio files are saved to a specified directory or the same directory
    as the input files if no save directory is provided. The `DJ_PRODUCED_DATA_DIR`
    environment variable can also be used to set the save directory. If no filter name is
    provided, the audio files remain unmodified. The operator updates the source file paths
    in the dataset after processing."""

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

        :param filter_name: ffmpeg audio filter name.
        :param filter_kwargs: keyword-arguments passed to ffmpeg filter.
        :param global_args: list-arguments passed to ffmpeg command-line.
        :param capture_stderr: whether to capture stderr.
        :param overwrite_output: whether to overwrite output file.
        :param save_dir: The directory where generated audio files will be stored.
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
        # there is no audio in this sample
        if self.audio_key not in sample or not sample[self.audio_key]:
            sample[Fields.source_file] = []
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.audio_key]

        if self.filter_name is None:
            return sample

        loaded_audio_keys = sample[self.audio_key]
        processed = {}
        for audio_key in loaded_audio_keys:
            if audio_key in processed:
                continue

            output_key = transfer_filename(audio_key, OP_NAME, self.save_dir, **self._init_parameters)
            stream = ffmpeg.input(audio_key).filter(self.filter_name, **self.filter_kwargs).output(output_key)
            if self.global_args is not None:
                stream = stream.global_args(*self.global_args)
            stream.run(capture_stderr=self.capture_stderr, overwrite_output=self.overwrite_output)
            processed[audio_key] = output_key

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(loaded_audio_keys):
            if sample[Fields.source_file][i] != value:
                if processed[value] != value:
                    sample[Fields.source_file][i] = value

        sample[self.audio_key] = [processed[key] for key in loaded_audio_keys]
        return sample
