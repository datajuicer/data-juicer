import os

import numpy as np

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import load_audio, load_data_with_context

from ...utils.lazy_loader import LazyLoader
from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_AUDIOS

audiomentations = LazyLoader("audiomentations")
sf = LazyLoader("soundfile")

OP_NAME = "audio_add_gaussian_noise_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_AUDIOS.register_module(OP_NAME)
class AudioAddGaussianNoiseMapper(Mapper):
    """Mapper to add Gaussian noise to audio samples.

    This operator adds Gaussian noise to audio data with a specified probability. The
    amplitude of the noise is randomly chosen between `min_amplitude` and `max_amplitude`.
    If `save_dir` is provided, the modified audio files are saved in that directory;
    otherwise, they are saved in the same directory as the input files. The `p` parameter
    controls the probability of applying this transformation to each sample. If no audio is
    present in the sample, it is returned unchanged."""

    def __init__(
        self,
        min_amplitude: float = 0.001,
        max_amplitude: float = 0.015,
        p: float = 0.5,
        save_dir: str = None,
        *args,
        **kwargs,
    ):
        """

        Initialization method.

        :param min_amplitude: float unit: linear amplitude.
            Default: 0.001. Minimum noise amplification factor.
        :param max_amplitude: float unit: linear amplitude.
            Default: 0.015. Maximum noise amplification factor.
        :param p: float range: [0.0, 1.0].  Default: 0.5.
            The probability of applying this transform.
        :param save_dir: str. Default: None.
            The directory where generated audio files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)

        if min_amplitude >= max_amplitude:
            raise ValueError("min_amplitude must be < max_amplitude")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.p = p
        self.audio_transform = audiomentations.AddGaussianNoise(
            min_amplitude=self.min_amplitude, max_amplitude=self.max_amplitude, p=self.p
        )
        self.save_dir = save_dir

    def process_single(self, sample, context=False):
        # there is no audio in this sample
        if self.audio_key not in sample or not sample[self.audio_key]:
            return sample

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.audio_key]

        # load audio
        loaded_audio_keys = sample[self.audio_key]
        sample, audios = load_data_with_context(sample, context, loaded_audio_keys, load_audio)
        processed = {}

        for audio_key in loaded_audio_keys:
            if audio_key in processed:
                continue

            new_audio_key = transfer_filename(audio_key, OP_NAME, self.save_dir, **self._init_parameters)
            if new_audio_key.endswith(".ogg"):
                new_audio_key = new_audio_key.replace(".ogg", ".wav")
            if not os.path.exists(new_audio_key) or new_audio_key not in audios.keys():
                audio_array, audio_sample_rate = audios[audio_key]
                audio_augment = self.audio_transform(audio_array, sample_rate=audio_sample_rate)
                audio_augment = np.asarray(audio_augment)
                sf.write(new_audio_key, audio_augment, audio_sample_rate)
                audios[new_audio_key] = (audio_augment, audio_sample_rate)
                if context:
                    sample[Fields.context][new_audio_key] = audios
            processed[audio_key] = new_audio_key

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(loaded_audio_keys):
            if sample[Fields.source_file][i] != value and processed[value] != value:
                sample[Fields.source_file][i] = processed[value]
        sample[self.audio_key] = [processed[key] for key in loaded_audio_keys]
        return sample
