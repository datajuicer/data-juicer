import sys

import librosa
import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_audio, load_data_with_context

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_AUDIOS

OP_NAME = "audio_duration_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_AUDIOS.register_module(OP_NAME)
class AudioDurationFilter(Filter):
    """Keep data samples whose audio durations are within a specified range.

    This operator filters data samples based on the duration of their audio files. It keeps
    samples where the audio duration is between a minimum and maximum value, in seconds. The
    operator supports two strategies for keeping samples: 'any' (keep if any audio meets the
    condition) or 'all' (keep only if all audios meet the condition). The audio duration is
    computed using the `librosa` library. If the audio duration has already been computed,
    it is retrieved from the sample's stats under the key 'audio_duration'. If no audio is
    present in the sample, an empty array is stored in the stats."""

    def __init__(
        self, min_duration: int = 0, max_duration: int = sys.maxsize, any_or_all: str = "any", *args, **kwargs
    ):
        """
        Initialization method.

        :param min_duration: The min audio duration to keep samples in seconds.
            It's 0 by default.
        :param max_duration: The max audio duration to keep samples in seconds.
            It's sys.maxsize by default.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all audios. 'any': keep this sample if any audios meet the
            condition. 'all': keep this sample only if all audios meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_duration = min_duration
        self.max_duration = max_duration
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.audio_duration in sample[Fields.stats]:
            return sample

        # there is no audio in this sample
        if self.audio_key not in sample or not sample[self.audio_key]:
            sample[Fields.stats][StatsKeys.audio_duration] = np.array([], dtype=np.float64)
            return sample

        # load audios
        loaded_audio_keys = sample[self.audio_key]
        sample, audios = load_data_with_context(sample, context, loaded_audio_keys, load_audio)

        audio_durations = {
            audio_key: librosa.get_duration(y=audio[0], sr=audio[1]) for audio_key, audio in audios.items()
        }

        # get audio durations
        sample[Fields.stats][StatsKeys.audio_duration] = [
            audio_durations[audio_key] for audio_key in sample[self.audio_key]
        ]

        return sample

    def process_single(self, sample):
        audio_durations = sample[Fields.stats][StatsKeys.audio_duration]
        keep_bools = np.array(
            [self.get_keep_boolean(duration, self.min_duration, self.max_duration) for duration in audio_durations]
        )
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
