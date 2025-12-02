import sys

import librosa
import numpy as np
from librosa.decompose import decompose
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_audio, load_data_with_context

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_AUDIOS

OP_NAME = "audio_nmf_snr_filter"


# run NMF to decompose the signal and noise from the input audio
def separate_signal_noise(audio, n_components=2, nmf_iter=500):
    # convert spectral domain using Short-time Fourier transform
    S = np.abs(librosa.stft(audio))

    # run NMF to decompose the audio
    W, H = decompose(S, n_components=n_components, init="random", random_state=0, max_iter=nmf_iter)

    # get signal and noise
    signal = np.dot(W[:, 0:1], H[0:1, :])
    noise = np.dot(W[:, 1:2], H[1:2, :])

    # convert back to time domain
    signal_audio = librosa.istft(signal * np.exp(1j * np.angle(S)))
    noise_audio = librosa.istft(noise * np.exp(1j * np.angle(S)))

    return signal_audio, noise_audio


# compute the SNR of an audio with NMF algorithm
def compute_nmf_snr(audio_data, nmf_iter=500):
    # separate the signal and noise parts from the original audio
    signal, noise = separate_signal_noise(audio_data, n_components=2, nmf_iter=nmf_iter)

    # compute the power of signal and noise
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)

    # compute SNR in dB
    if power_noise == 0:
        snr = np.finfo(np.float64).max
    else:
        snr = 10 * np.log10(power_signal / power_noise)

    return snr


@OPERATORS.register_module(OP_NAME)
@LOADED_AUDIOS.register_module(OP_NAME)
class AudioNMFSNRFilter(Filter):
    """Keep data samples whose audio Signal-to-Noise Ratios (SNRs) are within a specified
    range.

    This operator computes the SNR of each audio in a sample using Non-negative Matrix
    Factorization (NMF). It then filters the samples based on whether their SNRs fall within
    the given minimum and maximum thresholds. The SNR is computed for each audio, and the
    filtering strategy can be set to either 'any' or 'all'. In 'any' mode, a sample is kept
    if at least one of its audios meets the SNR criteria. In 'all' mode, all audios must
    meet the criteria for the sample to be kept. The NMF computation uses a specified number
    of iterations. If no audio is present in the sample, the SNR is recorded as an empty
    array. The key metric is stored in the 'audio_nmf_snr' field."""

    def __init__(
        self,
        min_snr: float = 0,
        max_snr: float = sys.maxsize,
        nmf_iter_num: PositiveInt = 500,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_snr: The min audio SNR to keep samples in dB. It's 0 by
            default.
        :param max_snr: The max audio SNR to keep samples in dB. It's
            sys.maxsize by default.
        :param nmf_iter_num: The max number of iterations to run NMF. It's 500
            in default.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all audios. 'any': keep this sample if any audios meet the
            condition. 'all': keep this sample only if all audios meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.nmf_iter_num = nmf_iter_num
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.audio_nmf_snr in sample[Fields.stats]:
            return sample

        # there is no audio in this sample
        if self.audio_key not in sample or not sample[self.audio_key]:
            sample[Fields.stats][StatsKeys.audio_nmf_snr] = np.array([], dtype=np.float64)
            return sample

        # load audios
        loaded_audio_keys = sample[self.audio_key]
        sample, audios = load_data_with_context(sample, context, loaded_audio_keys, load_audio)

        audio_snrs = {audio_key: compute_nmf_snr(audio[0], self.nmf_iter_num) for audio_key, audio in audios.items()}

        # get audio SNRs
        sample[Fields.stats][StatsKeys.audio_nmf_snr] = [audio_snrs[audio_key] for audio_key in sample[self.audio_key]]

        return sample

    def process_single(self, sample):
        audio_snrs = sample[Fields.stats][StatsKeys.audio_nmf_snr]
        keep_bools = np.array([self.get_keep_boolean(snr, self.min_snr, self.max_snr) for snr in audio_snrs])
        if len(keep_bools) <= 0:
            return True

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
