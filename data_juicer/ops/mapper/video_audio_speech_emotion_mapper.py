import gc

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import extract_audio_from_video
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

librosa = LazyLoader("librosa")
torch = LazyLoader("torch")

OP_NAME = "video_audio_speech_emotion_mapper"


@OPERATORS.register_module(OP_NAME)
class VideoAudioSpeechEmotionMapper(Mapper):
    """Mapper to generate video tags from audio streams extracted by video
    using the Audio Spectrogram Transformer.

    Source: This operator is a part of HumanVBench (CVPR 2026).
    """

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self, model_dir_emo="FunAudioLLM/SenseVoiceSmall", speech_Emo: str = MetaKeys.speech_emotion, *args, **kwargs
    ):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        kwargs.setdefault("mem_required", "20GB")
        super().__init__(*args, **kwargs)
        self._batched_op = True
        self._model_sampling_rate = 16000
        self.model_dir_emo = model_dir_emo

        self.model_key = prepare_model(
            model_type="SenseVoiceSmall",
            pretrained_model_name_or_path=self.model_dir_emo,
        )

        self.speech_Emo = speech_Emo
        self._no_audio_label = "EMPTY"

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if Fields.meta not in sample:
            sample[Fields.meta] = {}

        if self.speech_Emo in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.meta][self.speech_Emo] = []
            return sample

        if MetaKeys.video_audio_tags not in sample[Fields.meta]:
            raise ValueError("video_active_speaker_mapper must be operated after video_tagging_from_audio_mapper.")

        # load video paths
        loaded_video_keys = sample[self.video_key]
        audio_tags = sample[Fields.meta][MetaKeys.video_audio_tags]

        Emo_model, kwargs1 = get_model(self.model_key, rank=rank)

        video_audio_tags = []
        for id, video_path in enumerate(loaded_video_keys):
            if audio_tags[id] == "Speech":
                # only extract audio data and sr for index 0 for now
                ys, srs, valid_indexes = extract_audio_from_video(video_path, stream_indexes=[0])
                if len(valid_indexes) == 0:
                    # there is no valid audio streams. Skip!
                    video_audio_tags.append(self._no_audio_label)
                    continue

                # inference
                y = ys[0]
                sr = srs[0]
                # check if it meets the sampling rate condition of the model
                if sr != self._model_sampling_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self._model_sampling_rate)
                    sr = self._model_sampling_rate

                inputs = torch.tensor(y).to(next(Emo_model.parameters()).device)
                with torch.no_grad():
                    output_emo = Emo_model.inference(
                        data_in=inputs,
                        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                        use_itn=False,
                        **kwargs1,
                    )

                video_audio_tags.append(output_emo[0][0]["text"].split("<|", 2)[-1].split("|>")[0])
            else:
                video_audio_tags.append("")

        sample[Fields.meta][self.speech_Emo] = video_audio_tags
        gc.collect()
        torch.cuda.empty_cache()
        return sample
