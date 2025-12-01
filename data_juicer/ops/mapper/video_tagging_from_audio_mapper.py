import numpy as np

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import extract_audio_from_video
from data_juicer.utils.model_utils import get_model, prepare_model, torch

from ..base_op import OPERATORS, TAGGING_OPS, Mapper

OP_NAME = "video_tagging_from_audio_mapper"
torchaudio = LazyLoader("torchaudio")


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class VideoTaggingFromAudioMapper(Mapper):
    """Generates video tags from audio streams using the Audio Spectrogram Transformer.

    This operator extracts audio streams from videos and uses a Hugging Face Audio
    Spectrogram Transformer (AST) model to generate tags. The tags are stored in the
    specified metadata field, defaulting to 'video_audio_tags'. If no valid audio stream is
    found, the tag is set to 'EMPTY'. The operator resamples audio to match the model's
    required sampling rate if necessary. The tags are inferred based on the highest logit
    value from the model's output. If the tags are already present in the sample, the
    operator skips processing for that sample."""

    _accelerator = "cuda"

    def __init__(
        self,
        hf_ast: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        trust_remote_code: bool = False,
        tag_field_name: str = MetaKeys.video_audio_tags,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_ast: path to the HF model to tag from audios.
        :param trust_remote_code: whether to trust the remote code of HF models
        :param tag_field_name: the field name to store the tags. It's
            "video_audio_tags" in default.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["mem_required"] = "500MB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_ast, trust_remote_code=trust_remote_code
        )
        self._model_sampling_rate = 16000
        self._no_audio_label = "EMPTY"

        self.tag_field_name = tag_field_name

    def process_single(self, sample, rank=None):
        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.meta][self.tag_field_name] = np.array([], dtype=np.str_)
            return sample

        # load video paths
        loaded_video_keys = sample[self.video_key]

        model, feature_extractor = get_model(self.model_key, rank, self.use_cuda())
        video_audio_tags = []
        for video_path in loaded_video_keys:
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
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self._model_sampling_rate)
                y = resampler(torch.from_numpy(y).float()).numpy()
                sr = self._model_sampling_rate
            inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_tag_id = torch.argmax(logits, dim=-1).item()
            predicted_tag = model.config.id2label[predicted_tag_id]
            video_audio_tags.append(predicted_tag)
        sample[Fields.meta][self.tag_field_name] = np.array(video_audio_tags, dtype=np.str_)
        return sample
