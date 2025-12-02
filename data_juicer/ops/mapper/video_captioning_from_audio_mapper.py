import copy
import os

import regex as re

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens, extract_audio_from_video
from data_juicer.utils.model_utils import get_model, prepare_model, torch

from ..base_op import OPERATORS, Mapper

NAME = "video_captioning_from_audio_mapper"


@OPERATORS.register_module(NAME)
class VideoCaptioningFromAudioMapper(Mapper):
    """Mapper to caption a video according to its audio streams based on
    Qwen-Audio model.
    """

    _accelerator = "cuda"
    _batched_op = True

    def __init__(self, keep_original_sample: bool = True, *args, **kwargs):
        """
        Initialization method.

        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only captioned sample in the
            final datasets and the original sample will be removed. It's True
            in default.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["mem_required"] = "30GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        LazyLoader.check_packages(["transformers", "transformers_stream_generator", "einops", "accelerate", "tiktoken"])

        self.keep_original_sample = keep_original_sample
        self.extra_args = kwargs

        self._hf_qwen_audio = "Qwen/Qwen-Audio"
        self.model_key = prepare_model(
            model_type="huggingface",
            pretrained_model_name_or_path=self._hf_qwen_audio,
            trust_remote_code=True,
        )
        self.prompt = "<|startoftranscription|><|unknown|><|caption|>" "<|unknown|><|notimestamps|><|wo_itn|>"
        self.response_remove_pattern = re.compile(r"<\|.*?\|>")

    def _process_single_sample(self, sample, rank=None):
        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            return []

        # get paths of all video(s)
        loaded_video_keys = sample[self.video_key]

        # get models
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        offset = 0
        captioned_sample = copy.deepcopy(sample)
        # generate for each video chunk by chunk
        captioned_texts = ""
        left_video_keys = []
        for chunk in sample[self.text_key].split(SpecialTokens.eoc):
            # skip empty chunks
            if not chunk.strip():
                continue

            vid_count = chunk.count(SpecialTokens.video)

            captioned_text_list = []
            for video in loaded_video_keys[offset : offset + vid_count]:
                # only extract audio for index 0 for now
                _, _, valid_indexes = extract_audio_from_video(video, video + ".mp3", stream_indexes=[0])
                if len(valid_indexes) == 0:
                    # there is no valid audio streams. Skip!
                    continue
                extracted_audio_path = video + "_0.mp3"
                query = f"<audio>{extracted_audio_path}</audio>{self.prompt}"

                # start to inference
                audio_info = processor.process_audio(query)
                inputs = processor(query, return_tensors="pt", audio_info=audio_info).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, audio_info=audio_info)
                response = processor.decode(outputs[0], skip_special_tokens=True, audio_info=audio_info)
                # remove audio path
                response = response.replace(extracted_audio_path, "").replace("<audio>", "").replace("</audio>", "")
                response = self.response_remove_pattern.sub("", response).strip()
                if response == "":
                    # generate failure. Skip!
                    continue
                captioned_text_list.append(f"{SpecialTokens.video} {response}")
                left_video_keys.append(video)
                # remove extracted audio files
                os.remove(extracted_audio_path)
            offset += vid_count
            captioned_text = "".join(captioned_text_list)

            # add special tokens
            captioned_texts += f"{captioned_text}{SpecialTokens.eoc}"

        captioned_sample[self.text_key] = captioned_texts
        captioned_sample[self.video_key] = left_video_keys
        return [captioned_sample]

    def process_batched(self, samples, rank=None):
        # reconstruct samples from "dict of lists" to "list of dicts"
        reconstructed_samples = []
        for i in range(len(samples[self.text_key])):
            reconstructed_samples.append({key: samples[key][i] for key in samples})
        samples_after_split = []
        # do split for each sample within the batch
        for ori_sample in reconstructed_samples:
            if self.keep_original_sample:
                samples_after_split.append(ori_sample)
            generated_samples = self._process_single_sample(ori_sample, rank=rank)
            if len(generated_samples) != 0:
                samples_after_split.extend(generated_samples)
        # reconstruct samples from "list of dicts" to "dict of lists"
        keys = samples_after_split[0].keys()
        res_samples = {}
        for key in keys:
            res_samples[key] = [s[key] for s in samples_after_split]

        return res_samples
