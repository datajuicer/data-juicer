# yapf: disable
import copy
import random
from typing import Dict, Optional

import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    close_video,
    insert_texts_after_placeholders,
    load_data_with_context,
    load_video,
    remove_non_special_tokens,
    remove_special_tokens,
)
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    prepare_qwen_vl_inputs_for_vllm,
    torch,
    update_sampling_params,
)

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

vllm = LazyLoader("vllm")
simhash = LazyLoader('simhash', 'simhash-pybind')

OP_NAME = 'video_captioning_from_vlm_mapper'


DEFAULT_PROMPT = '''
Describe the input video in 1-2 sentences.
'''


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningFromVLMMapper(Mapper):
    """Generates video captions using a VLM that accepts videos as inputs.

    This operator processes video samples to generate captions based on the provided video.
    It uses a VLM model that accept videos as inputs, such as 'Qwen/Qwen3-VL-8B-Instruct',
    to generate multiple caption candidates for each video. The number of generated
    captions and the strategy to keep or filter these candidates can be configured.
    The final output can include both the original sample and the generated captions,
    depending on the configuration."""

    _accelerator = 'cuda'
    _batched_op = True

    def __init__(
        self,
        hf_model: str = 'Qwen/Qwen3-VL-8B-Instruct',
        enable_vllm: bool = True,
        caption_num: PositiveInt = 1,
        keep_candidate_mode: str = 'random_any',
        keep_original_sample: bool = True,
        prompt: Optional[str] = None,
        prompt_key: Optional[str] = None,
        model_params: Dict = {},
        sampling_params: Dict = {},
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_model: VLM model name on huggingface
            to generate caption
        :param enable_vllm: If true, use VLLM for loading hugging face or
            local llm.
        :param caption_num: how many candidate captions to generate
            for each video
        :param keep_candidate_mode: retain strategy for the generated
            $caption_num$ candidates.

            'random_any': Retain the random one from generated captions

            'similar_one_simhash': Retain the generated one that is most
                similar to the original caption

            'all': Retain all generated captions by concatenation

        Note:
            This is a batched_OP, whose input and output type are
            both list. Suppose there are $N$ list of input samples, whose batch
            size is $b$, and denote caption_num as $M$.
            The number of total samples after generation is $2Nb$ when
            keep_original_sample is True and $Nb$ when keep_original_sample is
            False. For 'random_any' and 'similar_one_simhash' mode,
            it's $(1+M)Nb$ for 'all' mode when keep_original_sample is True
            and $MNb$ when keep_original_sample is False.

        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only generated captions in the
            final datasets and the original captions will be removed. It's True
            in default.
        :param prompt: a string prompt to guide the generation of video-blip
            model for all samples globally. It's None in default, which means
            using the DEFAULT_PROMPT.
        :param prompt_key: the key name of fields in samples to store prompts
            for each sample. It's used for set different prompts for different
            samples. If it's none, use prompt in parameter "prompt". It's None
            in default.
        :param model_params: Parameters for initializing the model.
        :param sampling_params: Extra parameters passed to the model calling.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param args: extra args
        :param kwargs: extra kwargs
        """
        kwargs["mem_required"] = "20GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)

        if keep_candidate_mode not in [
                'random_any', 'similar_one_simhash', 'all'
        ]:
            raise ValueError(
                f'Keep strategy [{keep_candidate_mode}] is not supported. '
                f'Can only be one of '
                f'["random_any", "similar_one_simhash", "all"].')

        if keep_candidate_mode in ['random_any', 'similar_one_simhash']:
            self.num_newly_generated_samples = 1
        elif keep_candidate_mode in ['all']:
            self.num_newly_generated_samples = caption_num
        else:
            self.num_newly_generated_samples = 0

        # report a warning when both prompt and prompt_key are set
        if prompt and prompt_key:
            logger.warning(
                'Both the parameter `prompt` and `prompt_key` are '
                'set. Data-Juicer will consider `prompt_key` first.')

        self.caption_num = caption_num
        self.keep_candidate_mode = keep_candidate_mode
        self.keep_original_sample = keep_original_sample
        self.prompt = prompt
        self.prompt_key = prompt_key
        self.extra_args = kwargs

        self.enable_vllm = enable_vllm

        if not sampling_params:
            sampling_params = {
                "temperature": 0,
                "max_tokens": 1024,
                "top_k": -1,
                "stop_token_ids": [],
            }

        sampling_params = update_sampling_params(sampling_params, hf_model, self.enable_vllm)

        if self.enable_vllm:
            assert torch.cuda.device_count() >= 1, "must be executed in CUDA"
            # cannot initialize vllm replicas on different GPUs
            self.num_proc = 1
            if model_params.get("tensor_parallel_size") is None:
                tensor_parallel_size = torch.cuda.device_count()
                logger.info(
                    f"Set tensor_parallel_size to \
                                {tensor_parallel_size} for vllm."
                )
                model_params["tensor_parallel_size"] = tensor_parallel_size
            if model_params.get("mm_encoder_tp_mode") is None:
                model_params["mm_encoder_tp_mode"] = "data"
            self.model_key = prepare_model(
                model_type='vllm',
                pretrained_model_name_or_path=hf_model,
                **model_params,
            )
            self.sampling_params = vllm.SamplingParams(**sampling_params)
        else:
            self.model_key = prepare_model(
                model_type='huggingface',
                pretrained_model_name_or_path=hf_model,
                return_pipe=True,
                trust_remote_code=True,
                **model_params,
            )
            self.sampling_params = sampling_params

    def _process_single_sample(self, ori_sample, rank=None, context=False):

        # there is no videos in this sample
        if self.video_key not in ori_sample or not ori_sample[self.video_key]:
            return []

        # the generated results
        generated_samples = [
            copy.deepcopy(ori_sample)
            for _ in range(self.num_newly_generated_samples)
        ]
        for generated_sample in generated_samples:
            generated_sample[self.text_key] = ''

        # load videos
        loaded_video_keys = ori_sample[self.video_key]
        sample, videos = load_data_with_context(ori_sample, context,
                                                loaded_video_keys, load_video)

        text = sample[self.text_key]
        offset = 0
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        for chunk in text.split(SpecialTokens.eoc):

            video_count = chunk.count(SpecialTokens.video)

            # no video or no text
            if video_count == 0 or len(chunk) == 0:
                continue
            else:
                text_with_only_special_tokens = remove_non_special_tokens(
                    chunk)
                # generate candidate caption(s) in batch manner
                generated_text_candidates_single_chunk = [
                    [] for _ in range(self.caption_num)
                ]
                for video_key in loaded_video_keys[offset:offset +
                                                   video_count]:
                    video = videos[video_key]

                    # construct prompts
                    if self.prompt_key and isinstance(
                            ori_sample[self.prompt_key], str):
                        # check prompt_key is not None, and it's a str
                        # in the sample
                        prompt_text = ori_sample[self.prompt_key]
                    elif self.prompt and isinstance(self.prompt, str):
                        # check prompt is not None, and it's a str
                        prompt_text = self.prompt
                    else:
                        prompt_text = DEFAULT_PROMPT

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text,
                                },
                                {
                                    "type": "video",
                                    "video": video,
                                }
                            ]
                        }
                    ]

                    if self.enable_vllm:
                        inputs = prepare_qwen_vl_inputs_for_vllm(messages, processor)
                        for i in range(self.caption_num):
                            outputs = model.generate(inputs, sampling_params=self.sampling_params)
                            generated_text_candidates_single_chunk[i] += outputs.outputs[0].text
                    else:
                        inputs = processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors='pt',
                        ).to(model.device)
                        with torch.no_grad():
                            for i in range(self.caption_num):
                                generated_ids = model.generate(**inputs, max_new_tokens=128)
                                generated_ids_trimmed = [
                                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                ]
                                generated_text = processor.batch_decode(
                                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                )
                                generated_text_candidates_single_chunk[i] += generated_text

                # 3. insert a list of generated captions into the positions of
                # subsequent placeholders in the original string
                new_generated_text_all_videos = [
                    [] for _ in range(self.num_newly_generated_samples)
                ]
                # new_generated_text_all_videos is a helper array,
                # element [i][j]
                # denotes the reduced $i$-th result for the $j$-th video

                # reduce the captions according to given mode video by video
                for j in range(video_count):
                    new_generated_text_per_video = self._reduce_captions(
                        chunk,
                        [
                            captions[j] for captions in
                            generated_text_candidates_single_chunk
                        ],
                    )
                    assert self.num_newly_generated_samples == len(
                        new_generated_text_per_video)
                    for i in range(len(new_generated_text_per_video)):
                        new_generated_text_all_videos[i].append(
                            new_generated_text_per_video[i])

                # insert the captions according to given mode
                place_holders = [SpecialTokens.video] * video_count
                for i in range(self.num_newly_generated_samples):
                    generated_text_per_chunk = insert_texts_after_placeholders(
                        original_string=text_with_only_special_tokens,
                        placeholders=place_holders,
                        new_texts=new_generated_text_all_videos[i],
                    )
                    generated_samples[i][
                        self.
                        text_key] += f'{generated_text_per_chunk}' \
                                     f'{SpecialTokens.eoc}'

                offset += video_count

        if not context:
            for vid_key in videos:
                close_video(videos[vid_key])
        return generated_samples

    def _reduce_captions(self, chunk, generated_text_candidates_single_chunk):
        generated_text_per_chunk = []
        if self.keep_candidate_mode == 'random_any':
            generated_text_per_chunk.append(
                random.choice(generated_text_candidates_single_chunk))
        elif self.keep_candidate_mode == 'all':
            generated_text_per_chunk.extend(
                generated_text_candidates_single_chunk)
        elif self.keep_candidate_mode == 'similar_one_simhash':
            from ..deduplicator.document_simhash_deduplicator import (
                DocumentSimhashDeduplicator,
            )

            ori_normal_text = remove_special_tokens(chunk)
            # using a simhash OP to calculate their similarity
            # NOTE: simhash is just one method to calculate the similarities
            # between texts, but not the most accurate one. More methods (e.g.
            # embedding-based, ...) will be added.
            op_simhash = DocumentSimhashDeduplicator(window_size=2,
                                                     **self.extra_args)
            ori_text_hash = np.uint64(
                op_simhash.compute_hash({op_simhash.text_key:
                                         ori_normal_text})[HashKeys.simhash])
            generated_text_hashes = [
                np.uint64(
                    op_simhash.compute_hash(
                        {op_simhash.text_key:
                         candidate_text})[HashKeys.simhash])
                for candidate_text in generated_text_candidates_single_chunk
            ]
            hamming_distances = [
                simhash.num_differing_bits(ori_text_hash, generated_text_hash)
                for generated_text_hash in generated_text_hashes
            ]
            max_index = min(range(len(hamming_distances)),
                            key=hamming_distances.__getitem__)
            generated_text_per_chunk.append(
                generated_text_candidates_single_chunk[max_index])
        return generated_text_per_chunk

    def process_batched(self, samples, rank=None, context=False):
        """
        :param samples:
        :return:

        Note:
            This is a batched_OP, whose the input and output type are
            both list. Suppose there are $N$ input sample list with batch
            size as $b$, and denote caption_num as $M$.
            the number of total samples after generation is $2Nb$
            for 'random_any' and 'similar_one' mode,
            and $(1+M)Nb$ for 'all' mode.
        """
        # reconstruct samples from "dict of lists" to "list of dicts"
        reconstructed_samples = []
        for i in range(len(samples[self.text_key])):
            reconstructed_samples.append(
                {key: samples[key][i]
                 for key in samples})
        samples_after_generation = []
        # do generation for each sample within the batch
        for ori_sample in reconstructed_samples:
            if self.keep_original_sample:
                samples_after_generation.append(ori_sample)
            generated_samples = self._process_single_sample(ori_sample,
                                                            rank=rank,
                                                            context=context)
            if len(generated_samples) != 0:
                samples_after_generation.extend(generated_samples)
        # reconstruct samples from "list of dicts" to "dict of lists"
        keys = samples_after_generation[0].keys()
        res_samples = {}
        for key in keys:
            res_samples[key] = [s[key] for s in samples_after_generation]

        return res_samples
