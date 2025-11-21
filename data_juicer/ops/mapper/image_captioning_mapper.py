import copy
import random
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    insert_texts_after_placeholders,
    load_image,
    remove_non_special_tokens,
    remove_special_tokens,
)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

simhash = LazyLoader("simhash", "simhash-pybind")

OP_NAME = "image_captioning_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageCaptioningMapper(Mapper):
    """Generates image captions using a Hugging Face model and appends them to samples.

    This operator generates captions for images in the input samples using a specified
    Hugging Face model. It can generate multiple captions per image and apply different
    strategies to retain the generated captions. The operator supports three retention
    modes: 'random_any', 'similar_one_simhash', and 'all'. In 'random_any' mode, a random
    caption is retained. In 'similar_one_simhash' mode, the most similar caption to the
    original text (based on SimHash) is retained. In 'all' mode, all generated captions are
    concatenated and retained. The operator can also keep or discard the original sample
    based on the `keep_original_sample` parameter. If both `prompt` and `prompt_key` are
    set, the `prompt_key` takes precedence."""

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        hf_img2seq: str = "Salesforce/blip2-opt-2.7b",
        trust_remote_code: bool = False,
        caption_num: PositiveInt = 1,
        keep_candidate_mode: str = "random_any",
        keep_original_sample: bool = True,
        prompt: Optional[str] = None,
        prompt_key: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_img2seq: model name on huggingface to generate caption
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param caption_num: how many candidate captions to generate
            for each image
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
        :param prompt: a string prompt to guide the generation of blip2 model
            for all samples globally. It's None in default, which means no
            prompt provided.
        :param prompt_key: the key name of fields in samples to store prompts
            for each sample. It's used for set different prompts for different
            samples. If it's none, use prompt in parameter "prompt". It's None
            in default.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["mem_required"] = "16GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]

        super().__init__(*args, **kwargs)

        if keep_candidate_mode not in ["random_any", "similar_one_simhash", "all"]:
            raise ValueError(
                f"Keep strategy [{keep_candidate_mode}] is not supported. "
                f"Can only be one of "
                f'["random_any", "similar_one_simhash", "all"].'
            )

        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_img2seq, trust_remote_code=trust_remote_code
        )
        self.caption_num = caption_num
        self.keep_candidate_mode = keep_candidate_mode
        self.keep_original_sample = keep_original_sample
        self.prompt = prompt
        self.prompt_key = prompt_key
        self.extra_args = kwargs
        if keep_candidate_mode in ["random_any", "similar_one_simhash"]:
            self.num_newly_generated_samples = 1
        elif keep_candidate_mode in ["all"]:
            self.num_newly_generated_samples = self.caption_num
        else:
            self.num_newly_generated_samples = 0

        # report a warning when both prompt and prompt_key are set
        if self.prompt and self.prompt_key:
            logger.warning(
                "Both the parameter `prompt` and `prompt_key` are " "set. Data-Juicer will consider `prompt_key` first."
            )

    def _process_single_sample(self, ori_sample, rank=None):
        """

        :param ori_sample: a single data sample before applying generation
        :return: batched results after generation
        """
        # there is no image in this sample
        if self.image_key not in ori_sample or not ori_sample[self.image_key]:
            return []

        # the generated results
        generated_samples = [copy.deepcopy(ori_sample) for _ in range(self.num_newly_generated_samples)]
        for generated_sample in generated_samples:
            generated_sample[self.text_key] = ""

        # 1. load all image(s)
        loaded_image_keys = ori_sample[self.image_key]
        images = {}
        for loaded_image_key in loaded_image_keys:
            if loaded_image_key not in images:
                # avoid loading the same images
                image = load_image(loaded_image_key)
                images[loaded_image_key] = image

        offset = 0

        # we follow such assumption:
        # all text/img/video/audio data within a chunk are correlated.
        # As a result,
        # the original text will be removed,
        # the generated text will be placed following each SpecialTokens.img
        # and the original special tokens are kept in an order-preserving way.

        model, processor = get_model(self.model_key, rank, self.use_cuda())
        model.config.image_token_index = 50265

        # do generation for each image chunk by chunk
        for chunk in ori_sample[self.text_key].split(SpecialTokens.eoc):
            # skip empty chunks or contents after the last eoc token
            if not chunk.strip():
                continue

            img_count = chunk.count(SpecialTokens.image)
            text_with_only_special_tokens = remove_non_special_tokens(chunk)
            image_chunk = []
            for image_key in loaded_image_keys[offset : offset + img_count]:
                image = images[image_key]
                image_chunk.append(image)

            # 2. generate candidate caption(s) in batch manner
            generated_text_candidates_single_chunk = [[] for _ in range(self.caption_num)]
            # an assistant 2-D array,
            # generated_text_candidates_single_chunk[i][j] indicates
            # the $i$-th generated candidate for the $j$-th image

            # construct prompts
            if self.prompt_key and isinstance(ori_sample[self.prompt_key], str):
                # check prompt_key is not None, and it's a str in the sample
                prompt_texts = [ori_sample[self.prompt_key]] * len(image_chunk)
            elif self.prompt and isinstance(self.prompt, str):
                # check prompt is not None, and it's a str
                prompt_texts = [self.prompt] * len(image_chunk)
            else:
                prompt_texts = None

            inputs = processor(images=image_chunk, text=prompt_texts, return_tensors="pt").to(model.device)
            for i in range(self.caption_num):
                generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                generated_text_candidates_single_chunk[i] = generated_text

            # 3. insert a list of generated captions into the positions of
            # subsequent placeholders in the original string
            new_generated_text_all_images = [[] for _ in range(self.num_newly_generated_samples)]
            # new_generated_text_all_images is a helper array, element [i][j]
            # denotes the reduced $i$-th result for the $j$-th image

            # reduce the captions according to given mode image by image
            for j in range(img_count):
                new_generated_text_per_image = self._reduce_captions_per_image(
                    chunk, [captions[j] for captions in generated_text_candidates_single_chunk]
                )
                assert self.num_newly_generated_samples == len(new_generated_text_per_image)
                for i in range(len(new_generated_text_per_image)):
                    new_generated_text_all_images[i].append(new_generated_text_per_image[i])

            # insert the captions according to given mode
            place_holders = [SpecialTokens.image] * img_count
            for i in range(self.num_newly_generated_samples):
                new_generated_text_per_chunk = insert_texts_after_placeholders(
                    original_string=text_with_only_special_tokens,
                    placeholders=place_holders,
                    new_texts=new_generated_text_all_images[i],
                )
                generated_samples[i][self.text_key] += f"{new_generated_text_per_chunk}{SpecialTokens.eoc}"

            offset += img_count

        return generated_samples

    def _reduce_captions_per_image(self, chunk, generated_text_candidates_single_chunk):
        new_generated_text_per_chunk = []
        if self.keep_candidate_mode == "random_any":
            new_generated_text_per_chunk.append(random.choice(generated_text_candidates_single_chunk))
        elif self.keep_candidate_mode == "all":
            new_generated_text_per_chunk.extend(generated_text_candidates_single_chunk)
        elif self.keep_candidate_mode == "similar_one_simhash":
            from ..deduplicator.document_simhash_deduplicator import (
                DocumentSimhashDeduplicator,
            )

            ori_normal_text = remove_special_tokens(chunk)
            # using a simhash OP to calculate their similarity
            # NOTE: simhash is just one method to calculate the similarities
            # between texts, but not the most accurate one. More methods (e.g.
            # embedding-based, ...) will be added.
            op_simhash = DocumentSimhashDeduplicator(window_size=2, **self.extra_args)
            ori_text_hash = np.uint64(op_simhash.compute_hash({op_simhash.text_key: ori_normal_text})[HashKeys.simhash])
            generated_text_hashes = [
                np.uint64(op_simhash.compute_hash({op_simhash.text_key: candidate_text})[HashKeys.simhash])
                for candidate_text in generated_text_candidates_single_chunk
            ]
            hamming_distances = [
                simhash.num_differing_bits(ori_text_hash, generated_text_hash)
                for generated_text_hash in generated_text_hashes
            ]
            max_index = min(range(len(hamming_distances)), key=hamming_distances.__getitem__)
            new_generated_text_per_chunk.append(generated_text_candidates_single_chunk[max_index])
        return new_generated_text_per_chunk

    def process_batched(self, samples, rank=None):
        """
        Note:
            This is a batched_OP, whose input and output type are
            both list. Suppose there are $N$ input sample list with batch
            size as $b$, and denote caption_num as $M$.
            the number of total samples after generation is $2Nb$
            for 'random_any' and 'similar_one' mode,
            and $(1+M)Nb$ for 'all' mode.

        :param samples:
        :return:
        """
        # reconstruct samples from "dict of lists" to "list of dicts"
        reconstructed_samples = []
        for i in range(len(samples[self.text_key])):
            reconstructed_samples.append({key: samples[key][i] for key in samples})
        samples_after_generation = []
        # do generation for each sample within the batch
        for ori_sample in reconstructed_samples:
            if self.keep_original_sample:
                samples_after_generation.append(ori_sample)
            generated_samples = self._process_single_sample(ori_sample, rank=rank)
            if len(generated_samples) != 0:
                samples_after_generation.extend(generated_samples)
        # reconstruct samples from "list of dicts" to "dict of lists"
        keys = samples_after_generation[0].keys()
        res_samples = {}
        for key in keys:
            res_samples[key] = [s[key] for s in samples_after_generation]

        return res_samples
