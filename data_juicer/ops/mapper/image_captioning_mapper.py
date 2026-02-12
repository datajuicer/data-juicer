import copy
import random
from typing import List, Optional, Tuple

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
        gpu_batch_size: PositiveInt = 8,
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
        :param gpu_batch_size: the batch size for GPU inference. This controls
            how many images are processed together in a single GPU forward pass.
            Useful when the dataset batch size is larger than what the GPU can
            handle. Default is 8.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["memory"] = "16GB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]

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
        self.gpu_batch_size = gpu_batch_size
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
        """Reduce multiple candidate captions to the final caption(s) for a single image.

        Args:
            chunk: The text chunk containing the image placeholder.
            generated_text_candidates_single_chunk: List of candidate captions.

        Returns:
            List of reduced captions based on keep_candidate_mode.
        """
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

    def _batched_generate(
        self,
        images: List,
        prompts: Optional[List[str]],
        model,
        processor,
    ) -> List[List[str]]:
        """Generate captions for a batch of images using GPU batching.

        This method processes images in sub-batches of size gpu_batch_size to
        prevent GPU memory overflow while maximizing throughput.

        Args:
            images: List of PIL Image objects to generate captions for.
            prompts: Optional list of prompts, one per image. If None, no prompts
                are used.
            model: The HuggingFace model for caption generation.
            processor: The HuggingFace processor for the model.

        Returns:
            A 2D list where result[i][j] is the j-th candidate caption for the
            i-th image. Shape: [num_images, caption_num].
        """
        if not images:
            return []

        num_images = len(images)
        # Initialize result: [num_images][caption_num]
        all_captions = [[] for _ in range(num_images)]

        # Generate caption_num candidates for each image
        for caption_idx in range(self.caption_num):
            # Process images in sub-batches of gpu_batch_size
            batch_captions = []
            for batch_start in range(0, num_images, self.gpu_batch_size):
                batch_end = min(batch_start + self.gpu_batch_size, num_images)
                batch_images = images[batch_start:batch_end]

                # Handle prompts for this batch
                # Pass None only when all prompts in batch are None
                if prompts is None:
                    batch_prompts = None
                else:
                    batch_prompts = prompts[batch_start:batch_end]
                    if all(p is None for p in batch_prompts):
                        batch_prompts = None

                # Prepare inputs for the model
                inputs = processor(images=batch_images, text=batch_prompts, return_tensors="pt").to(model.device)

                # Generate captions
                generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True)
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                batch_captions.extend(generated_texts)

            # Distribute this round's captions to each image
            for img_idx, caption in enumerate(batch_captions):
                all_captions[img_idx].append(caption)

        return all_captions

    def _distribute_captions(
        self,
        all_captions: List[List[str]],
        image_to_sample_chunk: List[Tuple[int, int, int]],
        reconstructed_samples: List[dict],
    ) -> List[dict]:
        """Distribute generated captions back to samples and create output samples.

        This method takes the flat list of captions generated for all images and
        distributes them back to their corresponding samples, handling the text
        assembly with special tokens.

        Args:
            all_captions: 2D list where all_captions[i][j] is the j-th candidate
                caption for the i-th image (in flattened order across all samples).
            image_to_sample_chunk: List of tuples (sample_idx, chunk_idx, image_idx_in_chunk)
                that maps each image in all_captions back to its source sample and chunk.
            reconstructed_samples: List of original sample dictionaries.

        Returns:
            List of generated sample dictionaries with captions inserted.
        """
        samples_after_generation = []

        for sample_idx, ori_sample in enumerate(reconstructed_samples):
            # Add original sample if configured to keep it
            if self.keep_original_sample:
                samples_after_generation.append(ori_sample)

            # Skip samples without images
            if self.image_key not in ori_sample or not ori_sample[self.image_key]:
                continue

            # Prepare generated sample templates
            generated_samples = [copy.deepcopy(ori_sample) for _ in range(self.num_newly_generated_samples)]
            for generated_sample in generated_samples:
                generated_sample[self.text_key] = ""

            # Get all captions for images in this sample
            sample_image_captions = [
                (chunk_idx, img_idx_in_chunk, all_captions[global_img_idx])
                for global_img_idx, (s_idx, chunk_idx, img_idx_in_chunk) in enumerate(image_to_sample_chunk)
                if s_idx == sample_idx
            ]

            # Group captions by chunk
            chunks = ori_sample[self.text_key].split(SpecialTokens.eoc)
            chunk_captions = {}  # chunk_idx -> {img_idx_in_chunk -> [captions]}
            for chunk_idx, img_idx_in_chunk, captions in sample_image_captions:
                if chunk_idx not in chunk_captions:
                    chunk_captions[chunk_idx] = {}
                chunk_captions[chunk_idx][img_idx_in_chunk] = captions

            # Process each chunk
            for chunk_idx, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue

                img_count = chunk.count(SpecialTokens.image)
                text_with_only_special_tokens = remove_non_special_tokens(chunk)

                # Handle chunks with no images - just preserve special tokens
                if img_count == 0:
                    for i in range(self.num_newly_generated_samples):
                        generated_samples[i][self.text_key] += f"{text_with_only_special_tokens}{SpecialTokens.eoc}"
                    continue

                # Reduce captions for each image in this chunk
                new_generated_text_all_images = [[] for _ in range(self.num_newly_generated_samples)]

                for img_idx_in_chunk in range(img_count):
                    # Get captions for this image
                    if chunk_idx in chunk_captions and img_idx_in_chunk in chunk_captions[chunk_idx]:
                        image_candidate_captions = chunk_captions[chunk_idx][img_idx_in_chunk]
                    else:
                        # Fallback: no captions available (shouldn't happen normally)
                        image_candidate_captions = [""] * self.caption_num

                    # Reduce captions based on mode
                    reduced_captions = self._reduce_captions_per_image(chunk, image_candidate_captions)

                    for i, caption in enumerate(reduced_captions):
                        new_generated_text_all_images[i].append(caption)

                # Insert captions into generated samples
                placeholders = [SpecialTokens.image] * img_count
                for i in range(self.num_newly_generated_samples):
                    new_text = insert_texts_after_placeholders(
                        original_string=text_with_only_special_tokens,
                        placeholders=placeholders,
                        new_texts=new_generated_text_all_images[i],
                    )
                    generated_samples[i][self.text_key] += f"{new_text}{SpecialTokens.eoc}"

            # Add generated samples to output
            samples_after_generation.extend(generated_samples)

        return samples_after_generation

    def process_batched(self, samples, rank=None):
        """Process a batch of samples with true GPU batching for caption generation.

        This method collects all images from all samples in the batch, generates
        captions for them in GPU-efficient sub-batches, and then distributes the
        captions back to their respective samples.

        Note:
            This is a batched_OP, whose input and output type are
            both list. Suppose there are $N$ input sample list with batch
            size as $b$, and denote caption_num as $M$.
            the number of total samples after generation is $2Nb$
            for 'random_any' and 'similar_one' mode,
            and $(1+M)Nb$ for 'all' mode.

        Args:
            samples: Dict of lists containing the batch of samples.
            rank: Optional GPU rank for distributed processing.

        Returns:
            Dict of lists containing the processed samples with generated captions.
        """
        # Reconstruct samples from "dict of lists" to "list of dicts"
        batch_size = len(samples[self.text_key])
        reconstructed_samples = [{key: samples[key][i] for key in samples} for i in range(batch_size)]

        # Collect all images and their metadata from all samples
        all_images = []
        all_prompts = []
        # Maps global image index -> (sample_idx, chunk_idx, image_idx_in_chunk)
        image_to_sample_chunk = []

        for sample_idx, sample in enumerate(reconstructed_samples):
            # Skip samples without images
            if self.image_key not in sample or not sample[self.image_key]:
                continue

            # Load all images for this sample
            loaded_image_keys = sample[self.image_key]
            images_cache = {}
            for image_key in loaded_image_keys:
                if image_key not in images_cache:
                    images_cache[image_key] = load_image(image_key)

            # Determine prompt for this sample
            if self.prompt_key and isinstance(sample.get(self.prompt_key), str):
                sample_prompt = sample[self.prompt_key]
            elif self.prompt and isinstance(self.prompt, str):
                sample_prompt = self.prompt
            else:
                sample_prompt = None

            # Process each chunk in the sample's text
            offset = 0
            chunks = sample[self.text_key].split(SpecialTokens.eoc)
            for chunk_idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                img_count = chunk.count(SpecialTokens.image)
                for img_idx_in_chunk in range(img_count):
                    image_key = loaded_image_keys[offset + img_idx_in_chunk]
                    image = images_cache[image_key]
                    all_images.append(image)
                    all_prompts.append(sample_prompt)
                    image_to_sample_chunk.append((sample_idx, chunk_idx, img_idx_in_chunk))

                offset += img_count

        # Handle case where there are no images to process
        if not all_images:
            # Just return original samples if keep_original_sample is True
            if self.keep_original_sample:
                return samples
            # Otherwise return empty result
            keys = samples.keys()
            return {key: [] for key in keys}

        # Get model and processor
        model, processor = get_model(self.model_key, rank, self.use_cuda())
        model.config.image_token_index = 50265

        # Generate captions for all images in batched manner
        prompts_for_generation = all_prompts if any(p is not None for p in all_prompts) else None
        all_captions = self._batched_generate(
            images=all_images,
            prompts=prompts_for_generation,
            model=model,
            processor=processor,
        )

        # Distribute captions back to samples
        samples_after_generation = self._distribute_captions(
            all_captions=all_captions,
            image_to_sample_chunk=image_to_sample_chunk,
            reconstructed_samples=reconstructed_samples,
        )

        # Handle edge case: no samples after generation
        if not samples_after_generation:
            keys = samples.keys()
            return {key: [] for key in keys}

        # Reconstruct samples from "list of dicts" to "dict of lists"
        keys = samples_after_generation[0].keys()
        res_samples = {key: [s[key] for s in samples_after_generation] for key in keys}

        return res_samples
