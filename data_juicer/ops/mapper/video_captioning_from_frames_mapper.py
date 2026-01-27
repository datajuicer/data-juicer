# yapf: disable
import copy
import random
from typing import Optional

import numpy as np
from loguru import logger
from PIL import ImageOps
from pydantic import PositiveInt

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    close_video,
    extract_key_frames,
    extract_video_frames_uniformly,
    insert_texts_after_placeholders,
    load_data_with_context,
    load_image,
    load_video,
    remove_non_special_tokens,
    remove_special_tokens,
)
from data_juicer.utils.model_utils import get_model, prepare_model, torch

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

simhash = LazyLoader('simhash', 'simhash-pybind')

OP_NAME = 'video_captioning_from_frames_mapper'


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningFromFramesMapper(Mapper):
    """Generates video captions from sampled frames using an image-to-text model. Captions from
    different frames are concatenated into a single string.

    - Uses a Hugging Face image-to-text model to generate captions for sampled video frames.
    - Supports different frame sampling methods: 'all_keyframes' or 'uniform'.
    - Can apply horizontal and vertical flips to the frames before captioning.
    - Offers multiple strategies for retaining generated captions: 'random_any',
    'similar_one_simhash', or 'all'.
    - Optionally keeps the original sample in the final dataset.
    - Allows setting a global prompt or per-sample prompts to guide caption generation.
    - Generates a specified number of candidate captions per video, which can be reduced
    based on the selected retention strategy.
    - The number of output samples depends on the retention strategy and whether original
    samples are kept."""

    _accelerator = 'cuda'
    _batched_op = True

    def __init__(
        self,
        hf_img2seq: str = 'Salesforce/blip2-opt-2.7b',
        trust_remote_code: bool = False,
        caption_num: PositiveInt = 1,
        keep_candidate_mode: str = 'random_any',
        keep_original_sample: bool = True,
        prompt: Optional[str] = None,
        prompt_key: Optional[str] = None,
        frame_field: Optional[str] = None,
        frame_sampling_method: str = 'all_keyframes',
        frame_num: PositiveInt = 3,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        text_update_strategy: str = 'rewrite',
        caption_field: Optional[str] = None,
        legacy_split_by_text_token: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_img2seq: model name on huggingface to generate caption
        :param trust_remote_code: whether to trust the remote code of HF models.
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
        :param prompt: a string prompt to guide the generation of image-to-text
            model for all samples globally. It's None in default, which means
            no prompt provided.
        :param prompt_key: the key name of fields in samples to store prompts
            for each sample. It's used for set different prompts for different
            samples. If it's none, use prompt in parameter "prompt". It's None
            in default.
        :param frame_field: the field name of video frames to generate caption.
            If frame_field is None, extract frames from the video field.
        :param frame_sampling_method: sampling method of extracting frame
            videos from the videos. Should be one of ["all_keyframes", "uniform"].
            Only works when "frame_field" is none.
            The former one extracts all key frames (the number
            of which depends on the duration of the video) and the latter
            one extract specified number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from the video frames.
            Only works when "frame_sampling_method" is "uniform" or "frame_field" is given. If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param horizontal_flip: flip frame video horizontally (left to right).
        :param vertical_flip: flip frame video vertically (top to bottom).
        :param text_update_strategy: strategy to update the text field after caption
            generation. Can be one of ['keep_origin', 'rewrite'].
            'keep_origin': keep the original text unchanged.
            'rewrite': rewrite the text field with the generated captions concated by special tokens.
        :param caption_field: the field name to save the generated captions.
        :param legacy_split_by_text_token: Whether to split by special tokens (e.g. <__dj__video>)
            in the text field and read videos in order, or use the 'videos' or 'frames' field directly.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["memory"] = "20GB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]
        super().__init__(*args, **kwargs)

        if keep_candidate_mode not in [
                'random_any', 'similar_one_simhash', 'all'
        ]:
            raise ValueError(
                f'Keep strategy [{keep_candidate_mode}] is not supported. '
                f'Can only be one of '
                f'["random_any", "similar_one_simhash", "all"].')

        if caption_num > 1 and keep_candidate_mode == 'all':
            logger.warning(
                "`keep_candidate_mode` is set to 'all'. This behavior will be deprecated in future versions. "
                "Please consider using 'random_any' or 'similar_one_simhash' instead."
            )
            if not legacy_split_by_text_token:
                raise ValueError("`legacy_split_by_text_token` must be True when `keep_candidate_mode` is 'all'.")

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

        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method '
                f'[{frame_sampling_method}] is not supported. '
                f'Can only be one of ["all_keyframes", "uniform"].')

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.frame_field = frame_field
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        self.text_update_strategy = text_update_strategy
        self.caption_field = caption_field
        self.legacy_split_by_text_token = legacy_split_by_text_token

        if legacy_split_by_text_token:
            logger.warning(
                '`legacy_split_by_text_token` is set to true, '
                'spliting the text field by special tokens '
                '(e.g. <__dj__video>) to read videos in order. '
                'This behavior will be deprecated in future versions. '
                'Please set `legacy_split_by_text_token` to False, '
                'and use the "videos" or "frames" field directly.'
            )

        if not self.caption_field and self.text_update_strategy != 'rewrite':
            raise ValueError(
                'When ``text_update_strategy`` is not "rewrite", '
                '``caption_field`` should be provided to save videos captions.')

        if self.caption_field and self.keep_original_sample:
            raise ValueError('Only one of ``caption_field`` and ``keep_original_sample`` should be set.')

        self.model_key = prepare_model(
            model_type='huggingface',
            pretrained_model_name_or_path=hf_img2seq,
            trust_remote_code=trust_remote_code
        )

    def _gen_caption_from_frames(self, load_video_data, inp, model, processor, prompt):
        video_frames_chunk = []
        if load_video_data:
            video = inp
            # extract frame videos
            if self.frame_sampling_method == 'all_keyframes':
                frames = extract_key_frames(video)
            elif self.frame_sampling_method == 'uniform':
                frames = extract_video_frames_uniformly(
                    video, self.frame_num)
            else:
                frames = []
            video_frames = [frame.to_image() for frame in frames]

        else:
            video_frames = inp
            video_frames = [load_image(frame) for frame in video_frames]

        for frame in video_frames:
            if self.horizontal_flip:
                frame = ImageOps.mirror(frame)
            if self.vertical_flip:
                frame = ImageOps.flip(frame)
            video_frames_chunk.append(frame)

        # construct prompts
        if self.prompt_key and isinstance(prompt, str):
            # check prompt_key is not None, and it's a str
            # in the sample
            prompt_texts = [prompt] * len(video_frames_chunk)
        elif self.prompt and isinstance(self.prompt, str):
            # check prompt is not None, and it's a str
            prompt_texts = [self.prompt] * len(video_frames_chunk)
        else:
            prompt_texts = None

        inputs = processor(
            text=prompt_texts,
            images=video_frames_chunk,
            return_tensors='pt',
        ).to(model.device)

        generated_caption_candidates = []
        with torch.no_grad():
            for i in range(self.caption_num):
                generated_ids = model.generate(**inputs,
                                               max_new_tokens=128,
                                               do_sample=True)
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True)
                generated_caption_candidates.append(generated_text)

        return generated_caption_candidates

    def _process_captions(self, ori_text, video_count, generated_caption_candidates):
        # insert a list of generated captions into the positions of
        # subsequent placeholders in the original string
        new_generated_text_all_videos = [
            [] for _ in range(self.num_newly_generated_samples)
        ]
        # new_generated_text_all_videos is a helper array,
        # element [i][j]
        # denotes the reduced $i$-th result for the $j$-th video

        chunk = ori_text
        # reduce the captions according to given mode video by video
        for j in range(video_count):
            new_generated_text_per_video = self._reduce_captions(
                chunk,
                [
                    captions[j] for captions in
                    generated_caption_candidates
                ],
            )
            assert self.num_newly_generated_samples == len(
                new_generated_text_per_video)
            for i in range(len(new_generated_text_per_video)):
                new_generated_text_all_videos[i].append(
                    new_generated_text_per_video[i])

        if not self.legacy_split_by_text_token:
            return new_generated_text_all_videos

        text_with_only_special_tokens = remove_non_special_tokens(chunk)
        # insert the captions according to given mode
        place_holders = [SpecialTokens.video] * video_count

        updated_videos_text = []
        for i in range(self.num_newly_generated_samples):
            generated_text = insert_texts_after_placeholders(
                original_string=text_with_only_special_tokens,
                placeholders=place_holders,
                new_texts=new_generated_text_all_videos[i],
            )
            updated_videos_text.append(f'{generated_text}{SpecialTokens.eoc}')

        return updated_videos_text

    def _process_single_sample(self, ori_sample, rank=None, context=False):
        # there is no videos or frames in this sample
        if (not self.frame_field or self.frame_field not in ori_sample or not ori_sample.get(self.frame_field)) and \
           (self.video_key not in ori_sample or not ori_sample.get(self.video_key)):
            return []

        # the generated results
        generated_samples = [
            copy.deepcopy(ori_sample)
            for _ in range(self.num_newly_generated_samples)
        ]

        if self.frame_field and self.frame_field in ori_sample:
            load_video_data = False
            num_videos = len(ori_sample[self.frame_field])
        else:
            load_video_data = True
            # load videos
            loaded_video_keys = ori_sample[self.video_key]
            num_videos = len(loaded_video_keys)
            sample, videos = load_data_with_context(
                ori_sample, context, loaded_video_keys, load_video)

        text = ori_sample.get(self.text_key, "")
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        # =============
        if not self.legacy_split_by_text_token:
            if num_videos == 0:
                return generated_samples

            generated_text_candidates = [
                [] for _ in range(self.caption_num)
            ]
            for idx in range(num_videos):

                if load_video_data:
                    video_key = loaded_video_keys[idx]
                    inp = videos[video_key]
                else:
                    frames = ori_sample[self.frame_field][idx]
                    # select frames_num frames from inp frames
                    if self.frame_num >= len(frames):
                        inp = frames
                    else:
                        indices = np.linspace(0, len(frames)-1, self.frame_num, dtype=int)
                        inp = [frames[i] for i in indices]

                raw_generated_text_candidates = self._gen_caption_from_frames(
                    load_video_data, inp, model, processor, ori_sample.get(self.prompt_key, None))

                for i in range(self.caption_num):
                    generated_text = raw_generated_text_candidates[i]
                    generated_text_candidates[i] += [
                        '. '.join([txt.strip() for txt in generated_text])
                    ]

            chunk = text
            captions_list = self._process_captions(chunk, num_videos, generated_text_candidates)

            if self.text_update_strategy == 'keep_origin':
                for i in range(self.num_newly_generated_samples):
                    generated_samples[i][self.caption_field] = captions_list[i]
            elif self.text_update_strategy == 'rewrite':
                for i in range(self.num_newly_generated_samples):
                    generated_samples[i][self.text_key] = ' '.join([SpecialTokens.video + c_i for c_i in captions_list[i]]) + SpecialTokens.eoc

            if not context and load_video_data:
                for vid_key in videos:
                    close_video(videos[vid_key])

            return generated_samples

        # ========== legacy process, will be deprecated in future versions ==========
        for generated_sample in generated_samples:
            generated_sample[self.text_key] = ''

        offset = 0
        # split the original text into chunks by special token
        for chunk in text.split(SpecialTokens.eoc):
            video_count = chunk.count(SpecialTokens.video)

            # no video or no text
            if video_count == 0 or len(chunk.strip()) == 0:
                continue

            # generate candidate caption(s) in batch manner
            generated_text_candidates_single_chunk = [
                [] for _ in range(self.caption_num)
            ]

            index_list = list(range(offset, offset + video_count))
            for idx in index_list:
                if load_video_data:
                    video_key = loaded_video_keys[idx]
                    inp = videos[video_key]
                else:
                    frames = ori_sample[self.frame_field][idx]
                    # select frames_num frames from inp frames
                    if self.frame_num >= len(frames):
                        inp = frames
                    else:
                        indices = np.linspace(0, len(frames)-1, self.frame_num, dtype=int)
                        inp = [frames[i] for i in indices]

                generated_text_candidates = self._gen_caption_from_frames(
                    load_video_data, inp, model, processor, ori_sample.get(self.prompt_key, None))

                for i in range(self.caption_num):
                    generated_text = generated_text_candidates[i]
                    generated_text_candidates_single_chunk[i] += [
                        '. '.join([txt.strip() for txt in generated_text])
                    ]

            captions_list = self._process_captions(
                chunk, video_count, generated_text_candidates_single_chunk
            )

            for i in range(self.num_newly_generated_samples):
                generated_samples[i][self.text_key] += captions_list[i]

            offset += video_count

        if not context and load_video_data:
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
        keys = set()
        for s in samples_after_generation:
            keys.update(s.keys())

        res_samples = {}
        for key in keys:
            res_samples[key] = [s.get(key) for s in samples_after_generation]

        return res_samples
