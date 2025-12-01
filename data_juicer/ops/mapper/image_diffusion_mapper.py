import copy
import os
from typing import Optional

from PIL import Image
from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    load_data_with_context,
    load_image,
    remove_special_tokens,
)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = "image_diffusion_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageDiffusionMapper(Mapper):
    """Generate images using a diffusion model based on provided captions.

    This operator uses a Hugging Face diffusion model to generate images from given
    captions. It supports different modes for retaining generated samples, including random
    selection, similarity-based selection, and retaining all. The operator can also generate
    captions if none are provided, using a Hugging Face image-to-sequence model. The
    strength parameter controls the extent of transformation from the reference image, and
    the guidance scale influences how closely the generated images match the text prompt.
    Generated images can be saved in a specified directory or the same directory as the
    input files. This is a batched operation, processing multiple samples at once and
    producing a specified number of augmented images per sample."""

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        hf_diffusion: str = "CompVis/stable-diffusion-v1-4",
        trust_remote_code: bool = False,
        torch_dtype: str = "fp32",
        revision: str = "main",
        strength: Annotated[float, Field(ge=0, le=1)] = 0.8,
        guidance_scale: float = 7.5,
        aug_num: PositiveInt = 1,
        keep_original_sample: bool = True,
        caption_key: Optional[str] = None,
        hf_img2seq: str = "Salesforce/blip2-opt-2.7b",
        save_dir: str = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_diffusion: diffusion model name on huggingface to generate
            the image.
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param torch_dtype: the floating point type used to load the diffusion
            model. Can be one of ['fp32', 'fp16', 'bf16']
        :param revision: The specific model version to use. It can be a
            branch name, a tag name, a commit id, or any identifier allowed
            by Git.
        :param strength: Indicates extent to transform the reference image.
            Must be between 0 and 1. image is used as a starting point and
            more noise is added the higher the strength. The number of
            denoising steps depends on the amount of noise initially added.
            When strength is 1, added noise is maximum and the denoising
            process runs for the full number of iterations specified in
            num_inference_steps. A value of 1 essentially ignores image.
        :param guidance_scale: A higher guidance scale value encourages the
            model to generate images closely linked to the text prompt at the
            expense of lower image quality. Guidance scale is enabled when
            guidance_scale > 1.
        :param aug_num: The image number to be produced by stable-diffusion
            model.
        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only generated captions in the
            final datasets and the original captions will be removed. It's True
            by default.
        :param caption_key: the key name of fields in samples to store captions
            for each images. It can be a string if there is only one image in
            each sample. Otherwise, it should be a list. If it's none,
            ImageDiffusionMapper will produce captions for each images.
        :param hf_img2seq: model name on huggingface to generate caption if
            caption_key is None.
        :param save_dir: The directory where generated image files will be stored.
            If not specified, outputs will be saved in the same directory as their corresponding input files.
            This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable.
        """
        kwargs["mem_required"] = "8GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self._init_parameters.pop("save_dir", None)
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.aug_num = aug_num
        self.keep_original_sample = keep_original_sample
        self.caption_key = caption_key
        self.prompt = "A photo of a "
        self.save_dir = save_dir
        if not self.caption_key:
            from .image_captioning_mapper import ImageCaptioningMapper

            self.op_generate_caption = ImageCaptioningMapper(
                hf_img2seq=hf_img2seq, keep_original_sample=False, prompt=self.prompt
            )
        self.model_key = prepare_model(
            model_type="diffusion",
            pretrained_model_name_or_path=hf_diffusion,
            diffusion_type="image2image",
            torch_dtype=torch_dtype,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _real_guidance(self, caption: str, image: Image.Image, rank=None):
        canvas = image.resize((512, 512), Image.BILINEAR)
        prompt = caption

        diffusion_model = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())

        kwargs = dict(image=canvas, prompt=[prompt], strength=self.strength, guidance_scale=self.guidance_scale)

        has_nsfw_concept = True
        while has_nsfw_concept:
            outputs = diffusion_model(**kwargs)

            has_nsfw_concept = diffusion_model.safety_checker is not None and outputs.nsfw_content_detected[0]

        canvas = outputs.images[0].resize(image.size, Image.BILINEAR)

        return canvas

    def _process_single_sample(self, ori_sample, rank=None, context=False):
        """
        :param ori_sample: a single data sample before applying generation
        :return: batched results after generation
        """
        # there is no image in this sample
        if self.image_key not in ori_sample or not ori_sample[self.image_key]:
            return []

        # load images
        loaded_image_keys = ori_sample[self.image_key]
        ori_sample, images = load_data_with_context(
            ori_sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # load captions
        if self.caption_key:
            captions = ori_sample[self.caption_key]
            if not isinstance(captions, list):
                # one caption for all images
                captions = [captions] * len(images)
            else:
                assert len(captions) == len(images), "The num of captions must match the num of images."
            captions = [remove_special_tokens(c) for c in captions]
        else:
            caption_samples = {
                self.text_key: [SpecialTokens.image] * len(images),
                self.image_key: [[k] for k in loaded_image_keys],
            }
            caption_samples = self.op_generate_caption.process(caption_samples, rank=rank)
            captions = caption_samples[self.text_key]
            captions = [self.prompt + remove_special_tokens(c) for c in captions]

        # the generated results
        generated_samples = [copy.deepcopy(ori_sample) for _ in range(self.aug_num)]

        for aug_id in range(self.aug_num):
            diffusion_image_keys = []
            for index, value in enumerate(loaded_image_keys):
                related_parameters = self.add_parameters(self._init_parameters, caption=captions[index])
                diffusion_image_key = transfer_filename(value, OP_NAME, self.save_dir, **related_parameters)
                diffusion_image_keys.append(diffusion_image_key)
                if diffusion_image_key != value:
                    if not os.path.exists(diffusion_image_key) or diffusion_image_key not in images:
                        diffusion_image = self._real_guidance(captions[index], images[value], rank=rank)
                        images[diffusion_image_key] = diffusion_image
                        diffusion_image.save(diffusion_image_key)
                        if context:
                            generated_samples[aug_id][Fields.context][diffusion_image_key] = diffusion_image
                else:
                    diffusion_image = self._real_guidance(captions[index], images[value], rank=rank)
                    images[diffusion_image_key] = diffusion_image
                    if context:
                        generated_samples[aug_id][Fields.context][diffusion_image_key] = diffusion_image
                if self.image_bytes_key in generated_samples[aug_id] and index < len(
                    generated_samples[aug_id][self.image_bytes_key]
                ):
                    generated_samples[aug_id][self.image_bytes_key][index] = images[diffusion_image_key].tobytes()
            generated_samples[aug_id][self.image_key] = diffusion_image_keys

        return generated_samples

    def process_batched(self, samples, rank=None, context=False):
        """
        Note:
            This is a batched_OP, whose the input and output type are
            both list. Suppose there are $N$ input sample list with batch
            size as $b$, and denote aug_num as $M$.
            the number of total samples after generation is  $(1+M)Nb$.

        :param samples:
        :return:
        """
        # reconstruct samples from "dict of lists" to "list of dicts"
        reconstructed_samples = []
        for i in range(len(samples[self.text_key])):
            reconstructed_samples.append({key: samples[key][i] for key in samples})

        # do generation for each sample within the batch
        samples_after_generation = []
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
