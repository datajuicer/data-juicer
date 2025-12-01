import logging
import os
import random
import time

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.file_utils import transfer_data_dir
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

diffusers = LazyLoader("diffusers")
torch = LazyLoader("torch")
p2p_pipeline = LazyLoader("data_juicer.ops.common.prompt2prompt_pipeline", auto_install=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OP_NAME = "sdxl_prompt2prompt_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class SDXLPrompt2PromptMapper(Mapper):
    """Generates pairs of similar images using the SDXL model.

    This operator uses a Hugging Face diffusion model to generate image pairs based on two
    text prompts. The quality and similarity of the generated images are controlled by
    parameters such as `num_inference_steps` and `guidance_scale`. The first and second text
    prompts are specified using `text_key` and `text_key_second`, respectively. The
    generated images are saved in the specified `output_dir` with unique filenames. The
    operator requires both text keys to be set for processing."""

    _accelerator = "cuda"

    def __init__(
        self,
        hf_diffusion: str = "stabilityai/stable-diffusion-xl-base-1.0",
        trust_remote_code=False,
        torch_dtype: str = "fp32",
        num_inference_steps: float = 50,
        guidance_scale: float = 7.5,
        text_key=None,
        text_key_second=None,
        output_dir=DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_diffusion: diffusion model name on huggingface to generate
            the image.
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param torch_dtype: the floating point type used to load the diffusion
            model.
        :param num_inference_steps: The larger the value, the better the
        image generation quality; however, this also increases the time
        required for generation.
        :param guidance_scale: A higher guidance scale value encourages the
            model to generate images closely linked to the text prompt at the
            expense of lower image quality. Guidance scale is enabled when
        :param text_key: the key name used to store the first caption
            in the caption pair.
        :param text_key_second: the key name used to store the second caption
            in the caption pair.
        :param output_dir: the storage location of the generated images.
        """
        kwargs["mem_required"] = "38GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.hf_diffusion = hf_diffusion
        self.torch_dtype = torch_dtype
        self.model_key = prepare_model(
            model_type="sdxl-prompt-to-prompt",
            pretrained_model_name_or_path=hf_diffusion,
            pipe_func=p2p_pipeline.Prompt2PromptPipeline,
            torch_dtype=torch_dtype,
        )
        self.text_key_second = text_key_second
        self.output_dir = output_dir

        if text_key is not None:
            self.text_key = text_key

    def process_single(self, sample, rank=None, context=False):
        if self.text_key_second is None:
            logger.error(
                "This OP (sdxl_prompt2prompt_mapper) requires \
                processing multiple fields, and you need to specify \
                valid `text_key_second`"
            )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        random_num = str(random.randint(0, 9999))
        t1 = time.localtime()
        t2 = time.strftime("%Y-%m-%d-%H-%M-%S", t1)

        model = get_model(model_key=self.model_key, rank=rank, use_cuda=self.use_cuda())

        seed = random.randint(0, 9999)
        g_cpu = torch.Generator().manual_seed(seed)

        cross_attention_kwargs = {
            "edit_type": "refine",
            "n_self_replace": 0.4,
            "n_cross_replace": {"default_": 1.0, "confetti": 0.8},
        }

        with torch.no_grad():
            prompts = [sample[self.text_key], sample[self.text_key_second]]
            image = model(
                prompts,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=g_cpu,
            )

            new_output_dir = transfer_data_dir(self.output_dir, OP_NAME)
            for idx, img in enumerate(image["images"]):
                img_id = str(idx + 1)
                image_name = "image_pair_" + t2 + "_" + random_num + "_" + img_id + ".jpg"
                abs_image_path = os.path.join(new_output_dir, image_name)
                img.save(abs_image_path)
                sample["image_path" + img_id] = abs_image_path

        return sample
