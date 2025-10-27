import ast
import json
from typing import Dict, Optional

import numpy as np
from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    image_path_to_base64,
    load_data_with_context,
    load_image,
)
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

from ..base_op import OPERATORS, TAGGING_OPS, Mapper

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")

OP_NAME = "image_tagging_vlm_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ImageTaggingVLMMapper(Mapper):
    """Mapper to generates image tags.
    This operator generates tags based on the content of given images.
    The tags are generated using a vlm model and stored in the specified field name.
    If the tags are already present in the sample, the operator skips processing.
    """

    DEFAULT_SYSTEM_PROMPT = """
Generate comprehensive and specific descriptive tags for the provided image(s) following these rules:
1. Tags should be concise English phrases (nouns or gerunds)
2. Use lowercase and hyphenate multi-word tags
3. Include objects, actions, colors, materials, styles, emotions, and context
4. Prioritize prominent and distinctive elements
5. Output exactly 5-10 most relevant tags
6. Format strictly as: {"tags": ["tag1", "tag2", ...]}

Example valid responses:
{"tags": ["red-apple", "wooden-table", "natural-lighting", "food-photography", "fresh-fruit"]}
{"tags": ["mountain-landscape", "snowy-peaks", "sunset-glow", "alpine-lake", "conifer-forest"]}
"""
    DEFAULT_INPUT_TEMPLATE = """
Analyze both the provided image and its associated text description (if available) to generate comprehensive tags.
Text description: {text}
Verify text relevance before combining with visual elements. If text is missing or irrelevant, generate tags based solely on the image.
"""

    _accelerator = "cuda"

    def __init__(
        self,
        api_or_hf_model: str = "Qwen2.5-VL-7B-Instruct",
        is_api_model: bool = False,
        *,
        tag_field_name: str = MetaKeys.image_tags,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        model_params: Dict = {},
        sampling_params: Dict = {},
        try_num: PositiveInt = 3,
        **kwargs,
    ):
        """
        Initialization method.

        :param api_or_hf_model: API model name or HF model name.
        :param is_api_model: Whether the model is an API model.
            If true, use openai api to generate tags, otherwise use vllm.
        :param tag_field_name: the field name to store the tags. It's
            "image_tags" in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the task.
        :param input_template: Template for building the model input.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)
        self.is_api_model = is_api_model

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.tag_field_name = tag_field_name
        self.try_num = try_num

        sampling_params = update_sampling_params(sampling_params, api_or_hf_model, not self.is_api_model)

        if self.is_api_model:
            self.sampling_params = sampling_params

            self.model_key = prepare_model(
                model_type="api",
                model=api_or_hf_model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )
        else:
            assert torch.cuda.device_count() >= 1, "must be executed in CUDA"
            # cannot initialize vllm replicas on different GPUs
            self.num_proc = 1
            if model_params.get("tensor_parallel_size") is None:
                tensor_parallel_size = torch.cuda.device_count()
                logger.info(f"Set tensor_parallel_size to {tensor_parallel_size} for vllm.")
                model_params["tensor_parallel_size"] = tensor_parallel_size
            self.model_key = prepare_model(
                model_type="vllm", pretrained_model_name_or_path=api_or_hf_model, **model_params
            )
            self.sampling_params = vllm.SamplingParams(**sampling_params)

    def parse_output(self, raw_output):
        json_str = raw_output.strip()

        for pattern in ["```json", "```", "JSON:", "Response:"]:
            json_str = json_str.replace(pattern, "")

        try:
            result = json.loads(json_str, strict=False)
        except json.JSONDecodeError:
            try:
                # handle single quotation situations
                json_str = json_str.replace("'", '"')
                result = json.loads(json_str, strict=False)
            except Exception:
                try:
                    # ast as alternative option
                    result = ast.literal_eval(json_str)
                except Exception:
                    logger.error(f"Failed to parse model output: {raw_output}")
                    return []

        if not isinstance(result, dict):
            return []

        # support possible variations of key names
        tags = result.get("tags", result.get("tag", []))

        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",")]

        # dedup and filter
        valid_tags = []
        for tag in tags:
            if isinstance(tag, str):
                # normalize format: lowercase, replace spaces, limit length
                tag = tag.lower().replace(" ", "-")[:30]
                valid_tags.append(tag)

        return list(set(valid_tags))[:10]

    def process_single(self, sample, rank=None, context=False):
        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.meta][self.tag_field_name] = np.array([[]], dtype=np.str_)
            return sample

        # load videos
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context, loaded_image_keys, load_image)

        if self.is_api_model:
            model = get_model(self.model_key, rank, self.use_cuda())
        else:
            model, _ = get_model(self.model_key, rank, self.use_cuda())

        tags_list = []
        for img in images:
            input_prompt = self.input_template.format(text=sample.get(self.text_key, ""))
            user_content = [
                {"type": "text", "text": input_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_path_to_base64(img)}"},
                },
            ]
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": user_content,
                }
            )

            if self.is_api_model:
                for _ in range(self.try_num):
                    try:
                        client = get_model(self.model_key, rank=rank)
                        output = client(messages, **self.sampling_params)
                    except Exception as e:
                        logger.warning(f"Exception: {e}")
            else:
                response = model.chat(messages, self.sampling_params)
                output = response[0].outputs[0].text

            try:
                tags = self.parse_output(output)
            except Exception as e:
                logger.warning(f"Error parsing output: {e}")
                tags = []
            tags_list.append(tags)

        tags_list = np.array(tags_list, dtype=object)
        sample[Fields.meta][self.tag_field_name] = tags_list
        return sample
