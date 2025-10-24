import json
from typing import Dict, Optional

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from data_juicer.ops.load import load_ops
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.constant import Fields

OP_NAME = "detect_main_character_mapper"


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class DetectMainCharacterMapper(Mapper):
    """Extract all main character names based on the given image and its caption."""

    _accelerator = "cuda"

    def __init__(
        self,
        mllm_mapper_args: Optional[Dict] = {},
        filter_min_character_num: int = 0,
        *args,
        **kwargs,
    ):
        """Initialization.

        :param mllm_mapper_args: Arguments for multimodal language model mapper.
            Controls the generation of captions for bounding box regions. Default empty dict
            will use fixed values: max_new_tokens=256, temperature=0.2, top_p=None,
            num_beams=1, hf_model="llava-hf/llava-v1.6-vicuna-7b-hf".
        :param filter_min_character_num: Filters out samples where the number of main characters
            in the image is less than this threshold.
        """

        super().__init__(*args, **kwargs)

        self.FIXED_ARGS = {}
        self.FIXED_ARGS["mllm_mapper"] = {
            "max_new_tokens": 256,
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "hf_model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        }

        self.mllm_mapper_args = self._prepare_op_args("mllm_mapper", mllm_mapper_args)
        self.mllm_mapper = load_ops([{"mllm_mapper": self.mllm_mapper_args}])

        accelerator_methods = set([self.mllm_mapper[0].accelerator])
        if "cuda" in accelerator_methods:
            self.accelerator = "cuda"

        self.num_proc = self.mllm_mapper[0].runtime_np()
        self.filter_min_character_num = filter_min_character_num

    def _prepare_op_args(self, op_name, args_dict):
        for key in self.FIXED_ARGS[op_name]:
            if key not in args_dict:
                args_dict[key] = self.FIXED_ARGS[op_name][key]
        args_dict["accelerator"] = self.accelerator
        return args_dict

    def process_single(self, samples, rank=None):

        if Fields.meta not in samples:
            samples[Fields.meta] = {}

        prompt = (
            'I will provide you with an image and its corresponding description. You need to identify and count the main characters in the image (e.g., key people, key animals, key objects). The output should only be in JSON format, including the number of main characters and a list of their descriptions, as shown in the example: {"count": 3, "main_character": ["man in a blue shirt", "black cat sitting on a nearby fence", "skateboard with flame decals"]}. Below, I will provide the description of the corresponding image: "'
            + samples["text"]
            + '" Please identify the main characters along with their key, distinct characteristics, and output the result in JSON format.'
        )
        mllm_sample = {"text": prompt, "images": samples["images"]}

        result = self.mllm_mapper[0].process(mllm_sample)["text"][0].split("ASSISTANT:")[-1].strip()

        samples[Fields.meta]["main_character_list"] = []
        try:
            result = result.replace("\n", "").replace("\\", "")
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end > start:
                result_json = result[start:end]
                data = json.loads(result_json)
                if data.get("count", 0) >= self.filter_min_character_num:
                    samples[Fields.meta]["main_character_list"] = data.get("main_character", [])
                else:
                    samples[Fields.meta]["main_character_list"] = []
            else:
                samples[Fields.meta]["main_character_list"] = []
        except (json.JSONDecodeError, KeyError):
            samples[Fields.meta]["main_character_list"] = []

        return samples
