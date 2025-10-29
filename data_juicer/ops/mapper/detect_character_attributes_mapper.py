import json
import os
import random
from typing import Dict, Optional

from PIL import Image

import data_juicer
from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = "detect_character_attributes_mapper"


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class DetectCharacterAttributesMapper(Mapper):
    """Takes an image, a caption, and main character names as input to extract the characters' attributes."""

    _accelerator = "cuda"

    def __init__(
        self,
        detect_character_locations_mapper_args: Optional[Dict] = {},
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param detect_character_locations_mapper_args: Arguments for detect_character_locations_mapper_args.
            Controls the threshold for locating the main character.
            Default empty dict will use fixed values: default mllm_mapper_args,
            default image_text_matching_filter_args, yoloe_path="yoloe-11l-seg.pt",
            iou_threshold=0.7, matching_score_threshold=0.4,

        """
        super().__init__(*args, **kwargs)

        self.FIXED_ARGS = {}
        self.FIXED_ARGS["detect_character_locations_mapper"] = {
            "mllm_mapper_args": {
                "max_new_tokens": 256,
                "temperature": 0.2,
                "top_p": None,
                "num_beams": 1,
                "hf_model": "llava-hf/llava-v1.6-vicuna-7b-hf",
            },
            "image_text_matching_filter_args": {
                "min_score": 0,
                "max_score": 1.0,
                "hf_blip": "Salesforce/blip-itm-base-coco",
                "num_proc": 1,
            },
            "yoloe_path": "yoloe-11l-seg.pt",
            "iou_threshold": 0.7,
            "matching_score_threshold": 0.4,
        }

        self.detect_character_locations_mapper_args = self._prepare_op_args(
            "detect_character_locations_mapper", detect_character_locations_mapper_args
        )

        self.fused_op_list = [{"detect_character_locations_mapper": self.detect_character_locations_mapper_args}]
        self.fused_ops = load_ops(self.fused_op_list)

        accelerator_methods = set([op.accelerator for op in self.fused_ops])
        if "cuda" in accelerator_methods:
            self.accelerator = "cuda"

        # update num_proc with the min num_proc of all fusible filters
        self.num_proc = min([op.runtime_np() for op in self.fused_ops]) if self.fused_ops else 1

    def _prepare_op_args(self, op_name, args_dict):
        for key in self.FIXED_ARGS[op_name]:
            if key not in args_dict:
                args_dict[key] = self.FIXED_ARGS[op_name][key]
        args_dict["accelerator"] = self.accelerator
        return args_dict

    def process_single(self, samples, rank=None):

        if Fields.meta not in samples:
            samples[Fields.meta] = {}

        detect_location_dataset = data_juicer.core.NestedDataset.from_list(
            [{"main_character_list": samples["main_character_list"], "images": samples["images"]}]
        )

        character_locations = detect_location_dataset.map(
            self.fused_ops[0].process, num_proc=1, with_rank=True
        ).to_list()
        character_locations = character_locations[0][Fields.meta]["main_character_locations_list"]

        character_to_characteristics = {}
        character_to_cls = {}

        for temp_character in samples["main_character_list"]:

            # detect class
            prompt = (
                'Please classify the character "'
                + temp_character
                + "\" into the following categories: ['object', 'animal', 'person', 'text', 'other']. Only reply with the most fitting single category."
            )
            mllm_sample = {"text": prompt, "images": samples["images"]}
            output_text = self.fused_ops[0].fused_ops[0].process(mllm_sample)["text"][0].split("ASSISTANT:")[-1].strip()
            character_to_cls[temp_character] = output_text

            # detect feature
            prompt = (
                'I will provide you with the corresponding description of an image, as follows: "'
                + samples["text"]
                + "\" Please extract all descriptions of the features related to '"
                + temp_character
                + '\' from this text, which may include color, material, action, and other typical features, and compile them into a list of phrase string. Formatted like: ["in a blue shirt", "sitting on a nearby fence", "with flame decals"]. Return only the phrase string list.'
            )
            mllm_sample = {"text": prompt, "images": samples["images"]}
            output_text = self.fused_ops[0].fused_ops[0].process(mllm_sample)["text"][0].split("ASSISTANT:")[-1].strip()
            try:
                character_to_characteristics[temp_character] = json.loads(output_text)
            except json.JSONDecodeError:
                character_to_characteristics[temp_character] = [output_text]

        image = Image.open(samples["images"][0])
        valid_character_in_bbox_dict = {}
        for temp_character_with_bbox_idx, temp_character_with_bbox in enumerate(character_locations):
            crop_img = image.crop(temp_character_with_bbox["bbox"])

            cache_img_name = (
                "temp_"
                + str(random.randint(0, 9999))
                + "_"
                + str(temp_character_with_bbox_idx)
                + samples["images"][0].split("/")[-1]
            )
            cache_img_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE,
                cache_img_name,
            )
            crop_img.save(cache_img_path)

            try:
                temp_character_cls = character_to_cls[temp_character_with_bbox["main_character"]]
            except Exception:
                os.remove(cache_img_path)
                continue

            if "object" in temp_character_cls:
                prompt = (
                    "Please analyze the key characteristics of the main object in this image, specifically the '"
                    + temp_character_with_bbox["main_character"]
                    + "', which may include color, material, shape, and other typical features. Currently identified characteristics include \""
                    + str(temp_character_cls)
                    + '". Please expand this list and respond in an identically formatted phrase string list.'
                )
                mllm_sample = {"text": prompt, "images": [cache_img_path]}
                output_text = (
                    self.fused_ops[0].fused_ops[0].process(mllm_sample)["text"][0].split("ASSISTANT:")[-1].strip()
                )

            elif "animal" in temp_character_cls:
                prompt = (
                    "Please analyze the key characteristics of the primary animal in this image, specifically the '"
                    + temp_character_with_bbox["main_character"]
                    + "', which may include color, action, and other typical features. Currently identified characteristics include \""
                    + str(temp_character_cls)
                    + '". Please expand this list and respond in an identically formatted phrase string list.'
                )
                mllm_sample = {"text": prompt, "images": [cache_img_path]}
                output_text = (
                    self.fused_ops[0].fused_ops[0].process(mllm_sample)["text"][0].split("ASSISTANT:")[-1].strip()
                )

            elif "person" in temp_character_cls:
                prompt = (
                    "Please analyze the key characteristics of the primary person in this image, specifically the '"
                    + temp_character_with_bbox["main_character"]
                    + "', which may include clothing, ages, and other typical features. Currently identified characteristics include \""
                    + str(temp_character_cls)
                    + '". Please expand this list and respond in an identically formatted phrase string list.'
                )
                mllm_sample = {"text": prompt, "images": [cache_img_path]}
                output_text = (
                    self.fused_ops[0].fused_ops[0].process(mllm_sample)["text"][0].split("ASSISTANT:")[-1].strip()
                )

            else:
                prompt = (
                    "Please analyze the key characteristics of the primary character in this image, specifically the '"
                    + temp_character_with_bbox["main_character"]
                    + "'. Currently identified characteristics include \""
                    + str(temp_character_cls)
                    + '". Please expand this list and respond in an identically formatted phrase string list.'
                )
                mllm_sample = {"text": prompt, "images": [cache_img_path]}
                output_text = (
                    self.fused_ops[0].fused_ops[0].process(mllm_sample)["text"][0].split("ASSISTANT:")[-1].strip()
                )

            final_characteristic_list = []
            # filter
            try:
                characteristic_list = json.loads(output_text)
            except json.JSONDecodeError:
                characteristic_list = output_text

            if isinstance(characteristic_list, list):
                if len(characteristic_list) == 1:
                    characteristic_list = characteristic_list[0].replace("_", " ").split(", ")

                try:
                    for temp_characteristic in characteristic_list:

                        prompt = (
                            'Please analyze the main character in this image, specifically the "'
                            + temp_character_with_bbox["main_character"]
                            + '". Is "'
                            + temp_characteristic
                            + "\" one of its features? Only respond with 'yes' if it is a perfect match. Please only respond with 'yes' or 'no'."
                        )
                        mllm_sample = {"text": prompt, "images": [cache_img_path]}
                        output_text = (
                            self.fused_ops[0]
                            .fused_ops[0]
                            .process(mllm_sample)["text"][0]
                            .split("ASSISTANT:")[-1]
                            .strip()
                        )

                        if "yes" in output_text:
                            final_characteristic_list.append(temp_characteristic)
                except Exception:
                    os.remove(cache_img_path)
                    continue
            else:
                try:
                    characteristic_list = output_text.split("\n")
                    if len(characteristic_list) == 1:
                        characteristic_list = characteristic_list[0].replace("_", " ").split(", ")

                    for temp_characteristic in characteristic_list:
                        prompt = (
                            'Please analyze the main character in this image, specifically the "'
                            + temp_character_with_bbox["main_character"]
                            + '". Is "'
                            + temp_characteristic
                            + "\" one of its features? Only respond with 'yes' if it is a perfect match. Please only respond with 'yes' or 'no'."
                        )
                        mllm_sample = {"text": prompt, "images": [cache_img_path]}
                        output_text = (
                            self.fused_ops[0]
                            .fused_ops[0]
                            .process(mllm_sample)["text"][0]
                            .split("ASSISTANT:")[-1]
                            .strip()
                        )

                        if "yes" in output_text:
                            final_characteristic_list.append(temp_characteristic)
                except Exception:
                    os.remove(cache_img_path)
                    continue

            valid_character_in_bbox_dict[temp_character_with_bbox["main_character"]] = {}
            valid_character_in_bbox_dict[temp_character_with_bbox["main_character"]]["bbox"] = temp_character_with_bbox[
                "bbox"
            ]
            valid_character_in_bbox_dict[temp_character_with_bbox["main_character"]][
                "final_characteristic_list"
            ] = final_characteristic_list

            os.remove(cache_img_path)

        new_character_list = []
        for temp_character in samples["main_character_list"]:
            temp_character_json = {}
            temp_character_json["main_character"] = temp_character
            if temp_character in valid_character_in_bbox_dict:
                temp_character_json["bbox"] = valid_character_in_bbox_dict[temp_character]["bbox"]

                if len(valid_character_in_bbox_dict[temp_character]["final_characteristic_list"]) == 0:
                    temp_character_json["characteristic_list"] = character_to_characteristics[temp_character]
                else:
                    temp_character_json["characteristic_list"] = valid_character_in_bbox_dict[temp_character][
                        "final_characteristic_list"
                    ]

            else:
                temp_character_json["bbox"] = []
                temp_character_json["characteristic_list"] = character_to_characteristics[temp_character]

            new_character_list.append(temp_character_json)

        samples[Fields.meta]["main_character_attributes_list"] = new_character_list

        return samples
