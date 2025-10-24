import json
import os
import random
from typing import Dict, Optional

from PIL import Image

import data_juicer
from data_juicer.ops.load import load_ops
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.model_utils import check_model

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = "detect_character_locations_mapper"

ultralytics = LazyLoader("ultralytics")


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class DetectCharacterLocationsMapper(Mapper):
    """Given an image and a list of main character names, extract the bounding boxes for each present character."""

    _accelerator = "cuda"

    def __init__(
        self,
        mllm_mapper_args: Optional[Dict] = {},
        image_text_matching_filter_args: Optional[Dict] = {},
        yoloe_path="yoloe-11l-seg.pt",
        iou_threshold=0.7,
        matching_score_threshold=0.4,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param mllm_mapper_args: Arguments for multimodal language model mapper.
            Controls the generation of captions for bounding box regions. Default empty dict
            will use fixed values: max_new_tokens=256, temperature=0.2, top_p=None,
            num_beams=1, hf_model="llava-hf/llava-v1.6-vicuna-7b-hf".
        :param image_text_matching_filter_args: Arguments for image-text matching filter.
            Controls the matching between cropped image regions and text descriptions.
            Default empty dict will use fixed values: min_score=0.1, max_score=1.0,
            hf_blip="Salesforce/blip-itm-base-coco", num_proc=1.
        :param yoloe_path: The path to the YOLOE model.
        :param iou_threshold: We consider two bounding boxes from different models to be overlapping
            when their IOU score is higher than the iou_threshold.
        :param matching_score_threshold: If the matching score between the cropped image and the
            character's name exceeds the matching_score_threshold, they are considered a match.

        """
        super().__init__(*args, **kwargs)

        # Requires the weights for YOLOE and mobileclip_blt.
        self.yoloe_model = ultralytics.YOLO(check_model(yoloe_path))

        self.FIXED_ARGS = {}
        self.FIXED_ARGS["mllm_mapper"] = {
            "max_new_tokens": 256,
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "hf_model": "llava-hf/llava-v1.6-vicuna-7b-hf",
        }
        self.FIXED_ARGS["image_text_matching_filter"] = {
            "min_score": 0,
            "max_score": 1.0,
            "hf_blip": "Salesforce/blip-itm-base-coco",
            "num_proc": 1,
        }

        self.mllm_mapper_args = self._prepare_op_args("mllm_mapper", mllm_mapper_args)
        self.image_text_matching_filter_args = self._prepare_op_args(
            "image_text_matching_filter", image_text_matching_filter_args
        )

        self.fused_op_list = [
            {"mllm_mapper": self.mllm_mapper_args},
            {"image_text_matching_filter": self.image_text_matching_filter_args},
        ]
        self.fused_ops = load_ops(self.fused_op_list)

        accelerator_methods = set([op.accelerator for op in self.fused_ops])
        if "cuda" in accelerator_methods:
            self.accelerator = "cuda"

        # update num_proc with the min num_proc of all fusible filters
        self.num_proc = min([op.runtime_np() for op in self.fused_ops]) if self.fused_ops else 1
        self.iou_threshold = iou_threshold
        self.matching_score_threshold = matching_score_threshold

    def _prepare_op_args(self, op_name, args_dict):
        for key in self.FIXED_ARGS[op_name]:
            if key not in args_dict:
                args_dict[key] = self.FIXED_ARGS[op_name][key]
        args_dict["accelerator"] = self.accelerator
        return args_dict

    def iou_cal(self, bbox1, bbox2):

        max_x1 = max(bbox1[0], bbox2[0])
        max_y1 = max(bbox1[1], bbox2[1])

        min_x2 = min(bbox1[2], bbox2[2])
        min_y2 = min(bbox1[3], bbox2[3])

        if min_x2 - max_x1 < 0 or min_y2 - max_y1 < 0:
            return 0, 0, 0

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        intersection_area = (min_x2 - max_x1) * (min_y2 - max_y1)
        union_area = area1 + area2 - intersection_area
        iou = intersection_area / union_area

        return iou, area1, area2

    def process_single(self, samples, rank=None):

        if Fields.meta not in samples:
            samples[Fields.meta] = {}

        now_image = Image.open(samples["images"][0])
        self.yoloe_model.set_classes(
            samples["main_character_list"], self.yoloe_model.get_text_pe(samples["main_character_list"])
        )
        results = self.yoloe_model.predict(samples["images"][0], verbose=False)
        yoloe_bboxes = results[0].boxes.xyxy.tolist()
        bboxes_cls = results[0].boxes.cls.tolist()

        valid_main_character = []
        seen = set()
        for temp_bbox_idx in range(len(yoloe_bboxes)):
            if bboxes_cls[temp_bbox_idx] in seen:
                continue
            seen.add(bboxes_cls[temp_bbox_idx])
            temp_bbox_json = {}
            temp_bbox_json["main_character"] = samples["main_character_list"][int(bboxes_cls[temp_bbox_idx])]
            temp_bbox_json["yoloe_bbox"] = [
                round(yoloe_bboxes[temp_bbox_idx][0]),
                round(yoloe_bboxes[temp_bbox_idx][1]),
                round(yoloe_bboxes[temp_bbox_idx][2]),
                round(yoloe_bboxes[temp_bbox_idx][3]),
            ]
            valid_main_character.append(temp_bbox_json)

        final_bboxes = []
        for temp_character in valid_main_character:

            prompt = (
                'Please only provide the bounding box coordinate of the region "'
                + temp_character["main_character"]
                + '" describes.'
            )
            mllm_sample = {"text": prompt, "images": samples["images"]}
            output_text = self.fused_ops[0].process(mllm_sample)["text"][0].split("ASSISTANT:")[-1].strip()
            try:
                output_text = output_text.replace("json", "").replace("```", "")
                output_data = json.loads(output_text)
                if (
                    isinstance(output_data, list)
                    and len(output_data) == 4
                    and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in output_data)
                ):
                    temp_character["llm_bbox"] = [
                        int(output_data[0] * now_image.size[0]),
                        int(output_data[1] * now_image.size[1]),
                        int(output_data[2] * now_image.size[0]),
                        int(output_data[3] * now_image.size[1]),
                    ]
                final_bboxes.append(temp_character)
            except (json.JSONDecodeError, TypeError):
                continue

        final_filterd_character = []
        for temp_character_idx, temp_character in enumerate(final_bboxes):
            temp_iou, area1, area2 = self.iou_cal(temp_character["yoloe_bbox"], temp_character["llm_bbox"])

            if temp_iou > self.iou_threshold:
                if area1 > area2:
                    temp_json = {}
                    temp_json["main_character"] = temp_character["main_character"]
                    temp_json["bbox"] = temp_character["yoloe_bbox"]
                    final_filterd_character.append(temp_json)
                else:
                    temp_json = {}
                    temp_json["main_character"] = temp_character["main_character"]
                    temp_json["bbox"] = temp_character["llm_bbox"]
                    final_filterd_character.append(temp_json)
            else:
                yoloe_bbox_crop_img = now_image.crop(temp_character["yoloe_bbox"])
                llm_bbox_crop_img = now_image.crop(temp_character["llm_bbox"])

                random_num = str(random.random()).split(".")[-1]
                valid_img_name = samples["images"][0].split("/")[-1].split(".")[-2]

                temp_image_path_yoloe = os.path.join(
                    DATA_JUICER_ASSETS_CACHE,
                    f"cropped_images_{valid_img_name}_{random_num}_" f"<yoloe>.jpg",
                )
                yoloe_bbox_crop_img.save(temp_image_path_yoloe)

                temp_image_path_llm = os.path.join(
                    DATA_JUICER_ASSETS_CACHE,
                    f"cropped_images_{valid_img_name}_{random_num}_" f"<llm>.jpg",
                )
                llm_bbox_crop_img.save(temp_image_path_llm)

                crop_samples = [
                    {"text": SpecialTokens.image + temp_character["main_character"], "images": [temp_image_path_yoloe]},
                    {"text": SpecialTokens.image + temp_character["main_character"], "images": [temp_image_path_llm]},
                ]

                crop_samples = data_juicer.core.NestedDataset.from_list(crop_samples)
                if Fields.stats not in crop_samples.features:
                    crop_samples = crop_samples.add_column(name=Fields.stats, column=[{}] * crop_samples.num_rows)

                crop_image_filtered = crop_samples.map(
                    self.fused_ops[1].compute_stats,
                    num_proc=self.image_text_matching_filter_args["num_proc"],
                    with_rank=True,
                )
                os.remove(temp_image_path_yoloe)
                os.remove(temp_image_path_llm)

                if (
                    crop_image_filtered[0][Fields.stats][StatsKeys.image_text_matching_score][0]
                    < self.matching_score_threshold
                    and crop_image_filtered[1][Fields.stats][StatsKeys.image_text_matching_score][0]
                    < self.matching_score_threshold
                ):
                    continue

                if (
                    crop_image_filtered[0][Fields.stats][StatsKeys.image_text_matching_score][0]
                    > crop_image_filtered[1][Fields.stats][StatsKeys.image_text_matching_score][0]
                ):
                    temp_json = {}
                    temp_json["main_character"] = temp_character["main_character"]
                    temp_json["bbox"] = temp_character["yoloe_bbox"]
                    final_filterd_character.append(temp_json)
                else:
                    temp_json = {}
                    temp_json["main_character"] = temp_character["main_character"]
                    temp_json["bbox"] = temp_character["llm_bbox"]
                    final_filterd_character.append(temp_json)

        samples[Fields.meta]["main_character_locations_list"] = final_filterd_character

        return samples
