import os
import random
from copy import deepcopy
from typing import Dict, Optional

import numpy as np

import data_juicer
from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from data_juicer.ops.load import load_ops
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader

cv2 = LazyLoader("cv2", "opencv-python")

OP_NAME = "imgdiff_difference_caption_generator_mapper"


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class Difference_Caption_Generator_Mapper(Mapper):
    """Generates difference captions for bounding box regions in two images.

    This operator processes pairs of images and generates captions for the differences in
    their bounding box regions. It uses a multi-step process:
    - Describes the content of each bounding box region using a Hugging Face model.
    - Crops the bounding box regions from both images.
    - Checks if the cropped regions match the generated captions.
    - Determines if there are differences between the two captions.
    - Marks the difference area with a red box.
    - Generates difference captions for the marked areas.
    - The key metric is the similarity score between the captions, computed using a CLIP
      model.
    - If no valid bounding boxes or differences are found, it returns empty captions and
      zeroed bounding boxes.
    - Uses 'cuda' as the accelerator if any of the fused operations support it.
    - Caches temporary images during processing and clears them afterward."""

    _accelerator = "cuda"

    def __init__(
        self,
        mllm_mapper_args: Optional[Dict] = {},
        image_text_matching_filter_args: Optional[Dict] = {},
        text_pair_similarity_filter_args: Optional[Dict] = {},
        *args,
        **kwargs,
    ):
        """Initialization.

        :param mllm_mapper_args: Arguments for multimodal language model mapper.
            Controls the generation of captions for bounding box regions. Default empty dict
            will use fixed values: max_new_tokens=256, temperature=0.2, top_p=None,
            num_beams=1, hf_model="llava-hf/llava-v1.6-vicuna-7b-hf".
        :param image_text_matching_filter_args: Arguments for image-text matching filter.
            Controls the matching between cropped regions and generated captions.
            Default empty dict will use fixed values: min_score=0.1, max_score=1.0,
            hf_blip="Salesforce/blip-itm-base-coco", num_proc=1.
        :param text_pair_similarity_filter_args: Arguments for text pair similarity filter.
            Controls the similarity comparison between caption pairs. Default empty dict
            will use fixed values: min_score=0.1, max_score=1.0,
            hf_clip="openai/clip-vit-base-patch32", text_key_second="target_text", num_proc=1.
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
        self.FIXED_ARGS["image_text_matching_filter"] = {
            "min_score": 0.1,
            "max_score": 1.0,
            "hf_blip": "Salesforce/blip-itm-base-coco",
            "num_proc": 1,
        }
        self.FIXED_ARGS["text_pair_similarity_filter"] = {
            "min_score": 0.1,
            "max_score": 1.0,
            "hf_clip": "openai/clip-vit-base-patch32",
            "text_key_second": "target_text",
            "num_proc": 1,
        }

        self.mllm_mapper_args = self._prepare_op_args("mllm_mapper", mllm_mapper_args)
        self.image_text_matching_filter_args = self._prepare_op_args(
            "image_text_matching_filter", image_text_matching_filter_args
        )
        self.text_pair_similarity_filter_args = self._prepare_op_args(
            "text_pair_similarity_filter", text_pair_similarity_filter_args
        )

        self.fused_op_list = [
            {"mllm_mapper": self.mllm_mapper_args},
            {"image_text_matching_filter": self.image_text_matching_filter_args},
            {"text_pair_similarity_filter": self.text_pair_similarity_filter_args},
        ]

        self.fused_ops = load_ops(self.fused_op_list)
        self._name = "Difference_Caption_Generator_Mapper:(%s)" % ",".join([op._name for op in self.fused_ops])
        # set accelerator to 'cuda' if there exists any ops whose accelerator
        # is 'cuda'
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
        random_num = str(random.random()).split(".")[-1]
        if not os.path.exists(DATA_JUICER_ASSETS_CACHE):
            os.makedirs(DATA_JUICER_ASSETS_CACHE, exist_ok=True)
        cache_image_list = []

        if (
            len(samples[Fields.meta][MetaKeys.bbox_tag]) == 1
            and np.sum(samples[Fields.meta][MetaKeys.bbox_tag][0]) == 0
        ):
            for temp_image_path in cache_image_list:
                os.remove(temp_image_path)
            return {
                Fields.meta: {
                    "region_caption1": [""],
                    "region_caption2": [""],
                    MetaKeys.bbox_tag: np.zeros((1, 4), dtype=np.float32),
                    "bbox_difference_captions": [""],
                }
            }

        # fused_ops 1.mllm_mapper 2.image_text_matching_filter 3.text_pair_similarity_filter
        # keys of sample: "image_path1", "image_path2", Fields.meta[MetaKeys.bbox_tag]
        image_array1 = cv2.imread(samples["image_path1"])
        image_array2 = cv2.imread(samples["image_path2"])
        image_array2 = cv2.resize(image_array2, (image_array1.shape[1], image_array1.shape[0]))

        # Step1: describe the content of each bounding box region.
        text_mllm_samples = []
        text_mllm_samples_bboxes = []
        for temp_bbox in samples[Fields.meta][MetaKeys.bbox_tag]:
            if temp_bbox[2] * temp_bbox[3] > (image_array1.shape[0] * image_array2.shape[1]) / 10:
                continue

            temp_bbox_x1 = str(round(float((temp_bbox[0] - temp_bbox[2] / 2) / image_array1.shape[1]), 2))
            temp_bbox_y1 = str(round(float((temp_bbox[1] - temp_bbox[3] / 2) / image_array1.shape[0]), 2))
            temp_bbox_x2 = str(round(float((temp_bbox[0] + temp_bbox[2] / 2) / image_array1.shape[1]), 2))
            temp_bbox_y2 = str(round(float((temp_bbox[1] + temp_bbox[3] / 2) / image_array1.shape[0]), 2))

            while len(temp_bbox_x1) < 4:
                temp_bbox_x1 = temp_bbox_x1 + "0"
            while len(temp_bbox_y1) < 4:
                temp_bbox_y1 = temp_bbox_y1 + "0"
            while len(temp_bbox_x2) < 4:
                temp_bbox_x2 = temp_bbox_x2 + "0"
            while len(temp_bbox_y2) < 4:
                temp_bbox_y2 = temp_bbox_y2 + "0"

            str_bbox = "[" + temp_bbox_x1 + ", " + temp_bbox_y1 + ", " + temp_bbox_x2 + ", " + temp_bbox_y2 + "]"

            text_key = "Please provide a clear description for this region: " + str_bbox + "."
            image_key = [samples["image_path1"], samples["image_path2"]]
            text_mllm_samples.append({"text": text_key, "images": image_key})
            text_mllm_samples_bboxes.append(temp_bbox)

        caption_pairs = []
        for temp_bbox, sample in zip(text_mllm_samples_bboxes, text_mllm_samples):
            result = self.fused_ops[0].process(sample)
            temp_pairs = {}
            temp_pairs["bboxes"] = temp_bbox
            temp_pairs["caption1"] = result["text"][0].split("ASSISTANT: ")[-1]
            temp_pairs["caption2"] = result["text"][1].split("ASSISTANT: ")[-1]
            caption_pairs.append(temp_pairs)

        # Step2: crop bounding box regions
        crop_image1_list = []
        crop_image2_list = []
        crop_bbox_id_to_caption1 = {}
        crop_bbox_id_to_caption2 = {}
        crop_bbox_id_to_bbox = {}
        for temp_bbox_id, temp_sample in enumerate(caption_pairs):
            text_crop_image1_json = {}
            text_crop_image2_json = {}
            temp_bbox = temp_sample["bboxes"]
            crop_bbox_id_to_bbox["<" + str(temp_bbox_id) + ">"] = temp_bbox

            text_crop_image1_json["text"] = temp_sample["caption1"]
            crop_img = image_array1[
                int(temp_bbox[1] - temp_bbox[3] / 2) : int(temp_bbox[1] + temp_bbox[3] / 2),
                int(temp_bbox[0] - temp_bbox[2] / 2) : int(temp_bbox[0] + temp_bbox[2] / 2),
                :,
            ]
            valid_img_name = samples["image_path1"].split("/")[-1].split(".")[-2]
            temp_image_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE, f"cropped_images_{valid_img_name}_{random_num}_1_<{str(temp_bbox_id)}>.jpg"
            )
            cv2.imwrite(temp_image_path, crop_img)
            cache_image_list.append(temp_image_path)
            text_crop_image1_json["images"] = [temp_image_path]
            crop_bbox_id_to_caption1["<" + str(temp_bbox_id) + ">"] = temp_sample["caption1"]

            text_crop_image2_json["text"] = temp_sample["caption2"]
            crop_img_another = image_array2[
                int(temp_bbox[1] - temp_bbox[3] / 2) : int(temp_bbox[1] + temp_bbox[3] / 2),
                int(temp_bbox[0] - temp_bbox[2] / 2) : int(temp_bbox[0] + temp_bbox[2] / 2),
                :,
            ]
            valid_img_name = samples["image_path2"].split("/")[-1].split(".")[-2]
            temp_image_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE, f"cropped_images_{valid_img_name}_{random_num}_2_<{str(temp_bbox_id)}>.jpg"
            )
            cv2.imwrite(temp_image_path, crop_img_another)
            cache_image_list.append(temp_image_path)
            text_crop_image2_json["images"] = [temp_image_path]
            crop_bbox_id_to_caption2["<" + str(temp_bbox_id) + ">"] = temp_sample["caption2"]

            crop_image1_list.append(text_crop_image1_json)
            crop_image2_list.append(text_crop_image2_json)

        # Step3: check whether the content of the regions matches the captions
        crop_image1_samples = data_juicer.core.NestedDataset.from_list(crop_image1_list)
        if Fields.stats not in crop_image1_samples.features:
            crop_image1_samples = crop_image1_samples.add_column(
                name=Fields.stats, column=[{}] * crop_image1_samples.num_rows
            )
        crop_image1_filtered = crop_image1_samples.map(
            self.fused_ops[1].compute_stats, num_proc=self.image_text_matching_filter_args["num_proc"], with_rank=True
        )
        crop_image1_filtered = crop_image1_filtered.filter(
            self.fused_ops[1].process, num_proc=self.image_text_matching_filter_args["num_proc"]
        )
        crop_image1_filtered = crop_image1_filtered.to_list()

        crop_image2_samples = data_juicer.core.NestedDataset.from_list(crop_image2_list)
        if Fields.stats not in crop_image2_samples.features:
            crop_image2_samples = crop_image2_samples.add_column(
                name=Fields.stats, column=[{}] * crop_image2_samples.num_rows
            )
        crop_image2_filtered = crop_image2_samples.map(
            self.fused_ops[1].compute_stats, num_proc=self.image_text_matching_filter_args["num_proc"], with_rank=True
        )
        crop_image2_filtered = crop_image2_filtered.filter(
            self.fused_ops[1].process, num_proc=self.image_text_matching_filter_args["num_proc"]
        )
        crop_image2_filtered = crop_image2_filtered.to_list()

        crop_image2_filtered_bbox_id = []
        seen = []
        for temp_crop_image2_filtered in crop_image2_filtered:
            crop_image2_filtered_bbox_id.append(temp_crop_image2_filtered["images"][0].split("_")[-1].split(".")[-2])

        filtered_caption_pairs = []
        for temp_crop_image1_filtered in crop_image1_filtered:
            temp_bbox_id = temp_crop_image1_filtered["images"][0].split("_")[-1].split(".")[-2]
            if temp_bbox_id in seen:
                continue
            if temp_bbox_id in crop_image2_filtered_bbox_id:
                seen.append(temp_bbox_id)
                temp_filtered_caption_pairs = {}
                temp_filtered_caption_pairs["text"] = crop_bbox_id_to_caption1[temp_bbox_id]
                temp_filtered_caption_pairs["target_text"] = crop_bbox_id_to_caption2[temp_bbox_id]
                filtered_caption_pairs.append(temp_filtered_caption_pairs)

        if len(filtered_caption_pairs) == 0:
            for temp_image_path in cache_image_list:
                os.remove(temp_image_path)
            return {
                Fields.meta: {
                    "region_caption1": [""],
                    "region_caption2": [""],
                    MetaKeys.bbox_tag: np.zeros((1, 4), dtype=np.float32),
                    "bbox_difference_captions": [""],
                }
            }

        # Step4: determine whether there are differences between the two captions.
        filtered_caption_pairs = data_juicer.core.NestedDataset.from_list(filtered_caption_pairs)
        if Fields.stats not in filtered_caption_pairs.features:
            filtered_caption_pairs = filtered_caption_pairs.add_column(
                name=Fields.stats, column=[{}] * filtered_caption_pairs.num_rows
            )

        filtered_caption_pairs = filtered_caption_pairs.map(
            self.fused_ops[2].compute_stats, num_proc=self.text_pair_similarity_filter_args["num_proc"], with_rank=True
        )
        filtered_caption_pairs = filtered_caption_pairs.filter(
            self.fused_ops[2].process, num_proc=self.text_pair_similarity_filter_args["num_proc"]
        )
        filtered_caption_pairs = filtered_caption_pairs.to_list()

        effective_bboxes_caption1 = []
        effective_bboxes_caption2 = []

        for temp_filtered_caption_pairs in filtered_caption_pairs:
            effective_bboxes_caption1.append(temp_filtered_caption_pairs["text"])
            effective_bboxes_caption2.append(temp_filtered_caption_pairs["target_text"])

        effective_bboxes = []
        for temp_bbox_id in seen:
            if (
                crop_bbox_id_to_caption1[temp_bbox_id] in effective_bboxes_caption1
                and crop_bbox_id_to_caption2[temp_bbox_id] in effective_bboxes_caption2
            ):
                temp_bbox_json = {}
                temp_bbox_json["bboxes"] = crop_bbox_id_to_bbox[temp_bbox_id]
                temp_bbox_json["caption1"] = crop_bbox_id_to_caption1[temp_bbox_id]
                temp_bbox_json["caption2"] = crop_bbox_id_to_caption2[temp_bbox_id]
                effective_bboxes.append(temp_bbox_json)

        if len(effective_bboxes) == 0:
            for temp_image_path in cache_image_list:
                os.remove(temp_image_path)
            return {
                Fields.meta: {
                    "region_caption1": [""],
                    "region_caption2": [""],
                    MetaKeys.bbox_tag: np.zeros((1, 4), dtype=np.float32),
                    "bbox_difference_captions": [""],
                }
            }

        # Step5: Mark the difference area with a red box
        text_mllm_samples = []
        for temp_bbox_id, temp_bbox_json in enumerate(effective_bboxes):
            temp_bbox = temp_bbox_json["bboxes"]
            extend_width = 5
            if temp_bbox[0] - temp_bbox[2] / 2 - extend_width >= 0:
                extend_x1 = temp_bbox[0] - temp_bbox[2] / 2 - extend_width
            else:
                extend_x1 = 0

            if temp_bbox[1] - temp_bbox[3] / 2 - extend_width >= 0:
                extend_y1 = temp_bbox[1] - temp_bbox[3] / 2 - extend_width
            else:
                extend_y1 = 0

            if temp_bbox[0] + temp_bbox[2] / 2 + extend_width <= image_array1.shape[1]:
                extend_x2 = temp_bbox[0] + temp_bbox[2] / 2 + extend_width
            else:
                extend_x2 = image_array1.shape[1]

            if temp_bbox[1] + temp_bbox[3] / 2 + extend_width <= image_array1.shape[0]:
                extend_y2 = temp_bbox[1] + temp_bbox[3] / 2 + extend_width
            else:
                extend_y2 = image_array1.shape[0]

            temp_image_array1 = deepcopy(image_array1)
            temp_image_array2 = deepcopy(image_array2)

            cv2.rectangle(
                temp_image_array1, (int(extend_x1), int(extend_y1)), (int(extend_x2), int(extend_y2)), (0, 0, 255), 3
            )
            cv2.rectangle(
                temp_image_array2, (int(extend_x1), int(extend_y1)), (int(extend_x2), int(extend_y2)), (0, 0, 255), 3
            )

            gap = np.zeros((temp_image_array1.shape[0], 20, 3), dtype=np.uint8)
            concat_image = cv2.hconcat([temp_image_array1, gap, temp_image_array2])
            valid_img_name = samples["image_path1"].split("/")[-1].split(".")[-2]
            temp_image_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE, f"red_box_images_{valid_img_name}_{random_num}_<{str(temp_bbox_id)}>.jpg"
            )
            cv2.imwrite(temp_image_path, concat_image)
            cache_image_list.append(temp_image_path)

            text_key = (
                "Analyse the left image and the right image "
                "(separated by the black vertical bar). The detail "
                "within the red bounding box in the left image is: "
                + temp_bbox_json["caption1"]
                + ", the detail within "
                "the red bounding box in the right image is: " + temp_bbox_json["caption2"] + ". What is their "
                "difference? Answer with a few concise sentences."
            )
            image_key = [temp_image_path]
            text_mllm_samples.append({"text": text_key, "images": image_key})

        # Step6: generate the difference captions
        samples[Fields.meta]["region_caption1"] = []
        samples[Fields.meta]["region_caption2"] = []
        samples[Fields.meta][MetaKeys.bbox_tag] = []
        samples[Fields.meta]["bbox_difference_captions"] = []
        for temp_bbox_json, sample in zip(effective_bboxes, text_mllm_samples):
            result = self.fused_ops[0].process(sample)
            samples[Fields.meta]["region_caption1"].append(temp_bbox_json["caption1"])
            samples[Fields.meta]["region_caption2"].append(temp_bbox_json["caption2"])
            samples[Fields.meta][MetaKeys.bbox_tag].append(temp_bbox_json["bboxes"])
            samples[Fields.meta]["bbox_difference_captions"].append(result["text"][0].split("ASSISTANT: ")[-1])

        # Step7: clear the cache
        for temp_image_path in cache_image_list:
            os.remove(temp_image_path)

        return samples
