import difflib
import os
import random
import re
from typing import Dict, Optional

import numpy as np

import data_juicer
from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from data_juicer.ops.load import load_ops
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import SpecialTokens

cv2 = LazyLoader("cv2", "opencv-python")
nltk = LazyLoader("nltk")


def is_noun(word):
    # print(word)
    pos_tagged = nltk.pos_tag([word])
    pos = pos_tagged[0][1]

    if pos not in ["NN", "NNS", "NNP", "NNPS"]:
        return False

    return True


def compare_text_index(text1, text2):
    text1_split = []
    text2_split = []

    lemmatizer = nltk.stem.WordNetLemmatizer()

    d = difflib.Differ()
    diff = d.compare(
        re.sub(r"[^\w\s]", "", text1.lower().replace(" ", "\n")).splitlines(),
        re.sub(r"[^\w\s]", "", text2.lower().replace(" ", "\n")).splitlines(),
    )

    for line in diff:
        if line.startswith("+"):
            text2_split.append(lemmatizer.lemmatize(line.replace("+ ", "")))
        elif line.startswith("-"):
            text1_split.append(lemmatizer.lemmatize(line.replace("- ", "")))

    text1 = []
    text2 = []

    for temp_idx, temp_word1 in enumerate(text1_split):
        if temp_word1 not in text2_split:
            if is_noun(temp_word1):
                text1.append(temp_word1)

    for temp_idx, temp_word2 in enumerate(text2_split):
        if temp_word2 not in text1_split:
            if is_noun(temp_word2):
                text2.append(temp_word2)

    return text1, text2


def iou_filter(samples, iou_thresh):
    x1 = samples[:, 0] - samples[:, 2] / 2
    y1 = samples[:, 1] - samples[:, 3] / 2
    x2 = samples[:, 0] + samples[:, 2] / 2
    y2 = samples[:, 1] + samples[:, 3] / 2

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep_boxes = []
    index = np.arange(len(samples))

    while len(index) > 0:
        i = index[0]
        keep_boxes.append(i)

        x1_overlap = np.maximum(x1[i], x1[index[1:]])
        y1_overlap = np.maximum(y1[i], y1[index[1:]])
        x2_overlap = np.minimum(x2[i], x2[index[1:]])
        y2_overlap = np.minimum(y2[i], y2[index[1:]])  # len(y2_overlap) == len(index) - 1

        w = np.maximum(0, x2_overlap - x1_overlap + 1)
        h = np.maximum(0, y2_overlap - y1_overlap + 1)
        overlap_area = w * h

        ious = overlap_area / (areas[i] + areas[index[1:]] - overlap_area)

        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]  # update

    return samples[keep_boxes]


OP_NAME = "imgdiff_difference_area_generator_mapper"


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class Difference_Area_Generator_Mapper(Mapper):
    """Generates and filters bounding boxes for image pairs based on similarity, segmentation,
    and text matching.

    This operator processes image pairs to identify and filter regions with significant
    differences. It uses a sequence of operations:
    - Filters out image pairs with large differences.
    - Segments the images to identify potential objects.
    - Crops sub-images based on bounding boxes.
    - Determines if the sub-images contain valid objects using image-text matching.
    - Filters out sub-images that are too similar.
    - Removes overlapping bounding boxes.
    - Uses Hugging Face models for similarity and text matching, and FastSAM for
      segmentation.
    - Caches intermediate results in `DATA_JUICER_ASSETS_CACHE`.
    - Returns the filtered bounding boxes in the `MetaKeys.bbox_tag` field."""

    _accelerator = "cuda"

    def __init__(
        self,
        image_pair_similarity_filter_args: Optional[Dict] = {},
        image_segment_mapper_args: Optional[Dict] = {},
        image_text_matching_filter_args: Optional[Dict] = {},
        *args,
        **kwargs,
    ):
        """Initialization.

        :param image_pair_similarity_filter_args: Arguments for image pair similarity filter.
            Controls the similarity filtering between image pairs. Default empty dict will use
            fixed values: min_score_1=0.1, max_score_1=1.0, min_score_2=0.1, max_score_2=1.0,
            hf_clip="openai/clip-vit-base-patch32", num_proc=1.
        :param image_segment_mapper_args: Arguments for image segmentation mapper.
            Controls the image segmentation process. Default empty dict will use
            fixed values: imgsz=1024, conf=0.05, iou=0.5, model_path="FastSAM-x.pt".
        :param image_text_matching_filter_args: Arguments for image-text matching filter.
            Controls the matching between cropped image regions and text descriptions.
            Default empty dict will use fixed values: min_score=0.1, max_score=1.0,
            hf_blip="Salesforce/blip-itm-base-coco", num_proc=1.
        """
        super().__init__(*args, **kwargs)

        self.FIXED_ARGS = {}
        self.FIXED_ARGS["image_pair_similarity_filter"] = {
            "min_score_1": 0.1,
            "max_score_1": 1.0,
            "min_score_2": 0.1,
            "max_score_2": 1.0,
            "hf_clip": "openai/clip-vit-base-patch32",
            "num_proc": 1,
        }
        self.FIXED_ARGS["image_segment_mapper"] = {
            "imgsz": 1024,
            "conf": 0.05,
            "iou": 0.5,
            "model_path": "FastSAM-x.pt",
        }
        self.FIXED_ARGS["image_text_matching_filter"] = {
            "min_score": 0.1,
            "max_score": 1.0,
            "hf_blip": "Salesforce/blip-itm-base-coco",
            "num_proc": 1,
        }

        self.image_pair_similarity_filter_args = self._prepare_op_args(
            "image_pair_similarity_filter", image_pair_similarity_filter_args
        )
        self.image_segment_mapper_args = self._prepare_op_args("image_segment_mapper", image_segment_mapper_args)
        self.image_text_matching_filter_args = self._prepare_op_args(
            "image_text_matching_filter", image_text_matching_filter_args
        )

        self.fused_op_list = [
            {"image_pair_similarity_filter": self.image_pair_similarity_filter_args},
            {"image_segment_mapper": self.image_segment_mapper_args},
            {"image_text_matching_filter": self.image_text_matching_filter_args},
        ]

        self.fused_ops = load_ops(self.fused_op_list)
        self._name = "Difference_Area_Generator_Mapper:(%s)" % ",".join([op._name for op in self.fused_ops])
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
        self.fused_ops[0].min_score = self.image_pair_similarity_filter_args["min_score_1"]
        self.fused_ops[0].max_score = self.image_pair_similarity_filter_args["max_score_1"]

        if not os.path.exists(DATA_JUICER_ASSETS_CACHE):
            os.makedirs(DATA_JUICER_ASSETS_CACHE, exist_ok=True)

        # fused_ops 1.image_pair_similarity_filter 2.image_segment_mapper
        # 3.image_text_matching_filter keys of sample: "caption1", "caption2",
        # "image_path1", "image_path2"

        # Step1: filter out image pairs with large differences between the two
        # images.
        temp_sample = {}
        temp_sample["text"] = "temp image pairs " + random_num
        temp_sample["images"] = [samples["image_path1"], samples["image_path2"]]

        temp_sample = data_juicer.core.NestedDataset.from_list([temp_sample])
        if Fields.stats not in temp_sample.features:
            temp_sample = temp_sample.add_column(name=Fields.stats, column=[{}] * temp_sample.num_rows)
        new_samples_s1 = self.fused_ops[0].compute_stats_single(temp_sample[0], rank=rank)
        new_samples_s1 = self.fused_ops[0].process_single(new_samples_s1, rank=rank)

        if not new_samples_s1:
            return {Fields.meta: {MetaKeys.bbox_tag: np.zeros((1, 4), dtype=np.float32)}}

        # Step2: compare the differences between the two captions and identify
        # the "valid object".
        valid_object1, valid_object2 = compare_text_index(samples["caption1"], samples["caption2"])

        # Step3: segment the regions in two images that may contain valid
        # objects.
        temp_sample = {}
        temp_sample["images"] = [samples["image_path1"], samples["image_path2"]]
        temp_sample = data_juicer.core.NestedDataset.from_list([temp_sample])
        if Fields.meta not in temp_sample.features:
            temp_sample = temp_sample.add_column(name=Fields.meta, column=[{}] * temp_sample.num_rows)
        new_samples_s2 = self.fused_ops[1].process_single(temp_sample[0], rank=rank)

        image1_bboxes = new_samples_s2[Fields.meta][MetaKeys.bbox_tag][0]
        image2_bboxes = new_samples_s2[Fields.meta][MetaKeys.bbox_tag][1]

        # Step4: crop sub-images based on the bounding boxes for subsequent
        # image-text matching processes.
        crop_image1_samples = []
        crop_image2_samples = []
        crop_image1_path_to_bbox_dict = {}
        crop_image2_path_to_bbox_dict = {}  # Used to associate filenames with bounding boxes.

        image_array1 = cv2.imread(samples["image_path1"])
        image_array2 = cv2.imread(samples["image_path2"])
        image_array2 = cv2.resize(image_array2, (image_array1.shape[1], image_array1.shape[0]))

        # bbox from 1 -> crop 1, 2
        for temp_bbox_id, temp_bbox in enumerate(image1_bboxes):
            crop_img = image_array1[
                int(temp_bbox[1] - temp_bbox[3] / 2) : int(temp_bbox[1] + temp_bbox[3] / 2),
                int(temp_bbox[0] - temp_bbox[2] / 2) : int(temp_bbox[0] + temp_bbox[2] / 2),
                :,
            ]
            valid_img_name = samples["image_path1"].split("/")[-1].split(".")[-2]
            temp_image_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE,
                f"cropped_images_{valid_img_name}_{random_num}_1_" f"<{str(temp_bbox_id)}>.jpg",
            )
            cv2.imwrite(temp_image_path, crop_img)
            crop_image1_path_to_bbox_dict[temp_image_path] = temp_bbox
            for temp_valid_object1 in valid_object1:
                crop_image1_samples.append(
                    {
                        "text": f"{SpecialTokens.image}" + temp_valid_object1 + f"{SpecialTokens.eoc} ",
                        "images": [temp_image_path],
                    }
                )

            crop_img_another = image_array2[
                int(temp_bbox[1] - temp_bbox[3] / 2) : int(temp_bbox[1] + temp_bbox[3] / 2),
                int(temp_bbox[0] - temp_bbox[2] / 2) : int(temp_bbox[0] + temp_bbox[2] / 2),
                :,
            ]
            valid_img_name = samples["image_path2"].split("/")[-1].split(".")[-2]
            temp_image_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE,
                f"cropped_images_{valid_img_name}_{random_num}_2_" f"<{str(temp_bbox_id)}>.jpg",
            )
            cv2.imwrite(temp_image_path, crop_img_another)
            crop_image2_path_to_bbox_dict[temp_image_path] = temp_bbox
            for temp_valid_object2 in valid_object2:
                crop_image2_samples.append(
                    {
                        "text": f"{SpecialTokens.image}" + temp_valid_object2 + f"{SpecialTokens.eoc} ",
                        "images": [temp_image_path],
                    }
                )

        # bbox from 2 -> crop 2, 1
        for temp_bbox_id, temp_bbox in enumerate(image2_bboxes):
            temp_crop_image_pair_id = len(image1_bboxes) + temp_bbox_id

            crop_img = image_array2[
                int(temp_bbox[1] - temp_bbox[3] / 2) : int(temp_bbox[1] + temp_bbox[3] / 2),
                int(temp_bbox[0] - temp_bbox[2] / 2) : int(temp_bbox[0] + temp_bbox[2] / 2),
                :,
            ]
            valid_img_name = samples["image_path2"].split("/")[-1].split(".")[-2]
            temp_image_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE,
                f"cropped_images_{valid_img_name}_{random_num}_2_" f"<{str(temp_crop_image_pair_id)}>.jpg",
            )
            cv2.imwrite(temp_image_path, crop_img)
            crop_image2_path_to_bbox_dict[temp_image_path] = temp_bbox
            for temp_valid_object2 in valid_object2:
                crop_image2_samples.append(
                    {
                        "text": f"{SpecialTokens.image}" + temp_valid_object2 + f"{SpecialTokens.eoc} ",
                        "images": [temp_image_path],
                    }
                )

            crop_img_another = image_array1[
                int(temp_bbox[1] - temp_bbox[3] / 2) : int(temp_bbox[1] + temp_bbox[3] / 2),
                int(temp_bbox[0] - temp_bbox[2] / 2) : int(temp_bbox[0] + temp_bbox[2] / 2),
                :,
            ]
            valid_img_name = samples["image_path1"].split("/")[-1].split(".")[-2]
            temp_image_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE,
                f"cropped_images_{valid_img_name}_{random_num}_1_" f"<{str(temp_crop_image_pair_id)}>.jpg",
            )
            cv2.imwrite(temp_image_path, crop_img_another)
            crop_image1_path_to_bbox_dict[temp_image_path] = temp_bbox
            for temp_valid_object1 in valid_object1:
                crop_image1_samples.append(
                    {
                        "text": f"{SpecialTokens.image}" + temp_valid_object1 + f"{SpecialTokens.eoc} ",
                        "images": [temp_image_path],
                    }
                )

        # Step5: determine whether the sub-images contain valid objects.
        crop_image1_samples = data_juicer.core.NestedDataset.from_list(crop_image1_samples)
        if Fields.stats not in crop_image1_samples.features:
            crop_image1_samples = crop_image1_samples.add_column(
                name=Fields.stats, column=[{}] * crop_image1_samples.num_rows
            )
        crop_image1_filtered = crop_image1_samples.map(
            self.fused_ops[2].compute_stats, num_proc=self.image_text_matching_filter_args["num_proc"], with_rank=True
        )
        crop_image1_filtered = crop_image1_filtered.filter(
            self.fused_ops[2].process, num_proc=self.image_text_matching_filter_args["num_proc"]
        )
        crop_image1_filtered = crop_image1_filtered.to_list()

        crop_image2_samples = data_juicer.core.NestedDataset.from_list(crop_image2_samples)
        if Fields.stats not in crop_image2_samples.features:
            crop_image2_samples = crop_image2_samples.add_column(
                name=Fields.stats, column=[{}] * crop_image2_samples.num_rows
            )
        crop_image2_filtered = crop_image2_samples.map(
            self.fused_ops[2].compute_stats, num_proc=self.image_text_matching_filter_args["num_proc"], with_rank=True
        )
        crop_image2_filtered = crop_image2_filtered.filter(
            self.fused_ops[2].process, num_proc=self.image_text_matching_filter_args["num_proc"]
        )
        crop_image2_filtered = crop_image2_filtered.to_list()

        crop_image2_filtered_bbox_id = []
        seen = []
        for temp_crop_image2_filtered in crop_image2_filtered:
            crop_image2_filtered_bbox_id.append(temp_crop_image2_filtered["images"][0].split("_")[-1].split(".")[-2])

        filtered_sub_image_pairs = []
        for temp_crop_image1_filtered in crop_image1_filtered:
            temp_bbox_id = temp_crop_image1_filtered["images"][0].split("_")[-1].split(".")[-2]
            if temp_bbox_id in seen:
                continue
            if temp_bbox_id in crop_image2_filtered_bbox_id:
                seen.append(temp_bbox_id)
                temp_filtered_sub_image_pairs = {}
                temp_filtered_sub_image_pairs["text"] = temp_bbox_id
                valid_image_path1 = samples["image_path1"].split("/")[-1].split(".")[-2]
                valid_image_path2 = samples["image_path2"].split("/")[-1].split(".")[-2]
                temp_filtered_sub_image_pairs["images"] = [
                    temp_crop_image1_filtered["images"][0],
                    temp_crop_image1_filtered["images"][0]
                    .replace(valid_image_path1, valid_image_path2)
                    .replace("_1_<", "_2_<"),
                ]
                filtered_sub_image_pairs.append(temp_filtered_sub_image_pairs)

        # Step6: determine whether there are differences in the two images
        # corresponding to each bounding box.
        filtered_sub_image_pairs = data_juicer.core.NestedDataset.from_list(filtered_sub_image_pairs)
        self.fused_ops[0].min_score = self.image_pair_similarity_filter_args["min_score_2"]
        self.fused_ops[0].max_score = self.image_pair_similarity_filter_args["max_score_2"]
        if Fields.stats not in filtered_sub_image_pairs.features:
            filtered_sub_image_pairs = filtered_sub_image_pairs.add_column(
                name=Fields.stats, column=[{}] * filtered_sub_image_pairs.num_rows
            )
        filtered_sub_image_pairs = filtered_sub_image_pairs.map(
            self.fused_ops[0].compute_stats, num_proc=self.image_pair_similarity_filter_args["num_proc"], with_rank=True
        )
        filtered_sub_image_pairs = filtered_sub_image_pairs.filter(
            self.fused_ops[0].process, num_proc=self.image_pair_similarity_filter_args["num_proc"]
        )
        filtered_sub_image_pairs = filtered_sub_image_pairs.to_list()

        if len(filtered_sub_image_pairs) == 0:
            for temp_image_path in crop_image1_path_to_bbox_dict:
                os.remove(temp_image_path)
            for temp_image_path in crop_image2_path_to_bbox_dict:
                os.remove(temp_image_path)
            return {Fields.meta: {MetaKeys.bbox_tag: np.zeros((1, 4), dtype=np.float32)}}

        filtered_bboxes = []
        for temp_sub_image_pairs in filtered_sub_image_pairs:
            filtered_bboxes.append(crop_image1_path_to_bbox_dict[temp_sub_image_pairs["images"][0]])

        filtered_bboxes = np.array(filtered_bboxes)

        # Step7: remove overlapping bounding boxes.
        iou_thresh = 0.5
        filtered_bboxes = iou_filter(filtered_bboxes, iou_thresh)
        samples[Fields.meta] = {}
        samples[Fields.meta][MetaKeys.bbox_tag] = filtered_bboxes

        # Step8: clear the cache
        for temp_image_path in crop_image1_path_to_bbox_dict:
            os.remove(temp_image_path)
        for temp_image_path in crop_image2_path_to_bbox_dict:
            os.remove(temp_image_path)

        return samples
