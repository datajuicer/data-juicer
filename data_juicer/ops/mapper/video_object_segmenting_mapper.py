import os
import random
from datetime import datetime

import numpy as np

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = "video_object_segmenting_mapper"

LazyLoader._install_package("transformers>=4.56.0.dev0")
ultralytics = LazyLoader("ultralytics")
cv2 = LazyLoader("cv2", "opencv-python")
torch = LazyLoader("torch")


@TAGGING_OPS.register_module(OP_NAME)
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoObjectSegmentingMapper(Mapper):
    """Text-guided semantic segmentation of valid objects throughout the video (YOLOE + SAM2)."""

    _accelerator = "cuda"

    def __init__(
        self,
        sam2_hf_model: str = "facebook/sam2.1-hiera-tiny",
        yoloe_path: str = "yoloe-11l-seg.pt",
        yoloe_conf: float = 0.5,
        torch_dtype: str = "bf16",
        if_binarize: bool = True,
        if_save_visualization: bool = False,
        save_visualization_dir: str = DATA_JUICER_ASSETS_CACHE,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_model: Hugginface model id of SAM2.
        :param yoloe_path: The path to the YOLOE model.
        :param yoloe_conf: Confidence threshold for YOLOE object detection.
        :param torch_dtype: The floating point type used for model inference. Can
            be one of ['fp32', 'fp16', 'bf16'].
        :param if_binarize: Whether the final mask requires binarization.
            If 'if_save_visualization' is set to True, 'if_binarize' will
            automatically be adjusted to True.
        :param if_save_visualization: Whether to save visualization results.
        :param save_visualization_dir: The path for saving visualization results.

        """

        super().__init__(*args, **kwargs)

        # Requires the weights for YOLOE and mobileclip_blt.
        self.yoloe_model_key = prepare_model(model_type="yolo", model_path=yoloe_path)
        torch_dtype_dict = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self.torch_dtype = torch_dtype_dict[torch_dtype]
        self.sam2_model_key = prepare_model(
            model_type="huggingface", torch_dtype=self.torch_dtype, pretrained_model_name_or_path=sam2_hf_model
        )

        self.tag_field_name = MetaKeys.video_object_segment_tags
        self.yoloe_conf = yoloe_conf
        self.if_save_visualization = if_save_visualization
        self.save_visualization_dir = save_visualization_dir
        self.if_binarize = True if if_save_visualization else if_binarize

    def process_single(self, sample=None, rank=None):

        # check if it's generated already
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.meta][self.tag_field_name] = {
                "segment_data": [],
                "cls_id_dict": [],
                "object_cls_list": [],
                "yoloe_conf_list": [],
            }
            return sample

        sam2_model, sam2_processor = get_model(model_key=self.sam2_model_key, rank=rank, use_cuda=self.use_cuda())

        # Perform semantic segmentation on the first frame using YOLOE
        videoCapture = cv2.VideoCapture(sample[self.video_key][0])
        success, initial_frame = videoCapture.read()
        random_num_str = str(random.randint(10000, 99999))
        now_time_str = str(datetime.now())
        if success:
            if not os.path.exists(DATA_JUICER_ASSETS_CACHE):
                os.makedirs(DATA_JUICER_ASSETS_CACHE, exist_ok=True)

            temp_video_name = sample[self.video_key][0].split("/")[-1].replace(".mp4", "")
            temp_initial_frame_path = os.path.join(
                DATA_JUICER_ASSETS_CACHE,
                f"{temp_video_name}_initial_frame_{now_time_str}_{random_num_str}.jpg",
            )
            cv2.imwrite(temp_initial_frame_path, initial_frame)
        else:
            # Failed to load initial frame
            sample[Fields.meta][self.tag_field_name] = {
                "segment_data": [],
                "cls_id_dict": [],
                "object_cls_list": [],
                "yoloe_conf_list": [],
            }
            return sample

        main_character_list = sample.get("main_character_list")
        if not main_character_list:
            sample[Fields.meta][self.tag_field_name] = {
                "segment_data": [],
                "cls_id_dict": [],
                "object_cls_list": [],
                "yoloe_conf_list": [],
            }
            return sample

        yoloe_model = get_model(model_key=self.yoloe_model_key, rank=rank, use_cuda=self.use_cuda())
        yoloe_model.set_classes(main_character_list, yoloe_model.get_text_pe(main_character_list))
        results = yoloe_model.predict(temp_initial_frame_path, verbose=False, conf=self.yoloe_conf)
        yoloe_bboxes = results[0].boxes.xyxy.tolist()
        bboxes_cls = results[0].boxes.cls.tolist()
        bboxes_cls = [int(x) for x in bboxes_cls]
        cls_id_dict = results[0].names
        yoloe_conf_list = results[0].boxes.conf.tolist()

        obj_ids = []
        object_cls_list = []
        input_boxes = []
        for temp_cls, temp_box in zip(bboxes_cls, yoloe_bboxes):
            obj_ids.append(len(obj_ids))
            object_cls_list.append(temp_cls)
            input_boxes.append([int(x) for x in temp_box])

        input_boxes = [input_boxes]
        os.remove(temp_initial_frame_path)

        if len(obj_ids) == 0:
            sample[Fields.meta][self.tag_field_name] = {
                "segment_data": [],
                "cls_id_dict": [],
                "object_cls_list": [],
                "yoloe_conf_list": [],
            }
            return sample

        # Track objects with SAM2
        import transformers

        video_frames, _ = transformers.video_utils.load_video(sample[self.video_key][0])

        if rank is not None:
            device = f"cuda:{str(rank)}"
        else:
            device = "cuda"

        inference_session = sam2_processor.init_video_session(
            video=video_frames,
            inference_device=device if self.use_cuda() else "cpu",
            dtype=self.torch_dtype,
        )

        ann_frame_idx = 0
        sam2_processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=obj_ids,
            input_boxes=input_boxes,
        )

        # Get masks for all objects on the first frame
        outputs = sam2_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
        )
        video_res_masks = sam2_processor.post_process_masks(
            [outputs.pred_masks],
            original_sizes=[[inference_session.video_height, inference_session.video_width]],
            binarize=False,
        )[0]

        #  Propagate all objects through the video
        video_segments = []
        for sam2_video_output in sam2_model.propagate_in_video_iterator(inference_session):
            video_res_masks = sam2_processor.post_process_masks(
                [sam2_video_output.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=self.if_binarize,
            )[0]
            video_segments.append([video_res_masks[i].tolist() for i, obj_id in enumerate(inference_session.obj_ids)])

        sample[Fields.meta][self.tag_field_name] = {}
        sample[Fields.meta][self.tag_field_name]["segment_data"] = video_segments
        sample[Fields.meta][self.tag_field_name]["cls_id_dict"] = [cls_id_dict[key] for key in cls_id_dict]
        sample[Fields.meta][self.tag_field_name]["object_cls_list"] = object_cls_list
        sample[Fields.meta][self.tag_field_name]["yoloe_conf_list"] = yoloe_conf_list

        if self.if_save_visualization:
            if not os.path.exists(self.save_visualization_dir):
                os.makedirs(self.save_visualization_dir, exist_ok=True)

            for temp_frame_masks_id, temp_frame_masks in enumerate(
                sample[Fields.meta][self.tag_field_name]["segment_data"]
            ):
                for temp_obj_id, temp_mask in enumerate(temp_frame_masks):
                    temp_img = np.zeros((initial_frame.shape[0], initial_frame.shape[1], 3), np.uint8)
                    temp_mask = np.squeeze(np.array(temp_mask))
                    temp_img[temp_mask] = [225, 225, 225]

                    temp_mask_path = os.path.join(
                        self.save_visualization_dir,
                        f"{temp_video_name}_mask_{str(temp_obj_id)}_{str(temp_frame_masks_id)}_{now_time_str}_{random_num_str}.jpg",
                    )
                    cv2.imwrite(temp_mask_path, temp_img)

        return sample
