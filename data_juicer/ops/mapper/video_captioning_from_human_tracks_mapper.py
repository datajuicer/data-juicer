import gc
import os
import pickle
import shutil
import tempfile

import numpy as np

from data_juicer.utils.ASD_mapper_utils import get_video_array_cv2
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

cv2 = LazyLoader("cv2", "opencv-contrib-python")
torch = LazyLoader("torch")

OP_NAME = "video_captioning_from_human_tracks_mapper"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptioningFromHumanTracksMapper(Mapper):
    """
    Mapper to generate samples whose captions are generated based on
    a video-to-text model and sampled video frame.

    Source: This operator is a part of HumanVBench (CVPR 2026).
    """

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        human_track_query: str = "Descibe the person's apperance. Less than 80 words. ",
        trust_remote_code: bool = False,
        video_describe_model_path: str = "DAMO-NLP-SG/VideoLLaMA3-7B",
        temp_video_path: str = None,
        tag_field_name_track_video_caption: str = MetaKeys.track_video_caption,
        tag_field_name_video_track_is_child: str = MetaKeys.video_track_is_child,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_video_blip: video-blip model name on huggingface
            to generate caption
        """
        kwargs.setdefault("mem_required", "40GB")
        super().__init__(*args, **kwargs)

        self.query = human_track_query
        self.is_child_query = "Is the person a little child? Just answer yes or no. "
        self.temp_video_path = temp_video_path

        self.video_describe_model_path = (
            video_describe_model_path if video_describe_model_path else "DAMO-NLP-SG/VideoLLaMA3-7B"
        )
        self.model_key = prepare_model(
            model_type="huggingface",
            pretrained_model_name_or_path=video_describe_model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        self.tag_field_name_track_video_caption = tag_field_name_track_video_caption
        self.tag_field_name_video_track_is_child = tag_field_name_video_track_is_child

    def process_single(self, sample, rank=None):
        if Fields.meta not in sample:
            sample[Fields.meta] = {}

        if MetaKeys.human_track_data_path not in sample[Fields.meta]:
            raise ValueError(
                "video_captioning_from_human_tracks_mapper must be operated after video_human_tracks_extraction_mapper."
            )

        Total_information = []
        Is_child = []
        video_samples = sample[Fields.meta][MetaKeys.human_track_data_path]
        loaded_video_keys = sample[self.video_key]

        cropping_face_video_temp_path = tempfile.mkdtemp(dir=self.temp_video_path)

        model, processor = get_model(self.model_key, rank, self.use_cuda())

        for vedio_id, attribute_all_tracks_for_one_video in enumerate(video_samples):
            if len(attribute_all_tracks_for_one_video) == 0:
                Total_information.append([])
                Is_child.append([])
                continue
            description_for_each_track = []
            is_child_for_each_track = []
            video_array = get_video_array_cv2(loaded_video_keys[vedio_id])
            for track_id, tracks_now in enumerate(attribute_all_tracks_for_one_video):
                with open(tracks_now["bbox_path"], "rb") as f:
                    bbox_data = pickle.load(f)
                    xy_human_bbox = bbox_data["xy_human_bbox"]
                    track_frame = bbox_data["frame"]

                human_video_out_path = os.path.join(
                    cropping_face_video_temp_path,
                    loaded_video_keys[vedio_id].split("/")[-1][:-4] + "__" + str(track_id) + ".mp4",
                )
                wide_max = int((np.array(xy_human_bbox["x2"]) - np.array(xy_human_bbox["x1"])).max())
                height_max = int((np.array(xy_human_bbox["y2"]) - np.array(xy_human_bbox["y1"])).max())
                pad_max_w = (wide_max - (np.array(xy_human_bbox["x2"]) - np.array(xy_human_bbox["x1"]))).max()
                pad_max_h = (height_max - (np.array(xy_human_bbox["y2"]) - np.array(xy_human_bbox["y1"]))).max()
                pad_max = int(max(pad_max_w, pad_max_h)) + 1

                start_frame_id_in = 0
                start_frame_id_out = track_frame[start_frame_id_in]  # tag

                num_total_frames = len(track_frame)

                target_write_fps = 25 if num_total_frames > 25 else max(1, num_total_frames - 1)

                vOut = cv2.VideoWriter(
                    human_video_out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    target_write_fps,
                    (wide_max + 2, height_max + 2),
                )

                while start_frame_id_in < len(track_frame):
                    bsi = pad_max  #

                    start_frame_id_out = track_frame[start_frame_id_in]
                    frame_before_crop = video_array[start_frame_id_out]

                    frame = np.pad(
                        frame_before_crop,
                        ((pad_max, pad_max), (pad_max, pad_max), (0, 0)),
                        "constant",
                        constant_values=(110, 110),
                    )

                    mx1_o = int(xy_human_bbox["x1"][start_frame_id_in] + bsi)  # BBox center Y
                    mx2_o = int(xy_human_bbox["x2"][start_frame_id_in] + bsi)  # BBox center X
                    my1_o = int(xy_human_bbox["y1"][start_frame_id_in] + bsi)  # BBox center Y
                    my2_o = int(xy_human_bbox["y2"][start_frame_id_in] + bsi)  # BBox center X

                    wide_target = mx2_o - mx1_o
                    high_target = my2_o - my1_o
                    left_add = int((wide_max + 2 - wide_target) / 2)
                    right_add = wide_max + 2 - wide_target - left_add
                    # mx1 = mx1_o - left_add
                    # mx2 = mx2_o + right_add
                    up_add = int((height_max + 2 - high_target) / 2)
                    down_add = height_max + 2 - high_target - up_add
                    # my1 = my1_o - up_add
                    # my2 = my2_o + down_add
                    human = frame[my1_o:my2_o, mx1_o:mx2_o]

                    padded_image = cv2.copyMakeBorder(
                        human, up_add, down_add, left_add, right_add, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )

                    vOut.write(padded_image)
                    start_frame_id_in = start_frame_id_in + 1
                vOut.release()

                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": {"video_path": human_video_out_path, "fps": 1, "max_frames": 180},
                            },
                            {"type": "text", "text": self.query},
                        ],
                    },
                ]

                inputs = processor(
                    conversation=conversation, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt"
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                output_ids = model.generate(**inputs, max_new_tokens=1024)

                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                description_for_each_track.append([response])

                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": {"video_path": human_video_out_path, "fps": 1, "max_frames": 180},
                            },
                            {"type": "text", "text": self.is_child_query},
                        ],
                    },
                ]

                inputs = processor(
                    conversation=conversation, add_system_prompt=True, add_generation_prompt=True, return_tensors="pt"
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                output_ids = model.generate(**inputs, max_new_tokens=1024)

                outputs_ischild_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                is_child_for_each_track.append([outputs_ischild_response])

            Total_information.append(description_for_each_track)
            Is_child.append(is_child_for_each_track)

        shutil.rmtree(cropping_face_video_temp_path)
        sample[Fields.meta][self.tag_field_name_track_video_caption] = Total_information
        sample[Fields.meta][self.tag_field_name_video_track_is_child] = Is_child
        gc.collect()
        torch.cuda.empty_cache()
        return sample
