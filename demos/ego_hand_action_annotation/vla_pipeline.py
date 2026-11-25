import os
import sys
import copy
import time

import torch
import ray
from ray.data import ActorPoolStrategy

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.ops.mapper import (
    VideoCameraCalibrationMogeMapper,
    VideoHandReconstructionHaworMapper,
    VideoCameraPoseMegaSaMMapper,
    VideoExtractFramesMapper,
    VideoHandActionComputeMapper,
    ExportToLeRobotMapper)

sys.path.insert(0, os.path.dirname(__file__))

from custom_ops.video_action_captioning_mapper import VideoActionCaptioningMapper


@OPERATORS.register_module("video_hawor_megasam_combined_mapper")
class VideoHaWorMegaSaMCombinedMapper(Mapper):
    """Combined HaWoR hand reconstruction + MegaSaM camera pose estimation.

    Runs both GPU models sequentially on the same actor to:
    1. Avoid inter-stage data serialization overhead
    2. Simplify Ray scheduling (fewer GPU stages = less resource contention)
    3. Share GPU memory efficiently
    """

    _accelerator = "cuda"

    def __init__(
        self,
        # HaWoR params
        hawor_model_path: str = "hawor.ckpt",
        hawor_config_path: str = "model_config.yaml",
        hawor_detector_path: str = "detector.pt",
        mano_right_path: str = "MANO_RIGHT.pkl",
        mano_left_path: str = "MANO_LEFT.pkl",
        camera_calibration_field: str = MetaKeys.camera_calibration_moge_tags,
        hawor_tag_field: str = MetaKeys.hand_reconstruction_hawor_tags,
        frame_field: str = MetaKeys.video_frames,
        hawor_thresh: float = 0.2,
        # MegaSaM params
        megasam_tag_field: str = MetaKeys.video_camera_pose_tags,
        megasam_max_frames: int = 1000,
        megasam_droid_buffer: int = 1024,
        megasam_save_dir: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._hawor_kwargs = dict(
            hawor_model_path=hawor_model_path,
            hawor_config_path=hawor_config_path,
            hawor_detector_path=hawor_detector_path,
            mano_right_path=mano_right_path,
            mano_left_path=mano_left_path,
            camera_calibration_field=camera_calibration_field,
            tag_field_name=hawor_tag_field,
            frame_field=frame_field,
            thresh=hawor_thresh,
            batch_mode=True,
            skip_op_error=kwargs.get("skip_op_error", False),
        )
        self._megasam_kwargs = dict(
            tag_field_name=megasam_tag_field,
            camera_calibration_field=camera_calibration_field,
            frame_field=frame_field,
            max_frames=megasam_max_frames,
            droid_buffer=megasam_droid_buffer,
            save_dir=megasam_save_dir,
            batch_mode=True,
            skip_op_error=kwargs.get("skip_op_error", False),
        )

        self._hawor_op = None
        self._megasam_op = None

    def _ensure_ops(self):
        if self._hawor_op is None:
            self._hawor_op = VideoHandReconstructionHaworMapper(**self._hawor_kwargs)
        if self._megasam_op is None:
            self._megasam_op = VideoCameraPoseMegaSaMMapper(**self._megasam_kwargs)

    def process_single(self, sample=None, rank=None):
        from loguru import logger as _logger

        self._ensure_ops()

        sample = self._hawor_op.process_single(sample, rank=rank)

        try:
            sample = self._megasam_op.process_single(sample, rank=rank)
        except Exception as e:
            # MegaSaM failure should not discard HaWoR results.
            # Write empty camera pose so downstream can still detect
            # the missing data gracefully instead of crashing.
            import traceback
            _logger.error(f"MegaSaM failed (HaWoR result preserved): "
                          f"{e} -- {traceback.format_exc()}")
            megasam_field = self._megasam_kwargs.get(
                "tag_field_name", MetaKeys.video_camera_pose_tags)
            if Fields.meta not in sample:
                sample[Fields.meta] = {}
            # One empty entry per video clip
            n_videos = len(sample.get(self._hawor_kwargs.get(
                "frame_field", MetaKeys.video_frames), []))
            sample[Fields.meta][megasam_field] = [
                {} for _ in range(max(1, n_videos))
            ]


        return sample


if __name__ == '__main__':

    from ray.data import DataContext
    DataContext.get_current().enable_fallback_to_arrow_object_ext_type = True

    s_time = time.time()

    ray.init(address='auto')

    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)
    lerobot_output_dir = os.path.join(output_dir, "lerobot_dataset")

    video_key = "videos"
    skip_op_error = False

    video_paths = [
        "./data/1018.mp4",
        "./data/1034.mp4",
    ]
    samples = [
        {
            video_key: [video],
            "text": "",
            # Fields.stats: {},
            Fields.meta: {}
        } for video in video_paths
    ]

    ds = ray.data.from_items(samples)

    ds = ds.map_batches(
        VideoExtractFramesMapper,
        fn_constructor_kwargs=dict(
            frame_sampling_method="uniform",  # "all_keyframes"
            frame_num=20,
            video_backend='ffmpeg',
            output_format='path',  #'bytes',
            frame_dir=os.path.join(output_dir, 'frames'),
            frame_field=MetaKeys.video_frames,
            legacy_split_by_text_token=False,
            batch_mode=True,
            skip_op_error=skip_op_error,
            video_key=video_key,
            ),
        batch_size=1,
        num_cpus=1,
        batch_format="pyarrow",
        runtime_env={"conda": "base"},
    )

    ds = ds.map_batches(
        VideoCameraCalibrationMogeMapper,
        fn_constructor_kwargs=dict(
            model_path="Ruicheng/moge-2-vitl",
            tag_field_name=MetaKeys.camera_calibration_moge_tags,
            frame_field=MetaKeys.video_frames,
            output_depth=True,
            output_points=False,
            output_mask=False,
            save_dir=os.path.join(output_dir, 'moge_arrays'),
            batch_mode=True,
            skip_op_error=skip_op_error,
        ),
        batch_size=1,
        num_gpus=0.15,  # adjust the ratio based on the gpu type
        batch_format="pyarrow",
        compute=ActorPoolStrategy(min_size=1, max_size=2),  # adjust the scope based on available resources
        runtime_env={"conda": "base"},
    )

    ds = ds.map_batches(
        VideoHaWorMegaSaMCombinedMapper,
        fn_constructor_kwargs=dict(
            camera_calibration_field=MetaKeys.camera_calibration_moge_tags,
            hawor_tag_field=MetaKeys.hand_reconstruction_hawor_tags,
            megasam_tag_field=MetaKeys.video_camera_pose_tags,
            mano_right_path='/path/to/MANO_RIGHT.pkl',
            mano_left_path='/path/to/MANO_LEFT.pkl',
            frame_field=MetaKeys.video_frames,
            megasam_max_frames=1000,
            megasam_save_dir=os.path.join(output_dir, 'megasam_arrays'),
            batch_mode=True,
            skip_op_error=skip_op_error,
        ),
        batch_size=1,
        num_gpus=0.25,  # adjust the ratio based on the gpu type
        batch_format="pyarrow",
        runtime_env={"conda": "mega-sam"},
        compute=ActorPoolStrategy(min_size=1, max_size=2),  # adjust the scope based on available resources
    )

    ds = ds.map_batches(
        VideoHandActionComputeMapper,
        fn_constructor_kwargs=dict(
            hand_reconstruction_field=MetaKeys.hand_reconstruction_hawor_tags,  # outputs of VideoHandReconstructionHaworMapper
            camera_pose_field=MetaKeys.video_camera_pose_tags,  # outputs of VideoCameraPoseMegaSaMMapper
            tag_field_name=MetaKeys.hand_action_tags,
            hand_type="both",
            batch_mode=True,
            skip_op_error=skip_op_error,
        ),
        batch_size=1,
        num_cpus=1,
        batch_format="pyarrow",
        runtime_env={"conda": "base"},
    )

    ds = ds.map_batches(
        VideoActionCaptioningMapper,
        fn_constructor_kwargs=dict(
            api_or_hf_model='qwen-vl-max',
            is_api_model=True,
            hand_type='both',
            frame_field=MetaKeys.video_frames,
            tag_field_name="hand_action_caption",
            batch_mode=True,
            skip_op_error=skip_op_error,
        ),
        batch_size=1,
        num_cpus=1,
        batch_format="pyarrow",
        runtime_env={"conda": "base"},
    )

    ds = ds.map_batches(
        ExportToLeRobotMapper,
        fn_constructor_kwargs=dict(
            output_dir=lerobot_output_dir,
            hand_action_field=MetaKeys.hand_action_tags,  # outputs of VideoHandActionComputeMapper
            frame_field=MetaKeys.video_frames,
            video_key=video_key,
            task_description_key="text",
            fps=10,
            robot_type="egodex_hand",
            batch_mode=True,
            skip_op_error=skip_op_error,
        ),
        batch_size=1,
        num_cpus=1,
        batch_format="pyarrow",
        runtime_env={"conda": "base"},
    )

    ds.write_parquet(output_dir)
    # ds.write_json(output_dir, force_ascii=False)

    ExportToLeRobotMapper.finalize_dataset(
        output_dir=lerobot_output_dir,
        fps=10,
        robot_type="egodex_hand",
    )

    print(f"LeRobot exported to: {lerobot_output_dir}")
    print(f'>>>>total cost time: {time.time() - s_time}')
