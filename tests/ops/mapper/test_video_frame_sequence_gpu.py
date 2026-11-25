"""Opt-in real-checkpoint CUDA verification for ordered frame inputs.

Set DJ_RUN_GPU_TESTS=1 and DJ_GPU_MODEL_DIR to prepared checkpoints.
Run each model in a separate process to bound GPU memory.
"""

import gc
import json
import os
import unittest
from contextlib import chdir
from pathlib import Path

import numpy as np
import torch

from data_juicer.ops.mapper.video_animal_pose_mapper import VideoAnimalPoseMapper
from data_juicer.ops.mapper.video_camera_calibration_moge_mapper import (
    VideoCameraCalibrationMogeMapper,
)
from data_juicer.ops.mapper.video_depth_estimation_mapper import (
    VideoDepthEstimationMapper,
)
from data_juicer.ops.mapper.video_face_keypoints_mapper import VideoFaceKeypointsMapper
from data_juicer.ops.mapper.video_whole_body_pose_estimation_mapper import (
    VideoWholeBodyPoseEstimationMapper,
)
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys
from data_juicer.utils.model_utils import free_models, get_model

DATA = Path(__file__).resolve().parents[1] / "data"


class CudaVerifiedBodyMapper(VideoWholeBodyPoseEstimationMapper):
    """Attach worker/device evidence while retaining the real model path."""

    def process_single(self, sample, rank=None):
        result = super().process_single(sample, rank)
        if any(sample.get(MetaKeys.video_frames, [])):
            model = get_model(self.model_key, rank, self.use_cuda())
            result["cuda_providers"] = model.pose_estimation.session_pose.get_providers()
        else:
            result["cuda_providers"] = []
        result["worker_pid"] = os.getpid()
        return result


@unittest.skipUnless(os.environ.get("DJ_RUN_GPU_TESTS") == "1", "Set DJ_RUN_GPU_TESTS=1 for real CUDA inference.")
class VideoFrameSequenceGpuTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA is required when GPU tests are explicitly enabled.")
        self.models = Path(os.environ["DJ_GPU_MODEL_DIR"]).resolve()
        self.enterContext(chdir(self.models))
        self.paths = [str(DATA / "video4_frames/frame_2.jpg"), str(DATA / "video4_frames/frame_0.jpg")]
        self.common = dict(accelerator="cuda", num_proc=1, auto_op_parallelism=False)
        torch.cuda.reset_peak_memory_stats()

    def tearDown(self):
        free_models()
        gc.collect()
        torch.cuda.empty_cache()

    def sample(self):
        return {MetaKeys.video_frames: [[], self.paths], Fields.meta: {}, "fps": 24}

    def predict(self, op):
        result = op.process_single(self.sample(), rank=torch.cuda.device_count())
        json.dumps(result)
        return result[Fields.meta][op.tag_field_name]

    def assert_mapping(self, tags):
        self.assertEqual(tags["frame_paths"], self.paths)
        self.assertEqual(tags["video_indices"], [1, 1])

    def assert_cuda(self, model):
        self.assertEqual(next(model.parameters()).device.type, "cuda")

    def test_animal(self):
        self.paths = [str(DATA / "video13_frames/frame_2.jpg"), str(DATA / "video13_frames/frame_0.jpg")]
        op = VideoAnimalPoseMapper(
            vitpose_model_path=str(self.models / "apt36k.pth"),
            yoloe_model_path=str(self.models / "yoloe-26x-seg.pt"),
            if_save_visualization=False,
            **self.common,
        )
        tags = self.predict(op)
        self.assert_mapping(tags)
        self.assert_cuda(get_model(op.model_key, 0, True))
        self.assert_cuda(get_model(op.yolo_model_key, 0, True))
        self.assertTrue(all(len(frame) > 0 for frame in tags["pose_list"]))
        self.assertEqual(np.asarray(tags["pose_list"][0]).shape[1:], (17, 2))
        self.assertTrue(np.isfinite(tags["pose_list"][0]).all())

    def test_face(self):
        op = VideoFaceKeypointsMapper(
            ldeq_model_path=str(self.models / "final.pth.tar"),
            if_save_visualization=False,
            **self.common,
        )
        tags = self.predict(op)
        self.assert_mapping(tags)
        model, detector, _ = get_model(op.model_key, 0, True)
        self.assert_cuda(model)
        self.assertEqual(detector.det_model.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertTrue(all(len(frame) > 0 for frame in tags["face_keypoints"]))
        self.assertEqual(np.asarray(tags["face_keypoints"][0]).shape[1:], (98, 2))
        self.assertTrue(np.isfinite(tags["face_keypoints"][0]).all())

    def test_body(self):
        import onnxruntime

        onnxruntime.preload_dlls()
        op = VideoWholeBodyPoseEstimationMapper(
            onnx_det_model=str(self.models / "yolox_l.onnx"),
            onnx_pose_model=str(self.models / "dw-ll_ucoco_384.onnx"),
            if_save_visualization=False,
            **self.common,
        )
        tags = self.predict(op)
        self.assert_mapping(tags)
        self.assertTrue(all(len(frame) > 0 for frame in tags["body_keypoints"]))
        model = get_model(op.model_key, 0, True)
        for session in [model.pose_estimation.session_det, model.pose_estimation.session_pose]:
            self.assertEqual(session.get_providers()[0], "CUDAExecutionProvider")

    def test_depth(self):
        op = VideoDepthEstimationMapper(
            video_depth_model_path=str(self.models / "video_depth_anything_vits.pth"),
            max_res=256,
            if_save_visualization=False,
            if_save_point_cloud=False,
            **self.common,
        )
        tags = self.predict(op)
        self.assert_mapping(tags)
        self.assert_cuda(get_model(op.model_key, 0, True))
        self.assertEqual(len(tags["depth_data"]), 2)
        self.assertTrue(np.isfinite(tags["depth_data"]).all())

    def test_moge(self):
        op = VideoCameraCalibrationMogeMapper(
            model_path=str(self.models / "moge-model.pt"),
            output_points=False,
            output_depth=False,
            output_mask=False,
            frame_batch_size=1,
            **self.common,
        )
        tags = self.predict(op)
        self.assert_cuda(get_model(op.model_key, 0, True))
        self.assertEqual(len(tags), 2)
        self.assertEqual(len(tags[1][CameraCalibrationKeys.hfov]), 2)
        self.assertTrue(np.isfinite(tags[1][CameraCalibrationKeys.intrinsics]).all())

    @unittest.skipUnless(os.environ.get("DJ_RUN_RAY_TESTS") == "1", "Enable Ray tests for the CUDA actor check.")
    def test_ray_gpu_actor(self):
        import ray

        from data_juicer.core.data.ray_dataset import RayDataset

        started_ray = not ray.is_initialized()
        if started_ray:
            ray.init(
                num_cpus=2,
                num_gpus=1,
                include_dashboard=False,
                object_store_memory=200 * 1024 * 1024,
                runtime_env={"env_vars": {"RAY_JOB": "1"}},
            )
            self.addCleanup(ray.shutdown)
        op = CudaVerifiedBodyMapper(
            onnx_det_model=str(self.models / "yolox_l.onnx"),
            onnx_pose_model=str(self.models / "dw-ll_ucoco_384.onnx"),
            if_save_visualization=False,
            num_gpus=1,
            num_cpus=1,
            ray_execution_mode="actor",
            **self.common,
        )
        rows = [self.sample(), {MetaKeys.video_frames: [], Fields.meta: {}, "fps": 24}]
        dataset = RayDataset(ray.data.from_items(rows, override_num_blocks=2), auto_op_parallelism=False)
        result = dataset.process(op).get(2)
        nonempty = next(row for row in result if row["cuda_providers"])
        self.assertNotEqual(nonempty["worker_pid"], os.getpid())
        self.assertEqual(nonempty["cuda_providers"][0], "CUDAExecutionProvider")
        self.assert_mapping(nonempty[Fields.meta][op.tag_field_name])
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
