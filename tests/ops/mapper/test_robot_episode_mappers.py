import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from data_juicer.core.data import NestedDataset
from data_juicer.ops.mapper.robot_depth_estimation_mapper import (
    RobotDepthEstimationMapper,
)
from data_juicer.ops.mapper.robot_destination_grounding_mapper import (
    RobotDestinationGroundingMapper,
)
from data_juicer.ops.mapper.robot_gripper_anchor_mapper import RobotGripperAnchorMapper
from data_juicer.ops.mapper.robot_gripper_tracking_mapper import (
    RobotGripperTrackingMapper,
)
from data_juicer.ops.mapper.robot_interaction_geometry_mapper import (
    RobotInteractionGeometryMapper,
)
from data_juicer.ops.mapper.robot_object_tracking_mapper import (
    RobotObjectTrackingMapper,
)
from data_juicer.utils import model_utils
from data_juicer.utils import robot_perception_utils as perception
from data_juicer.utils import robot_utils as robot
from data_juicer.utils.constant import Fields


class RobotEpisodeMappersTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "output"
        metadata = self.root / "metadata.parquet"
        pq.write_table(pa.table({"other_information.language_instruction_2": ["pick up the cup"]}), metadata)
        states = self.root / "states.parquet"
        pq.write_table(
            pa.table(
                {
                    "frame_index": list(range(8)),
                    "episode_index": [7] * 8,
                    "observation.state.gripper_position": [1.0, 1.0, 0.8, 0.2, 0.0, 0.2, 0.8, 1.0],
                }
            ),
            states,
        )
        self.sample = {
            "source": "droid",
            "episode_id": "episode-7",
            "episode_index": 7,
            "metadata_path": str(metadata),
            "state_path": str(states),
            "raw_dir": str(self.root / "raw"),
            "primary_view": "left",
            Fields.meta: {},
        }
        self.views = {}
        for camera in ("left", "right"):
            images = []
            for i in range(8):
                path = self.output / "droid" / "episode-7" / "frames" / camera / f"frame_{i:02d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (i * 20, 20, 40)).save(path)
                images.append(str(path))
            self.views[camera] = {
                "images": images,
                "frame_indices": list(range(8)),
                "source_frame_count": 8,
                "source_fps": 10.0,
                "sampled_video": str(self.root / camera / "sampled.mp4"),
                "image_size": {"width": 32, "height": 32},
            }
        self.mask = np.zeros((32, 32), dtype=bool)
        self.mask[8:24, 8:24] = True

    def _op(self, cls, **kwargs):
        return cls(output_dir=str(self.output), accelerator="cpu", num_proc=1, **kwargs)

    def _geometry_inputs(self):
        sample = self._op(RobotGripperAnchorMapper).process_single(copy.deepcopy(self.sample))
        root = Path(sample["episode_dir"])
        robot.write_json(root / "ir/sampling.json", {"views": copy.deepcopy(self.views)})
        detection = root / "object.json"
        tracking = root / "object-tracking.json"
        gripper = root / "gripper.json"
        robot.write_json(detection, {"vlm_target_object": {"target_object": "cup"}})
        robot.write_json(tracking, {"frames": [{"frame_index": i, "bbox_xyxy": [0, 0, 10, 10]} for i in range(8)]})
        robot.write_json(
            gripper,
            {
                "tracking": {
                    "frames": [
                        {"frame_index": i, "bbox_xyxy": [5, 0, 15, 10] if i == 4 else [20, 20, 30, 30]}
                        for i in range(8)
                    ]
                }
            },
        )
        robot.write_json(root / "ir/destination.json", {"sample_index": 7, "view": "left", "point": [500, 500]})
        sample.update(object_outputs=[str(detection), str(tracking)], gripper_output=str(gripper))
        return sample

    def test_geometry_uses_joint_tracks_and_exports_json(self):
        sample = self._geometry_inputs()
        result = self._op(RobotInteractionGeometryMapper).process_single(sample)
        ir = robot.read_json(Path(result["episode_dir"]) / "ir/episode_ir.json")
        self.assertEqual(ir["geometry"]["contact_sample_index"], 4)
        self.assertEqual(ir["geometry"]["contact_overlap_ratio"], 0.5)
        self.assertEqual(ir["geometry"]["destination_sample_index"], 7)
        self.assertEqual(ir["geometry"]["secondary_view"], "right")
        self.assertEqual(ir["instruction"], "pick up the cup")

    def test_geometry_rejects_unsynchronized_views(self):
        sample = self._geometry_inputs()
        root = Path(sample["episode_dir"])
        views = copy.deepcopy(self.views)
        views["right"]["frame_indices"][-1] = 9
        robot.write_json(root / "ir/sampling.json", {"views": views})
        with self.assertRaisesRegex(ValueError, "not synchronized"):
            self._op(RobotInteractionGeometryMapper).process_single(sample)
        self.assertFalse((root / "ir/episode_ir.json").exists())

    def test_geometry_rejects_missing_joint_tracks_and_contact(self):
        sample = self._geometry_inputs()
        for frames, message in (
            ([{"frame_index": 0, "bbox_xyxy": [0, 0, 1, 1]}], "Insufficient joint tracks"),
            ([{"frame_index": i, "bbox_xyxy": [20, 20, 30, 30]} for i in range(8)], "No geometric contact"),
        ):
            with self.subTest(message=message):
                robot.write_json(sample["gripper_output"], {"tracking": {"frames": frames}})
                with self.assertRaisesRegex(ValueError, message):
                    self._op(RobotInteractionGeometryMapper).process_single(sample)

    def test_depth_rejects_invalid_configuration_before_loading(self):
        for options in ({"inference_batch_size": -1}, {"process_res": 13}, {"process_res_method": "crop"}):
            with (
                self.subTest(options=options),
                patch("data_juicer.ops.mapper.robot_depth_estimation_mapper.prepare_model") as prepare,
                self.assertRaises(ValueError),
            ):
                self._op(RobotDepthEstimationMapper, **options)
            prepare.assert_not_called()

    def test_depth_requires_dense_source_frames(self):
        sample = self._geometry_inputs()
        views = copy.deepcopy(self.views)
        views["left"]["frame_indices"] = [0, 2, 4, 6]
        robot.write_json(Path(sample["episode_dir"]) / "ir/sampling.json", {"views": views})
        op = self._op(RobotDepthEstimationMapper)
        model = Mock(model_name="da3metric-large")
        with patch("data_juicer.ops.mapper.robot_depth_estimation_mapper.get_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "every source frame"):
                op.process_single(sample)
        model.inference.assert_not_called()

    def test_depth_any_view_requires_whole_view_inference(self):
        sample = self._geometry_inputs()
        op = self._op(RobotDepthEstimationMapper)
        model = Mock(model_name="da3-large")
        with patch("data_juicer.ops.mapper.robot_depth_estimation_mapper.get_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "inference_batch_size=0"):
                op.process_single(sample)
        model.inference.assert_not_called()

    def test_depth_mapper_can_run_twice_after_models_are_freed(self):
        sample = self._geometry_inputs()
        loaded = []

        class DepthModel:
            model_name = "da3metric-large"

            def __init__(self, device):
                self.device = device

            def to(self, device):
                self.device = device
                return self

            def inference(self, image, **kwargs):
                if self.device != "cuda:0":
                    raise RuntimeError("Depth model was offloaded to CPU")
                return SimpleNamespace(depth=np.full((len(image), 4, 4), len(loaded)), is_metric=False)

        def prepare(device, **kwargs):
            model = DepthModel(device)
            loaded.append(model)
            return model

        with (
            patch.dict(model_utils.MODEL_FUNCTION_MAPPING, {"robot_depth": prepare}),
            patch.dict(model_utils.MODEL_ZOO, {}, clear=True),
            patch.object(model_utils, "cuda_device_count", return_value=1),
            patch("data_juicer.ops.base_op.is_cuda_available", return_value=True),
        ):
            op = RobotDepthEstimationMapper(
                output_dir=str(self.output), accelerator="cuda", num_proc=None, auto_op_parallelism=False
            )
            dataset = NestedDataset.from_list([sample])
            for run in range(2):
                result = op.run(dataset).to_list()[0]
                self.assertEqual(len(loaded), run + 1)
                self.assertEqual(loaded[-1].device, "cpu")
                self.assertFalse(model_utils.MODEL_ZOO)
                manifest = robot.read_json(result["depth_ir"])
                records = manifest["views"]["left"]["frames"]
                self.assertEqual(len(records), 8)
                with np.load(Path(result["episode_dir"]) / records[0]["path"]) as prediction:
                    np.testing.assert_array_equal(prediction["depth"], np.full((4, 4), run + 1))

    def _check_six_stage_pipeline(self, through_dataset):
        def read_frame(path, index):
            return np.asarray(Image.open(self.views["left"]["images"][index]))

        def chat(client, messages, **kwargs):
            prompt = messages[-1]["content"][0]["text"]
            if "selected_candidate" in prompt:
                result = {"selected_candidate": 1, "reason": "visible gripper"}
            elif "image_index" in prompt:
                result = {"image_index": 1}
            elif "target_object" in prompt:
                result = {"target_object": "cup", "referring_expression": "the cup"}
            else:
                result = {"point_2d": [500, 500], "evidence": "visible object"}
            return {"content": json.dumps(result)}

        processor = Mock(return_value={"pixel_values": torch.zeros((1, 3, 32, 32))})
        processor.post_process_instance_segmentation.return_value = [
            {"masks": self.mask[None], "scores": np.array([0.9]), "boxes": np.array([[8, 8, 24, 24]])}
        ]
        model = Mock()

        def track(*args, **kwargs):
            return (
                [{"frame_index": i, "bbox_xyxy": [8, 8, 24, 24], "mask": self.mask.copy()} for i in range(8)],
                {"tracking_prompt_frame_idx": kwargs.get("prompt_frame_idx", 0)},
            )

        depth_model = Mock(model_name="da3metric-large")
        depth_model.inference.side_effect = lambda image, **kw: SimpleNamespace(
            depth=np.full((len(image), 4, 4), 2.0), is_metric=False
        )
        depth_op = self._op(RobotDepthEstimationMapper, inference_batch_size=3)
        ops = [
            self._op(RobotGripperAnchorMapper),
            self._op(RobotObjectTrackingMapper, model="test-vlm"),
            self._op(RobotGripperTrackingMapper, model="test-vlm"),
            self._op(RobotDestinationGroundingMapper, model="test-vlm"),
            depth_op,
            self._op(RobotInteractionGeometryMapper),
        ]
        with (
            patch.object(robot, "decode_camera_views", return_value=copy.deepcopy(self.views)),
            patch.object(robot, "read_video_frame", side_effect=lambda reader, path, index: read_frame(path, index)),
            patch.object(robot, "video_frame_count", return_value=8),
            patch.object(perception.VLLMChatClient, "chat", chat),
            patch.object(perception.sam3, "load_sam3", return_value=(model, processor, self.root)),
            patch.object(
                perception.sam3, "load_video_tracker", return_value=(model, processor, {"model": "mock-sam3"})
            ),
            patch.object(perception.sam3, "track_video", side_effect=track) as tracker,
            patch("data_juicer.ops.mapper.robot_depth_estimation_mapper.get_model", return_value=depth_model) as load,
        ):
            sample = copy.deepcopy(self.sample)
            if through_dataset:
                dataset = NestedDataset.from_list([sample])
                for op in ops:
                    dataset = op.run(dataset)
                sample = dataset.to_list()[0]
            else:
                for op in ops:
                    sample = op.process_single(sample)
            if not through_dataset:
                self.assertEqual(tracker.call_count, 2)
                load.assert_called_once_with(depth_op.model_key, rank=None, use_cuda=False)
        root = Path(sample["episode_dir"])
        ir = robot.read_json(root / "ir/episode_ir.json")
        self.assertEqual(ir["instruction"], "pick up the cup")
        self.assertEqual(ir["geometry"]["destination_sample_index"], 7)
        self.assertEqual(len(ir["geometry"]["tracked_depth"]), 8)
        self.assertTrue(all(r["object"] == 2 and r["gripper"] == 2 for r in ir["geometry"]["tracked_depth"]))
        self.assertEqual(len(list((root / "ir/da3_depth").glob("*/*.npz"))), 16)
        if not through_dataset:
            self.assertEqual([len(c.kwargs["image"]) for c in depth_model.inference.call_args_list], [3, 3, 2, 3, 3, 2])
        pa.Table.from_pylist([sample])
        json.dumps(ir, allow_nan=False)

    def test_six_stage_pipeline_with_mocked_vlm_and_sam3(self):
        self._check_six_stage_pipeline(through_dataset=False)

    def test_six_stage_pipeline_through_dataset_mapping(self):
        self._check_six_stage_pipeline(through_dataset=True)


if __name__ == "__main__":
    unittest.main()
