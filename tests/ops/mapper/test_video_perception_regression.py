import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from PIL import Image

from data_juicer.core.data import NestedDataset
from data_juicer.ops.load import load_ops
from data_juicer.ops.mapper import video_normal_map_mapper as normal
from data_juicer.ops.mapper import video_optical_flow_mapper as flow
from data_juicer.ops.mapper import video_universal_segmentation_mapper as segmentation
from data_juicer.ops.op_env import OPEnvManager
from data_juicer.utils.constant import Fields, MetaKeys


class VideoPerceptionRegressionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.paths = []
        for name, color in [("frame_0.png", (255, 0, 0)), ("frame_2.png", (0, 255, 0)), ("frame_10.png", (0, 0, 255))]:
            path = self.root / name
            Image.new("RGB", (8, 8), color).save(path)
            self.paths.append(str(path))

    def make_op(self, module, **kwargs):
        cls = {
            "normal": normal.VideoNormalMapMapper,
            "flow": flow.VideoOpticalFlowMapper,
            "segmentation": segmentation.VideoUniversalSegmentationMapper,
        }[module]
        target = {"normal": normal, "flow": flow, "segmentation": segmentation}[module]
        with (
            patch.object(target, "prepare_model", return_value="offline"),
            patch.object(target, "LazyLoader"),
            patch.object(target, "load_ops") as load,
        ):
            op = cls(accelerator="cpu", if_save_visualization=False, frame_dir=str(self.root), **kwargs)
        op.input_size = (8, 8)
        return op, load

    def session(self):
        session = Mock()
        session.run.return_value = [np.zeros((1, 1, 8, 8), np.float32), np.ones((1, 3, 8, 8), np.float32)]
        return session

    def test_normal_loads_with_environment_management_in_cpu_mode(self):
        runtime = Mock()
        runtime.get_available_providers.return_value = ["CPUExecutionProvider"]
        with (
            patch.object(normal, "LazyLoader", return_value=runtime) as lazy,
            patch.object(normal, "prepare_model", return_value="offline"),
            patch.object(normal, "load_ops", return_value=[Mock()]),
        ):
            ops = load_ops(
                [{"video_normal_map_mapper": {"accelerator": "cpu", "frame_dir": str(self.root)}}],
                op_env_manager=OPEnvManager(),
            )
        self.assertIn("onnxruntime", ops[0].runtime_env["uv"])
        self.assertNotIn("onnxruntime-gpu", ops[0].runtime_env["uv"])
        lazy.assert_called_once_with("onnxruntime")

    def test_normal_cuda_environment_requires_only_gpu_runtime(self):
        runtime = Mock()
        runtime.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with (
            patch.object(normal, "LazyLoader", return_value=runtime) as lazy,
            patch.object(normal, "prepare_model", return_value="offline"),
            patch.object(normal, "load_ops", return_value=[Mock()]),
        ):
            ops = load_ops(
                [{"video_normal_map_mapper": {"accelerator": "cuda", "frame_dir": str(self.root)}}],
                op_env_manager=OPEnvManager(),
            )
        self.assertIn("onnxruntime-gpu", ops[0].runtime_env["uv"])
        self.assertNotIn("onnxruntime", ops[0].runtime_env["uv"])
        lazy.assert_called_once_with("onnxruntime", package_name="onnxruntime-gpu")

    def test_normal_rejects_cuda_with_cpu_only_runtime(self):
        runtime = Mock()
        runtime.get_available_providers.return_value = ["CPUExecutionProvider"]
        with (
            patch.object(normal, "LazyLoader", return_value=runtime),
            patch.object(normal, "prepare_model") as prepare,
            patch.object(normal.VideoNormalMapMapper, "use_cuda", return_value=True),
            self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"),
        ):
            normal.VideoNormalMapMapper(accelerator="cuda")
        prepare.assert_not_called()

    def test_empty_frames_do_not_load_models(self):
        for name, module in [("normal", normal), ("flow", flow), ("segmentation", segmentation)]:
            with self.subTest(operator=name):
                op, _ = self.make_op(name)
                with patch.object(module, "get_model") as load:
                    result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: []})
                load.assert_not_called()
                self.assertTrue(all(not value for value in result[Fields.meta][op.tag_field_name].values()))

    def test_normal_results_are_json_serializable(self):
        op, _ = self.make_op("normal")
        with patch.object(normal, "get_model", return_value=self.session()):
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: self.paths[:1]})
        encoded = json.loads(json.dumps(result))
        self.assertEqual(np.asarray(encoded[Fields.meta][op.tag_field_name]["pred_norm"]).shape, (1, 8, 8, 3))

    def test_nested_frame_groups_preserve_order(self):
        op, _ = self.make_op("normal")
        session = self.session()
        groups = [[self.paths[2], self.paths[0]], [self.paths[1]]]
        with patch.object(normal, "get_model", return_value=session):
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: groups})
        tags = result[Fields.meta][op.tag_field_name]
        self.assertEqual(tags["frame_paths"], [self.paths[2], self.paths[0], self.paths[1]])
        self.assertEqual(tags["video_indices"], [0, 0, 1])
        self.assertEqual(len(tags["pred_norm"]), 3)

    def test_raw_video_uses_returned_frames_and_custom_key(self):
        op, load = self.make_op("normal", video_key="clips")
        extractor = load.return_value[0]
        extractor.process_single.return_value = {MetaKeys.video_frames: [[self.paths[2], self.paths[0]]]}
        with patch.object(normal, "get_model", return_value=self.session()):
            result = op.process_single({Fields.meta: {}, "clips": ["/different/location/input.mp4"]})
        self.assertEqual(result[Fields.meta][op.tag_field_name]["frame_paths"], [self.paths[2], self.paths[0]])
        self.assertEqual(extractor.process_single.call_args.args[0]["clips"], ["/different/location/input.mp4"])

    def test_unreadable_frame_fails_with_path(self):
        path = str(self.root / "missing.png")
        op, _ = self.make_op("normal")
        with patch.object(normal, "get_model", return_value=self.session()):
            with self.assertRaisesRegex(ValueError, "missing.png"):
                op.process_single({Fields.meta: {}, MetaKeys.video_frames: [path]})

    def test_flow_uses_cpu_rgb_and_keeps_video_boundaries(self):
        op, _ = self.make_op("flow")
        seen = []

        def predict(first, second):
            self.assertEqual(first.device.type, "cpu")
            seen.append((first[0, :, 0, 0].tolist(), second[0, :, 0, 0].tolist()))
            return [torch.ones((first.shape[0], 2, 8, 8))]

        model = Mock(side_effect=predict)
        model.parameters.side_effect = lambda: iter([torch.zeros(1)])
        op.raft_preprocess = lambda first, second, transform: (first, second)
        groups = [self.paths[:2], [self.paths[2]]]
        with patch.object(flow, "get_model", return_value=(model, None)):
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: groups}, rank=7)
        self.assertEqual(seen, [([255, 0, 0], [0, 255, 0])])
        tags = json.loads(json.dumps(result))[Fields.meta][op.tag_field_name]
        self.assertEqual(tags["frame_pairs"], [self.paths[:2]])
        self.assertEqual(tags["video_indices"], [0])

    def test_disabled_segmentation_does_not_load_model(self):
        op, _ = self.make_op(
            "segmentation",
            if_output_semantic_segmentation=False,
            if_output_instance_segmentation=False,
            if_output_panoptic_segmentation=False,
        )
        with patch.object(segmentation, "get_model") as load:
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: self.paths})
        load.assert_not_called()
        self.assertEqual(result[Fields.meta][op.tag_field_name]["semantic_segmentation_map"], [])

    def test_segmentation_cpu_outputs_and_jsonl_export(self):
        op, _ = self.make_op("segmentation")
        devices = []

        class Inputs(dict):
            def to(self, device):
                devices.append(str(device))
                return self

        processor = Mock(side_effect=lambda **kwargs: Inputs(pixel_values=torch.zeros(1, 3, 8, 8)))
        processor.post_process_semantic_segmentation.return_value = [torch.zeros(8, 8, dtype=torch.int64)]
        for task in ["instance", "panoptic"]:
            getattr(processor, f"post_process_{task}_segmentation").return_value = [
                {
                    "segmentation": torch.ones(8, 8, dtype=torch.int64),
                    "segments_info": [{"id": 1, "label_id": 0, "score": 0.75}],
                }
            ]
        model = Mock(device=torch.device("cpu"))
        with patch.object(segmentation, "get_model", return_value=(model, processor)):
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: self.paths[:1]}, rank=7)
        self.assertEqual(devices, ["cpu"] * 3)
        tags = result[Fields.meta][op.tag_field_name]
        self.assertEqual(tags["instance_segmentation_info"][0][0]["id"], 1)
        json.dumps(result)
        output = self.root / "result.jsonl"
        NestedDataset.from_list([copy.deepcopy(result)]).to_json(str(output))
        exported = json.loads(output.read_text())
        self.assertEqual(
            exported[Fields.meta][op.tag_field_name]["semantic_segmentation_map"], tags["semantic_segmentation_map"]
        )

    def make_videos(self):
        paths = []
        for index in range(2):
            path = self.root / f"clip_{index}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (64, 48))
            self.assertTrue(writer.isOpened())
            for frame in range(12):
                writer.write(np.full((48, 64, 3), (index * 80, frame * 15, 40), dtype=np.uint8))
            writer.release()
            paths.append(str(path))
        return paths

    def test_real_video_extraction_and_dataset_export(self):
        videos = self.make_videos()
        frame_dir = self.root / "extracted"
        stale_dir = frame_dir / "clip_0"
        stale_dir.mkdir(parents=True)
        stale = stale_dir / "frame_99.jpg"
        Image.new("RGB", (64, 48)).save(stale)
        with patch.object(normal, "prepare_model", return_value="offline"), patch.object(normal, "LazyLoader"):
            op = normal.VideoNormalMapMapper(
                accelerator="cpu",
                video_key="clips",
                frame_num=2,
                frame_dir=str(frame_dir),
                if_save_visualization=False,
                num_proc=None,
                auto_op_parallelism=False,
            )
        op.input_size = (8, 8)
        session = self.session()
        with patch.object(normal, "get_model", return_value=session):
            dataset = op.run(NestedDataset.from_list([{"clips": videos}]))
        output = self.root / "video-result.jsonl"
        dataset.to_json(str(output))
        metadata = json.loads(output.read_text())[Fields.meta]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        tags = metadata[op.tag_field_name]
        self.assertEqual(tags["video_indices"], [0, 0, 1, 1])
        self.assertEqual(session.run.call_count, 4)
        self.assertNotIn(str(stale), tags["frame_paths"])
        self.assertTrue(all(Path(path).is_file() for path in tags["frame_paths"]))
        self.assertEqual(np.asarray(tags["pred_norm"]).shape, (4, 48, 64, 3))

    def test_real_video_extraction_does_not_join_flow_between_videos(self):
        videos = self.make_videos()
        with (
            patch.object(flow, "prepare_model", return_value="offline"),
            patch.object(flow.LazyLoader, "check_packages"),
        ):
            op = flow.VideoOpticalFlowMapper(
                accelerator="cpu",
                video_key="clips",
                frame_num=2,
                frame_dir=str(self.root / "extracted"),
                if_save_visualization=False,
                num_proc=None,
                auto_op_parallelism=False,
            )
        op.raft_preprocess = lambda first, second, transforms: (first, second)
        model = Mock(return_value=[torch.zeros(1, 2, 48, 64)])
        model.parameters.side_effect = lambda: iter([torch.zeros(1)])
        with patch.object(flow, "get_model", return_value=(model, None)):
            result = op.process_single({"clips": videos})
        tags = result[Fields.meta][op.tag_field_name]
        self.assertEqual(tags["video_indices"], [0, 1])
        self.assertEqual(len(tags["frame_pairs"]), 2)
        self.assertEqual(model.call_count, 2)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
