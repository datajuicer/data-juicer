import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from PIL import Image

from data_juicer.core.data import NestedDataset
from data_juicer.ops.mapper import video_human_3d_pose_mapper as human
from data_juicer.utils.constant import Fields, MetaKeys


class Human3RRegressionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.paths = []
        for i in [0, 2, 10]:
            path = self.root / f"frame_{i}.png"
            Image.new("RGB", (64, 48), (i, 0, 0)).save(path)
            self.paths.append(str(path))

    def op(self, **kwargs):
        with (
            patch.object(human, "prepare_model", return_value="offline"),
            patch.object(human.LazyLoader, "check_packages"),
            patch.object(human, "load_ops") as load,
        ):
            op = human.VideoHuman3DPoseMapper(accelerator="cpu", frame_dir=str(self.root), **kwargs)
        return op, load

    def predict(self, views, model, device, **kwargs):
        self.assertEqual(str(device), "cpu")
        pred = []
        for view in views:
            pred.append(
                {
                    name: torch.ones(shape)
                    for name, shape in {
                        "camera_pose": (1, 7),
                        "conf": (1, 4, 4),
                        "smpl_shape": (1, 1, 10),
                        "smpl_transl": (1, 1, 3),
                        "smpl_rotmat": (1, 1, 53, 3, 3),
                        "smpl_expression": (1, 1, 10),
                        "smpl_id": (1, 1),
                        "smpl_loc": (1, 1, 2),
                    }.items()
                }
            )
        return {"pred": pred}, None

    def test_empty_frames_do_not_load_models(self):
        op, _ = self.op()
        with patch.object(human, "get_model") as load:
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: []})
        load.assert_not_called()
        self.assertEqual(result[Fields.meta][op.tag_field_name]["valid_frame_list"], [])

    def test_uses_exact_frame_subset_and_order(self):
        op, _ = self.op()
        requested = [self.paths[2], self.paths[0]]
        with (
            patch.object(
                human,
                "get_model",
                return_value=(SimpleNamespace(mhmr_img_res=None, device=torch.device("cpu")), self.predict),
            ),
            patch.object(op, "prepare_input", wraps=op.prepare_input) as prepare,
        ):
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: requested}, rank=7)
        self.assertEqual(prepare.call_args.kwargs["img_paths"], requested)
        self.assertEqual(result[Fields.meta][op.tag_field_name]["valid_frame_list"], [0, 1])

    def test_json_serializable_pose_and_export(self):
        op, _ = self.op()
        with patch.object(
            human,
            "get_model",
            return_value=(SimpleNamespace(mhmr_img_res=None, device=torch.device("cpu")), self.predict),
        ):
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: self.paths[:1]})
        json.dumps(result)
        self.assertEqual(np.asarray(result[Fields.meta][op.tag_field_name]["smpl_shape"]).shape, (1, 1, 10))
        output = self.root / "poses.jsonl"
        NestedDataset.from_list([copy.deepcopy(result)]).to_json(str(output))
        self.assertEqual(json.loads(output.read_text())[Fields.meta][op.tag_field_name]["valid_frame_list"], [0])

    def test_multiple_videos_have_independent_inference(self):
        op, _ = self.op()
        predict = Mock(side_effect=self.predict)
        with patch.object(
            human, "get_model", return_value=(SimpleNamespace(mhmr_img_res=None, device=torch.device("cpu")), predict)
        ):
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: [self.paths[:2], self.paths[2:]]})
        self.assertEqual([len(c.args[0]) for c in predict.call_args_list], [2, 1])
        tags = result[Fields.meta][op.tag_field_name]
        self.assertEqual(tags["video_indices"], [0, 0, 1])
        self.assertEqual(tags["frame_paths"], self.paths)
        self.assertEqual(tags["valid_frame_list"], [0, 1, 2])

    def test_reset_overlap_does_not_duplicate_frame_results(self):
        op, _ = self.op()
        paths = [self.paths[0]] * 101
        with patch.object(
            human,
            "get_model",
            return_value=(SimpleNamespace(mhmr_img_res=None, device=torch.device("cpu")), self.predict),
        ):
            result = op.process_single({Fields.meta: {}, MetaKeys.video_frames: paths})
        self.assertEqual(result[Fields.meta][op.tag_field_name]["valid_frame_list"], list(range(101)))

    def test_raw_video_respects_returned_frame_paths(self):
        op, load = self.op(video_key="clips")
        load.return_value[0].process_single.return_value = {MetaKeys.video_frames: [self.paths[:1]]}
        with patch.object(
            human,
            "get_model",
            return_value=(SimpleNamespace(mhmr_img_res=None, device=torch.device("cpu")), self.predict),
        ):
            result = op.process_single({Fields.meta: {}, "clips": ["/other/path/person.mp4"]})
        self.assertEqual(result[Fields.meta][op.tag_field_name]["frame_paths"], self.paths[:1])

    def test_real_video_extraction_and_dataset_export(self):
        videos = []
        for index in range(2):
            path = self.root / f"person_{index}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6, (64, 48))
            self.assertTrue(writer.isOpened())
            for frame in range(12):
                writer.write(np.full((48, 64, 3), (index * 80, frame * 15, 40), dtype=np.uint8))
            writer.release()
            videos.append(str(path))
        with (
            patch.object(human, "prepare_model", return_value="offline"),
            patch.object(human.LazyLoader, "check_packages"),
        ):
            op = human.VideoHuman3DPoseMapper(
                accelerator="cpu",
                video_key="clips",
                frame_num=2,
                frame_dir=str(self.root / "extracted"),
                num_proc=None,
                auto_op_parallelism=False,
            )
        inference = Mock(side_effect=self.predict)
        model = SimpleNamespace(mhmr_img_res=None, device=torch.device("cpu"))
        with patch.object(human, "get_model", return_value=(model, inference)):
            dataset = op.run(NestedDataset.from_list([{"clips": videos}]))
        output = self.root / "video-poses.jsonl"
        dataset.to_json(str(output))
        metadata = json.loads(output.read_text())[Fields.meta]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        tags = metadata[op.tag_field_name]
        self.assertEqual(tags["video_indices"], [0, 0, 1, 1])
        self.assertEqual(tags["valid_frame_list"], [0, 1, 2, 3])
        self.assertEqual([len(call.args[0]) for call in inference.call_args_list], [2, 2])
        self.assertTrue(all(Path(path).is_file() for path in tags["frame_paths"]))

    def test_frames_without_people_keep_original_indices(self):
        op, _ = self.op()

        def predict(views, model, device, **kwargs):
            output, state = self.predict(views, model, device, **kwargs)
            output["pred"][0].pop("smpl_shape")
            return output, state

        model = SimpleNamespace(mhmr_img_res=None, device=torch.device("cpu"))
        with patch.object(human, "get_model", return_value=(model, predict)):
            result = op.process_single({MetaKeys.video_frames: [self.paths[:2], self.paths[2:]]})
        tags = result[Fields.meta][op.tag_field_name]
        self.assertEqual(tags["valid_frame_list"], [1])
        self.assertEqual(tags["frame_paths"], [self.paths[1]])
        self.assertEqual(tags["video_indices"], [0])


if __name__ == "__main__":
    unittest.main()
