"""Offline regressions for ordered video frames and per-video results."""

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
from data_juicer.ops.base_op import Mapper
from data_juicer.ops.mapper import video_animal_pose_mapper as animal
from data_juicer.ops.mapper import video_camera_calibration_moge_mapper as moge
from data_juicer.ops.mapper import video_depth_estimation_mapper as depth
from data_juicer.ops.mapper import video_face_keypoints_mapper as face
from data_juicer.ops.mapper import video_hand_reconstruction_hawor_mapper as hawor
from data_juicer.ops.mapper import video_whole_body_pose_estimation_mapper as body
from data_juicer.ops.mapper.video_extract_frames_mapper import VideoExtractFramesMapper
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys


class VideoFrameSequenceContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = []
        for index in [0, 2, 10]:
            path = self.root / f"frame_{index}.png"
            Image.new("RGB", (32, 24), (index, index, index)).save(path)
            self.paths.append(str(path))

    def op(self, module, **kwargs):
        classes = {
            animal: animal.VideoAnimalPoseMapper,
            face: face.VideoFaceKeypointsMapper,
            body: body.VideoWholeBodyPoseEstimationMapper,
            depth: depth.VideoDepthEstimationMapper,
            moge: moge.VideoCameraCalibrationMogeMapper,
            hawor: hawor.VideoHandReconstructionHaworMapper,
        }
        operator = classes[module].__new__(classes[module])
        Mapper.__init__(operator, accelerator="cpu", num_proc=1, auto_op_parallelism=False)
        operator.frame_field = MetaKeys.video_frames
        operator.tag_field_name = "predictions"
        operator.model_key = "offline"
        operator.fused_ops = [Mock()]
        operator.if_save_visualization = False
        operator.save_visualization_dir = str(self.root / "visualizations")
        operator.frame_dir = str(self.root)
        if module is animal:
            operator.yolo_model_key = "detector"
            operator.animal_class = ["cat"]
            operator.inference_top_down_pose_model = Mock()
        if module is depth:
            operator.max_res = -1
            operator.torch_dtype = "fp32"
            operator.metric = False
            operator.if_save_point_cloud = False
            operator.save_video = Mock()
            operator.read_video_frames = Mock()
            operator.grayscale = False
        if module is moge:
            for name in ["intrinsics", "hfov", "vfov", "points", "depth", "mask"]:
                setattr(operator, "output_" + name, name in ["intrinsics", "hfov"])
            operator.frame_batch_size = 8
            operator.save_dir = None
        if module is hawor:
            operator.camera_calibration_field = "calibration"
            operator.det_model_key = "detector"
            operator.thresh = 0.2
            operator.detect_track = Mock(return_value=([], {}))
            operator.hawor_motion_estimation = Mock(return_value={"left": {}, "right": {}})
        for name, value in kwargs.items():
            setattr(operator, name, value)
        return operator

    def sample(self, frames):
        return {MetaKeys.video_frames: frames, Fields.meta: {}}

    def face_model(self):
        detector = SimpleNamespace(get=Mock(return_value=[SimpleNamespace(bbox=np.array([4, 4, 20, 20]))]))
        model = Mock(return_value={"keypoints": torch.full((1, 98, 2), 0.5)})
        args = SimpleNamespace(model_mode="offline", z_width=1, heatmap_size=4)
        return model, detector, args

    def body_model(self, image):
        marker = float(image[0, 0, 0])
        return tuple(np.array([[[marker, marker]]]) for _ in range(4)) + (
            np.array([[marker] * 4]),
            np.zeros_like(image),
        )

    def animal_models(self, operator):
        detector = Mock()
        detector.predict.return_value = [
            SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([[1.0, 2.0, 20.0, 21.0]]))])
        ]

        def predict(model, image, boxes, **kwargs):
            marker = float(image[0, 0, 0])
            return [{"keypoints": np.full((17, 3), marker), "bbox": [1, 2, 20, 21]}], None

        operator.inference_top_down_pose_model.side_effect = predict
        return [Mock(), detector]

    def depth_model(self):
        model = Mock()
        model.infer_video_depth.side_effect = lambda frames, fps, **kwargs: (frames[..., 0].astype(float), fps)
        return model

    def moge_model(self):
        model = Mock()

        def infer(batch):
            values = batch[:, 0, 0, 0]
            intrinsics = torch.eye(3).repeat(len(batch), 1, 1)
            intrinsics[:, 0, 0] = values + 1.0
            intrinsics[:, 1, 1] = 1.0
            intrinsics[:, 0, 2] = 0.5
            intrinsics[:, 1, 2] = 0.5
            return {"intrinsics": intrinsics}

        model.infer.side_effect = infer
        return model

    def test_empty_pose_and_depth_frames_do_not_load_models(self):
        for module in [animal, face, body, depth]:
            for frames in [[], [[]], [[], []]]:
                with self.subTest(module=module.OP_NAME, frames=frames):
                    operator = self.op(module)
                    with patch.object(module, "get_model") as load:
                        result = operator.process_single(self.sample(frames))
                    load.assert_not_called()
                    self.assertEqual(result[Fields.meta]["predictions"]["video_indices"], [])
                    self.assertEqual(result[Fields.meta]["predictions"]["frame_paths"], [])

    def test_empty_modern_calibration_and_hand_frames_do_not_load_models(self):
        for module in [moge, hawor]:
            for frames in [[], [[]], [[], []]]:
                with self.subTest(module=module.OP_NAME, frames=frames):
                    operator = self.op(module)
                    with patch.object(module, "get_model") as load:
                        result = operator.process_single(self.sample(frames))
                    load.assert_not_called()
                    self.assertEqual(len(result[Fields.meta]["predictions"]), len(frames))

    def test_missing_frames_do_not_load_models(self):
        for module in [animal, face, body, depth, moge, hawor]:
            with self.subTest(module=module.OP_NAME):
                operator = self.op(module)
                with patch.object(module, "get_model") as load:
                    operator.process_single({Fields.meta: {}})
                load.assert_not_called()

    def test_body_predictions_follow_explicit_subset_and_order(self):
        operator = self.op(body)
        paths = [self.paths[2], self.paths[0]]
        with patch.object(body, "get_model", return_value=self.body_model):
            result = operator.process_single(self.sample(paths))
        tags = result[Fields.meta]["predictions"]
        self.assertEqual(tags["frame_paths"], paths)
        self.assertEqual(tags["body_keypoints"], [[[[10.0, 10.0]]], [[[0.0, 0.0]]]])
        self.assertEqual(tags["video_indices"], [0, 0])
        operator.fused_ops[0].process_single.assert_not_called()
        json.dumps(result)

    def test_body_nested_groups_retain_empty_video_positions(self):
        operator = self.op(body)
        with patch.object(body, "get_model", return_value=self.body_model):
            result = operator.process_single(self.sample([[], self.paths[:2], self.paths[2:]]))
        tags = result[Fields.meta]["predictions"]
        self.assertEqual(tags["frame_paths"], self.paths)
        self.assertEqual(tags["video_indices"], [1, 1, 2])

    def test_raw_video_uses_extractor_paths_without_scanning_directory(self):
        operator = self.op(body, video_key="clips")
        paths = [self.paths[2], self.paths[0]]
        operator.fused_ops[0].process_single.return_value = {MetaKeys.video_frames: [paths]}
        with patch.object(body, "get_model", return_value=self.body_model):
            result = operator.process_single({"clips": ["unrelated-name.mp4"]})
        operator.fused_ops[0].process_single.assert_called_once_with({"clips": ["unrelated-name.mp4"]})
        self.assertEqual(result[Fields.meta]["predictions"]["frame_paths"], paths)

    def test_face_nested_frames_are_serializable_and_on_cpu(self):
        operator = self.op(face)
        models = self.face_model()
        with patch.object(face, "get_model", return_value=models):
            result = operator.process_single(self.sample([self.paths[2:], [], self.paths[:1]]), rank=7)
        tags = result[Fields.meta]["predictions"]
        self.assertEqual(tags["frame_paths"], [self.paths[2], self.paths[0]])
        self.assertEqual(tags["video_indices"], [0, 2])
        self.assertEqual(np.asarray(tags["face_keypoints"]).shape, (2, 1, 98, 2))
        self.assertEqual(models[0].call_args.args[0].device.type, "cpu")
        json.dumps(result)

    def test_cuda_worker_rank_wraps_to_visible_device(self):
        for module in [face, depth, moge]:
            with self.subTest(module=module.OP_NAME):
                operator = self.op(module)
                operator.use_cuda = Mock(return_value=True)
                sample = self.sample(self.paths[:1])
                sample["fps"] = 24
                models = self.face_model() if module is face else self.depth_model()
                input_tensor, initial_state = Mock(), Mock()
                if module is face:
                    operator.preprocess = Mock(return_value=input_tensor)
                if module is moge:
                    operator._process_video_frames_batched = Mock()
                with (
                    patch.object(torch.cuda, "device_count", return_value=1),
                    patch.object(torch, "zeros", return_value=initial_state),
                    patch.object(module, "get_model", return_value=models),
                ):
                    operator.process_single(sample, rank=7)
                if module is face:
                    input_tensor.to.assert_called_once_with("cuda:0")
                    initial_state.to.assert_called_once_with("cuda:0")
                elif module is depth:
                    self.assertEqual(models.infer_video_depth.call_args.kwargs["device"], "cuda:0")
                else:
                    self.assertEqual(operator._process_video_frames_batched.call_args.args[2], "cuda:0")

    def test_face_no_detections_keep_frame_alignment(self):
        operator = self.op(face)
        models = self.face_model()
        models[1].get.return_value = []
        with patch.object(face, "get_model", return_value=models):
            result = operator.process_single(self.sample(self.paths))
        tags = result[Fields.meta]["predictions"]
        self.assertEqual(tags["face_keypoints"], [[], [], []])
        self.assertEqual(tags["face_bboxes"], [[], [], []])
        self.assertEqual(tags["frame_paths"], self.paths)

    def test_animal_predictions_follow_group_and_frame_order(self):
        operator = self.op(animal)
        with patch.object(animal, "get_model", side_effect=self.animal_models(operator)):
            result = operator.process_single(self.sample([self.paths[2:], self.paths[:1]]))
        tags = result[Fields.meta]["predictions"]
        self.assertEqual(tags["frame_paths"], [self.paths[2], self.paths[0]])
        self.assertEqual(tags["video_indices"], [0, 1])
        self.assertEqual(tags["pose_list"][0][0][0], [10.0, 10.0])
        self.assertEqual(tags["pose_list"][1][0][0], [0.0, 0.0])
        json.dumps(result)

    def test_animal_no_detections_keep_frame_alignment(self):
        operator = self.op(animal)
        models = self.animal_models(operator)
        models[1].predict.return_value = [SimpleNamespace(boxes=[])]
        with patch.object(animal, "get_model", side_effect=models):
            result = operator.process_single(self.sample(self.paths))
        tags = result[Fields.meta]["predictions"]
        self.assertEqual(tags["pose_list"], [[], [], []])
        self.assertEqual(tags["animal_bboxes"], [[], [], []])
        self.assertEqual(tags["frame_paths"], self.paths)

    def test_pose_visualizations_do_not_collide_between_sequences(self):
        operator = self.op(body, if_save_visualization=True)
        with patch.object(body, "get_model", return_value=self.body_model):
            operator.process_single(self.sample([self.paths[:1], self.paths[:1]]))
        images = list((self.root / "visualizations").rglob("temp_frame_pose_0.jpg"))
        self.assertEqual(len(images), 2)
        self.assertNotEqual(images[0].parent, images[1].parent)

    def test_unreadable_frame_fails_instead_of_shifting_predictions(self):
        for module in [body, face, animal, depth]:
            with self.subTest(module=module.OP_NAME):
                operator = self.op(module)
                sample = self.sample([str(self.root / "missing.png")])
                sample["fps"] = 30
                models = self.face_model() if module is face else Mock()
                if module is animal:
                    kwargs = {"side_effect": self.animal_models(operator)}
                else:
                    kwargs = {"return_value": models}
                with patch.object(module, "get_model", **kwargs):
                    with self.assertRaisesRegex(ValueError, "Could not read video frame"):
                        operator.process_single(sample)
                self.assertNotIn("predictions", sample[Fields.meta])

    def test_depth_runs_each_video_separately(self):
        operator = self.op(depth)
        model = self.depth_model()
        paths = [[self.paths[2], self.paths[0]], [], self.paths[1:2]]
        sample = self.sample(paths)
        sample["fps"] = [24, None, 12]
        with patch.object(depth, "get_model", return_value=model):
            result = operator.process_single(sample)
        self.assertEqual([len(call.args[0]) for call in model.infer_video_depth.call_args_list], [2, 1])
        self.assertEqual([call.args[1] for call in model.infer_video_depth.call_args_list], [24, 12])
        tags = result[Fields.meta]["predictions"]
        self.assertEqual(tags["frame_paths"], [self.paths[2], self.paths[0], self.paths[1]])
        self.assertEqual(tags["video_indices"], [0, 0, 2])
        self.assertEqual(tags["frame_indices"], [0, 1, 0])
        self.assertEqual(tags["video_fps"], [24, -1, 12])
        self.assertEqual([frame[0][0] for frame in tags["depth_data"]], [10.0, 0.0, 2.0])
        json.dumps(result)

    def test_depth_accepts_shared_fps_for_flat_frames(self):
        operator = self.op(depth)
        sample = self.sample(self.paths)
        sample["fps"] = 30
        with patch.object(depth, "get_model", return_value=self.depth_model()):
            result = operator.process_single(sample)
        self.assertEqual(result[Fields.meta]["predictions"]["fps"], 30)

    def test_depth_rejects_missing_invalid_or_misaligned_fps_before_model_load(self):
        for fps in [None, 0, -1, float("nan"), float("inf"), [24, 12]]:
            with self.subTest(fps=fps):
                operator = self.op(depth)
                sample = self.sample([self.paths])
                sample["fps"] = fps
                with patch.object(depth, "get_model") as load:
                    with self.assertRaisesRegex(ValueError, "fps"):
                        operator.process_single(sample)
                load.assert_not_called()

    def test_depth_raw_videos_use_custom_key_and_independent_inference(self):
        operator = self.op(depth, video_key="clips")
        frames = np.zeros((2, 24, 32, 3), dtype=np.uint8)
        operator.read_video_frames.side_effect = [(frames, 24), (frames[:1], 12)]
        model = self.depth_model()
        with patch.object(depth, "get_model", return_value=model):
            result = operator.process_single({"clips": ["first.mp4", "second.mp4"]})
        self.assertEqual(
            [call.args[0] for call in operator.read_video_frames.call_args_list], ["first.mp4", "second.mp4"]
        )
        self.assertEqual(result[Fields.meta]["predictions"]["video_indices"], [0, 0, 1])
        self.assertEqual(result[Fields.meta]["predictions"]["video_fps"], [24, 12])

    def test_depth_rejects_prediction_count_mismatch(self):
        operator = self.op(depth)
        sample = self.sample(self.paths)
        sample["fps"] = 30
        model = Mock()
        model.infer_video_depth.return_value = (np.zeros((1, 24, 32)), 30)
        with patch.object(depth, "get_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "number of input frames"):
                operator.process_single(sample)
        self.assertNotIn("predictions", sample[Fields.meta])

    def test_moge_frames_only_preserve_calibration_frame_correspondence(self):
        operator = self.op(moge)
        paths = [self.paths[2], self.paths[0], self.paths[1]]
        model = self.moge_model()
        with patch.object(moge, "get_model", return_value=model):
            result = operator.process_single(self.sample(paths))
        intrinsics = result[Fields.meta]["predictions"][0][CameraCalibrationKeys.intrinsics]
        expected = [32 * (index / 255 + 1) for index in [10, 0, 2]]
        np.testing.assert_allclose([matrix[0][0] for matrix in intrinsics], expected, rtol=1e-6)

    def test_moge_preserves_empty_groups_and_bytes_input(self):
        operator = self.op(moge)
        frames = [[Path(self.paths[2]).read_bytes()], [], [Path(self.paths[0]).read_bytes()]]
        with patch.object(moge, "get_model", return_value=self.moge_model()):
            result = operator.process_single(self.sample(frames))
        tags = result[Fields.meta]["predictions"]
        self.assertEqual(len(tags), 3)
        self.assertEqual(tags[1], {})
        self.assertEqual(len(tags[0][CameraCalibrationKeys.hfov]), 1)
        self.assertEqual(len(tags[2][CameraCalibrationKeys.hfov]), 1)

    def test_moge_rejects_existing_tag_group_count_mismatch(self):
        operator = self.op(moge)
        sample = self.sample([self.paths])
        sample[Fields.meta]["predictions"] = [{}, {}]
        with patch.object(moge, "get_model") as load:
            with self.assertRaisesRegex(ValueError, "same length"):
                operator.process_single(sample)
        load.assert_not_called()

    def test_hawor_frames_only_use_matching_calibration_and_frame_paths(self):
        operator = self.op(hawor)
        paths = [self.paths[2], self.paths[0]]
        sample = self.sample(paths)
        sample[Fields.meta]["calibration"] = [{CameraCalibrationKeys.hfov: [1.0, 1.0]}]
        with patch.object(hawor, "get_model", side_effect=[(Mock(), None, None, None), Mock()]):
            result = operator.process_single(sample)
        self.assertEqual(operator.hawor_motion_estimation.call_args.kwargs["frame_file_paths"], paths)
        self.assertEqual(len(result[Fields.meta]["predictions"]), 1)
        self.assertEqual(result[Fields.meta]["predictions"][0]["fov_x"], 1.0)

    def test_hawor_empty_groups_do_not_shift_calibration_indices(self):
        operator = self.op(hawor)
        sample = self.sample([[], self.paths[:1]])
        sample[Fields.meta]["calibration"] = [{}, {CameraCalibrationKeys.hfov: [1.2]}]
        with patch.object(hawor, "get_model", side_effect=[(Mock(), None, None, None), Mock()]):
            result = operator.process_single(sample)
        self.assertEqual([tag["fov_x"] for tag in result[Fields.meta]["predictions"]], [0.0, 1.2])

    def test_hawor_rejects_calibration_count_mismatch_before_model_load(self):
        for calibration in [[], [{CameraCalibrationKeys.hfov: [1.0, 1.0]}]]:
            with self.subTest(calibration=calibration):
                operator = self.op(hawor)
                sample = self.sample([self.paths])
                sample[Fields.meta]["calibration"] = calibration
                with patch.object(hawor, "get_model") as load:
                    with self.assertRaisesRegex(ValueError, "calibration"):
                        operator.process_single(sample)
                load.assert_not_called()

    def test_hawor_accepts_one_shared_static_calibration(self):
        operator = self.op(hawor)
        sample = self.sample([self.paths])
        sample[Fields.meta]["calibration"] = [{CameraCalibrationKeys.hfov: [1.0]}]
        with patch.object(hawor, "get_model", side_effect=[(Mock(), None, None, None), Mock()]):
            result = operator.process_single(sample)
        self.assertEqual(result[Fields.meta]["predictions"][0]["fov_x"], 1.0)

    def test_hawor_corrupt_frames_cannot_shift_frame_ids(self):
        with self.assertRaisesRegex(ValueError, "index 1"):
            hawor.VideoHandReconstructionHaworMapper._decode_frames([self.paths[0], str(self.root / "missing.png")])

    def test_hawor_bytes_are_materialized_in_input_order(self):
        operator = self.op(hawor)
        paths = [self.paths[2], self.paths[0]]
        sample = self.sample([Path(path).read_bytes() for path in paths])
        sample[Fields.meta]["calibration"] = [{CameraCalibrationKeys.hfov: [1.0, 1.0]}]
        observed = []

        def reconstruct(*args, frame_file_paths, **kwargs):
            observed.extend(int(cv2.imread(path)[0, 0, 0]) for path in frame_file_paths)
            return {"left": {}, "right": {}}

        operator.hawor_motion_estimation.side_effect = reconstruct
        with patch.object(hawor, "get_model", side_effect=[(Mock(), None, None, None), Mock()]):
            operator.process_single(sample)
        self.assertEqual(observed, [10, 0])

    def test_real_extraction_to_pose_dataset_and_jsonl_export(self):
        video = self.root / "source.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 6, (32, 24))
        self.assertTrue(writer.isOpened())
        for index in range(18):
            writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
        writer.release()
        extractor = VideoExtractFramesMapper(
            frame_sampling_method="uniform",
            frame_num=12,
            frame_dir=str(self.root / "frames"),
            legacy_split_by_text_token=False,
            video_key="clips",
            num_proc=1,
            auto_op_parallelism=False,
        )
        extracted = extractor.process_single({"clips": [str(video)]})
        paths = extracted[MetaKeys.video_frames][0]
        self.assertEqual(len(paths), 12)
        # A stale unrelated frame must not be picked up by downstream operators.
        Image.new("RGB", (32, 24)).save(Path(paths[0]).parent / "frame_999.png")
        operator = self.op(body, video_key="clips")
        operator.fused_ops = [extractor]
        with patch.object(body, "get_model", return_value=self.body_model):
            dataset = operator.run(NestedDataset.from_list([{"clips": [str(video)]}]))
        output = self.root / "predictions.jsonl"
        dataset.to_json(str(output))
        meta = json.loads(output.read_text())[Fields.meta]
        if isinstance(meta, str):
            meta = json.loads(meta)
        self.assertEqual(meta["predictions"]["frame_paths"], paths)
        self.assertEqual(
            [Path(path).name for path in meta["predictions"]["frame_paths"]], [f"frame_{i}.jpg" for i in range(12)]
        )
        self.assertEqual(len(meta["predictions"]["body_keypoints"]), 12)

    def test_explicit_empty_frames_do_not_fall_back_to_raw_video(self):
        operator = self.op(body)
        sample = self.sample([])
        sample["videos"] = ["should-not-be-read.mp4"]
        with patch.object(body, "get_model") as load:
            operator.process_single(sample)
        operator.fused_ops[0].process_single.assert_not_called()
        load.assert_not_called()

    def test_empty_and_nonempty_samples_share_a_dataset_export_schema(self):
        for module in [body, face, animal, depth, moge, hawor]:
            with self.subTest(module=module.OP_NAME):
                operator = self.op(module)
                samples = [self.sample([]), self.sample(self.paths[:1])]
                kwargs = {}
                if module is body:
                    kwargs["return_value"] = self.body_model
                elif module is face:
                    kwargs["return_value"] = self.face_model()
                elif module is animal:
                    kwargs["side_effect"] = self.animal_models(operator)
                elif module is depth:
                    kwargs["return_value"] = self.depth_model()
                    for sample in samples:
                        sample["fps"] = 30
                elif module is moge:
                    kwargs["return_value"] = self.moge_model()
                else:
                    kwargs["side_effect"] = [(Mock(), None, None, None), Mock()]
                    samples[0][Fields.meta]["calibration"] = []
                    samples[1][Fields.meta]["calibration"] = [{CameraCalibrationKeys.hfov: [1.0]}]
                with patch.object(module, "get_model", **kwargs):
                    dataset = operator.run(NestedDataset.from_list(samples))
                output = self.root / (module.OP_NAME + ".jsonl")
                dataset.to_json(str(output))
                records = [json.loads(line) for line in output.read_text().splitlines()]
                tags = []
                for record in records:
                    meta = record[Fields.meta]
                    tags.append((json.loads(meta) if isinstance(meta, str) else meta)["predictions"])
                if module in [moge, hawor]:
                    self.assertEqual(tags[0], [])
                    self.assertEqual(len(tags[1]), 1)
                else:
                    self.assertEqual(tags[0]["frame_paths"], [])
                    self.assertEqual(tags[1]["frame_paths"], self.paths[:1])

    def test_moge_to_hawor_frame_only_pipeline_preserves_reconstruction_indices(self):
        paths = [self.paths[2], self.paths[0], self.paths[1]]
        calibration = self.op(moge, frame_field="frames", tag_field_name="calibration")
        reconstruction = self.op(hawor, frame_field="frames")
        observed = []

        def reconstruct(images, tracks, model, focal, frame_file_paths, **kwargs):
            observed.extend(frame_file_paths)
            return {
                "left": {
                    index: {
                        "global_orient": np.zeros(3),
                        "hand_pose": np.zeros(45),
                        "betas": np.zeros(10),
                        "transl": np.array([image[0, 0, 0], 0.0, 0.0]),
                    }
                    for index, image in enumerate(images)
                },
                "right": {},
            }

        reconstruction.hawor_motion_estimation.side_effect = reconstruct
        with patch.object(moge, "get_model", return_value=self.moge_model()):
            dataset = calibration.run(NestedDataset.from_list([{"frames": paths}]))
        with patch.object(hawor, "get_model", side_effect=[(Mock(), None, None, None), Mock()]):
            dataset = reconstruction.run(dataset)
        output = self.root / "reconstructed.jsonl"
        dataset.to_json(str(output))
        meta = json.loads(output.read_text())[Fields.meta]
        meta = json.loads(meta) if isinstance(meta, str) else meta
        left = meta["predictions"][0]["left"]
        self.assertEqual(left["frame_ids"], [0, 1, 2])
        self.assertEqual([value[0] for value in left["transl"]], [10.0, 0.0, 2.0])

    def test_cached_outputs_are_not_recomputed(self):
        for module in [animal, face, body, depth, hawor]:
            with self.subTest(module=module.OP_NAME):
                operator = self.op(module)
                sample = self.sample(self.paths)
                sample[Fields.meta]["predictions"] = {"cached": True}
                with patch.object(module, "get_model") as load:
                    self.assertIs(operator.process_single(sample), sample)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
