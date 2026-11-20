import math
import types
import unittest
from unittest import mock

import numpy as np
import torch

from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase

from .test_hawor_imports import load_test_modules

hw, hv, dw = load_test_modules()


class DWPoseTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_nms(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=float)
        self.assertEqual(dw.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5), [0, 2])
        scores = np.array([[0.9, 0.0], [0.8, 0.0], [0.0, 0.7]])
        dets = dw.multiclass_nms(boxes, scores, 0.5, 0.1)
        self.assertEqual(dets.shape, (2, 6))
        self.assertIsNone(dw.multiclass_nms(boxes, np.zeros((3, 2)), 0.5, 0.1))

    @TEST_TAG("standalone")
    def test_detector_grid_postprocess(self):
        count = sum((8 // s * (8 // s) for s in (8, 16, 32)))
        outputs = np.zeros((1, count, 4))
        processed = dw.demo_postprocess(outputs, (8, 8))
        np.testing.assert_allclose(processed[0, 0], [0, 0, 8, 8])
        p6_count = sum((64 // s * (64 // s) for s in (8, 16, 32, 64)))
        self.assertEqual(dw.demo_postprocess(np.zeros((1, p6_count, 4)), (64, 64), p6=True).shape[1], p6_count)

    @TEST_TAG("standalone")
    def test_detector_inference(self):
        count = sum((8 // s * (8 // s) for s in (8, 16, 32)))
        predictions = np.zeros((1, count, 7), dtype=float)
        predictions[0, 0] = [4, 4, 1, 1, 0.9, 0.9, 0.0]
        session = mock.Mock()
        session.get_inputs.return_value = [types.SimpleNamespace(name="images")]
        session.run.return_value = [predictions]
        with (
            mock.patch.object(dw, "preprocess_det", return_value=(np.zeros((3, 8, 8)), 1.0)),
            mock.patch.object(dw, "demo_postprocess", side_effect=lambda value, size: value),
        ):
            detected = dw.inference_detector(session, np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(detected.shape, (1, 4))
        predictions[..., 4:] = 0
        with (
            mock.patch.object(dw, "preprocess_det", return_value=(np.zeros((3, 8, 8)), 1.0)),
            mock.patch.object(dw, "demo_postprocess", side_effect=lambda value, size: value),
        ):
            self.assertEqual(dw.inference_detector(session, np.zeros((8, 8, 3), dtype=np.uint8)).size, 0)

    @TEST_TAG("standalone")
    def test_bbox_center_scale_and_aspect_ratio(self):
        single_center, single_scale = dw.bbox_xyxy2cs(np.array([0, 0, 8, 4]), padding=1.25)
        np.testing.assert_allclose(single_center, [4, 2])
        np.testing.assert_allclose(single_scale, [10, 5])
        centers, scales = dw.bbox_xyxy2cs(np.array([[0, 0, 4, 8], [2, 2, 6, 6]]))
        self.assertEqual(centers.shape, (2, 2))
        np.testing.assert_allclose(dw._fix_aspect_ratio(np.array([10.0, 5.0]), 1.0), [10, 10])
        np.testing.assert_allclose(dw._fix_aspect_ratio(np.array([4.0, 8.0]), 0.5), [4, 8])

    @TEST_TAG("standalone")
    def test_affine_geometry(self):
        np.testing.assert_allclose(dw._rotate_point(np.array([1.0, 0.0]), math.pi / 2), [0, 1], atol=1e-07)
        np.testing.assert_allclose(dw._get_3rd_point(np.array([1, 0]), np.array([0, 0])), [0, 1])
        warp = dw.get_warp_matrix(np.array([4.0, 4.0]), np.array([8.0, 8.0]), 0, (8, 8))
        inverse = dw.get_warp_matrix(np.array([4.0, 4.0]), np.array([8.0, 8.0]), 0, (8, 8), inv=True)
        np.testing.assert_allclose(warp, inverse, atol=1e-06)
        image = np.arange(8 * 4 * 3, dtype=np.uint8).reshape(8, 4, 3)
        affine, fixed_scale = dw.top_down_affine((4, 8), np.array([4.0, 8.0]), np.array([2.0, 4.0]), image)
        self.assertEqual(affine.shape, (8, 4, 3))
        np.testing.assert_allclose(fixed_scale, [4, 8])

    @TEST_TAG("standalone")
    def test_detector_and_pose_preprocessing(self):
        image = np.arange(8 * 4 * 3, dtype=np.uint8).reshape(8, 4, 3)
        affine, fixed_scale = dw.top_down_affine((4, 8), np.array([4.0, 8.0]), np.array([2.0, 4.0]), image)
        self.assertEqual(affine.shape, (8, 4, 3))
        np.testing.assert_allclose(fixed_scale, [4, 8])
        preprocessed, ratio = dw.preprocess_det(image, (16, 16))
        self.assertEqual(preprocessed.shape, (3, 16, 16))
        self.assertEqual(ratio, 2.0)
        gray, _ = dw.preprocess_det(np.zeros((4, 8), dtype=np.uint8), (8, 8), swap=(0, 1))
        self.assertEqual(gray.shape, (8, 8))
        pose_images, pose_centers, pose_scales = dw.preprocess_pose(image, [], input_size=(4, 8))
        self.assertEqual((len(pose_images), len(pose_centers), len(pose_scales)), (1, 1, 1))

    @TEST_TAG("standalone")
    def test_simcc_decode_and_postprocess(self):
        simcc_x = np.array([[[0, 1, 0, 0], [0, -1, -2, -3]]], dtype=float)
        simcc_y = np.array([[[0, 0, 2, 0], [0, -1, -2, -3]]], dtype=float)
        locations, values = dw.get_simcc_maximum(simcc_x, simcc_y)
        np.testing.assert_allclose(locations[0, 0], [1, 2])
        np.testing.assert_allclose(locations[0, 1], [-1, -1])
        decoded, decoded_values = dw.decode(simcc_x, simcc_y, 2)
        np.testing.assert_allclose(decoded[0, 0], [0.5, 1])
        np.testing.assert_allclose(decoded_values, values)
        keys, vals = dw.postprocess([[simcc_x, simcc_y]], (4, 4), [np.array([4, 4])], [np.array([8, 8])], 2)
        self.assertEqual((keys.shape, vals.shape), ((1, 2, 2), (1, 2)))

    @TEST_TAG("standalone")
    def test_pose_session_inference(self):
        session = mock.Mock()
        session.get_inputs.return_value = [types.SimpleNamespace(name="input", shape=[1, 3, 8, 4])]
        session.get_outputs.return_value = [types.SimpleNamespace(name="x"), types.SimpleNamespace(name="y")]
        session.run.return_value = [np.ones((1, 1, 8)), np.ones((1, 1, 16))]
        outputs = dw.inference(session, [np.zeros((8, 4, 3))])
        self.assertEqual(len(outputs), 1)
        with mock.patch.object(
            dw, "preprocess_pose", return_value=([np.zeros((8, 4, 3))], [np.array([2, 4])], [np.array([4, 8])])
        ):
            keypoints, scores = dw.inference_pose(session, [], np.zeros((8, 4, 3)))
        self.assertEqual((keypoints.shape, scores.shape), ((1, 1, 2), (1, 1)))

    @TEST_TAG("standalone")
    def test_smart_resize_and_padding(self):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        self.assertEqual(dw.smart_resize(rgb, (2, 3)).shape, (2, 3, 3))
        self.assertEqual(dw.smart_resize(rgb, (8, 12)).shape, (8, 12, 3))
        multispectral = np.zeros((4, 6, 4), dtype=np.uint8)
        self.assertEqual(dw.smart_resize(multispectral, (2, 3)).shape, (2, 3, 4))
        self.assertEqual(dw.smart_resize_k(multispectral, 2, 2).shape, (8, 12, 4))
        padded, pad = dw.padRightDownCorner(np.ones((3, 5, 3)), 4, 7)
        self.assertEqual((padded.shape, pad), ((4, 8, 3), [0, 0, 1, 3]))
        self.assertTrue(np.all(padded[-1] == 7))

    @TEST_TAG("standalone")
    def test_transfer_and_npmax(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        source = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}
        transferred = dw.transfer(model, source)
        self.assertEqual(set(transferred), {"0.weight", "0.bias"})
        self.assertEqual(dw.npmax(np.array([[1, 2], [5, 3]])), (1, 0))

    @TEST_TAG("standalone")
    def test_hand_detection(self):
        candidate = np.zeros((18, 3), dtype=float)
        candidate[2, :2], candidate[3, :2], candidate[4, :2] = ([20, 20], [40, 40], [70, 70])
        subset = np.full((1, 18), -1)
        subset[0, [2, 3, 4]] = [2, 3, 4]
        hands = dw.handDetect(candidate, subset, np.zeros((100, 100, 3)))
        self.assertEqual(len(hands), 1)
        self.assertFalse(hands[0][3])

    @TEST_TAG("standalone")
    def test_face_detection(self):
        candidate = np.zeros((18, 3), dtype=float)
        subset = np.full((1, 18), -1)
        candidate[0, :2], candidate[14, :2] = ([50, 50], [60, 50])
        subset[0, 0], subset[0, 14] = (0, 14)
        faces = dw.faceDetect(candidate, subset, np.zeros((100, 100, 3)))
        self.assertEqual(len(faces), 1)

    @TEST_TAG("standalone")
    def test_drawing_helpers(self):
        candidate = np.stack([np.linspace(0.1, 0.9, 18), np.linspace(0.9, 0.1, 18)], axis=1)
        subset = np.arange(18, dtype=float)[None]
        canvas = np.zeros((64, 64, 3), dtype=np.uint8)
        body = dw.draw_bodypose(canvas.copy(), candidate, subset)
        self.assertGreater(int(body.sum()), 0)

        hand = np.stack([np.linspace(0.1, 0.9, 21), np.linspace(0.2, 0.8, 21)], axis=1)
        hand_canvas = dw.draw_handpose(canvas.copy(), [hand])
        face_canvas = dw.draw_facepose(canvas.copy(), [hand[:5]])
        self.assertGreater(int(hand_canvas.sum()), 0)
        self.assertGreater(int(face_canvas.sum()), 0)
        pose = {
            "bodies": {"candidate": candidate, "subset": subset},
            "hands": [hand],
            "faces": [hand[:5]],
        }
        self.assertEqual(dw.draw_pose(pose, 64, 64).shape, canvas.shape)

    @TEST_TAG("standalone")
    def test_wholebody_with_mocked_sessions(self):
        fake_session = mock.Mock()
        runtime = types.SimpleNamespace(InferenceSession=mock.Mock(return_value=fake_session))
        with mock.patch.object(dw, "onnxruntime", runtime):
            wholebody = dw.Wholebody("det.onnx", "pose.onnx", "cpu")
        self.assertEqual(runtime.InferenceSession.call_count, 2)
        raw_keys = np.zeros((1, 133, 2), dtype=float)
        raw_scores = np.ones((1, 133), dtype=float)
        with (
            mock.patch.object(dw, "inference_detector", return_value=np.array([[0, 0, 1, 1]])),
            mock.patch.object(dw, "inference_pose", return_value=(raw_keys, raw_scores)),
        ):
            keys, scores, boxes = wholebody(np.zeros((8, 8, 3)))
        self.assertEqual((keys.shape, scores.shape, boxes.shape), ((1, 134, 2), (1, 134), (1, 4)))

    @TEST_TAG("standalone")
    def test_detector_output_parts(self):
        detector = dw.DWposeDetector.__new__(dw.DWposeDetector)
        candidate = np.full((1, 134, 2), 4.0)
        confidence = np.ones((1, 134))
        detector.pose_estimation = mock.Mock(return_value=(candidate, confidence, np.array([[1, 2, 3, 4]])))
        with mock.patch.object(dw, "draw_pose", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
            body, foot, faces, hands, boxes, canvas = detector(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(body.shape, (1, 18, 2))
        self.assertEqual(foot.shape, (1, 6, 2))
        self.assertEqual(faces.shape, (1, 68, 2))
        self.assertEqual(hands.shape, (2, 21, 2))
        self.assertEqual(canvas.shape, (8, 8, 3))


if __name__ == "__main__":
    unittest.main()
