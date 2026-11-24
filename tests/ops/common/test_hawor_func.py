import math
import unittest

import numpy as np
import torch

from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase

from .test_hawor_imports import load_test_modules

hw, hv, dw = load_test_modules()


class HaworGeometryTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_transform_round_trip(self):
        center = np.array([4.0, 4.0])
        point = np.array([3.0, 5.0])
        transformed = hw.transform(point, center, 0.04, (8, 8), rot=25, asint=False)
        restored = hw.transform(transformed, center, 0.04, (8, 8), invert=1, rot=25, asint=False)
        np.testing.assert_allclose(restored, point, atol=1e-06)
        self.assertEqual(hw.transform(point, center, 0.04, (8, 8)).dtype.kind, "i")

    @TEST_TAG("standalone")
    def test_crop_shapes(self):
        center = np.array([4.0, 4.0])
        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        cropped = hw.crop(image, center, 0.04, (6, 4))
        self.assertEqual(cropped.shape, (6, 4, 3))
        rotated = hw.crop(image, center, 0.04, (6, 6), rot=15)
        self.assertEqual(rotated.shape, (6, 6, 3))

    @TEST_TAG("standalone")
    def test_boxes_to_center_and_scale(self):
        centers, scales = hw.boxes_2_cs(np.array([[0, 2, 10, 6], [2, 0, 4, 8]], dtype=float))
        np.testing.assert_allclose(centers, [[5, 4], [3, 4]])
        np.testing.assert_allclose(scales, [0.05, 0.04])

    @TEST_TAG("standalone")
    def test_exists_and_default(self):
        self.assertTrue(hw.exists(0))
        self.assertEqual(hw.default(None, lambda: 7), 7)
        self.assertEqual(hw.default(3, 7), 3)

    @TEST_TAG("standalone")
    def test_keypoint_rectangle_confidence_threshold(self):
        keypoints = np.array([[1, 2, 0.9], [5, 8, 0.8], [99, 99, 0.0]])
        self.assertEqual(hw.get_keypoints_rectangle(keypoints, 0.5), (4, 6, 24))
        self.assertEqual(hw.get_keypoints_rectangle(keypoints, 1.0), (0, 0, 0))

    @TEST_TAG("standalone")
    def test_lightweight_keypoint_renderers(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        hand = np.c_[
            np.linspace(8, 55, 21),
            np.linspace(10, 50, 21),
            np.ones(21),
        ]
        rendered_hand = hw.render_openpose(image, hand)
        self.assertEqual(rendered_hand.shape, image.shape)
        self.assertGreater(int(rendered_hand.sum()), 0)
        rendered_landmarks = hw.render_openpose_landmarks(image, hand)
        self.assertGreater(int(rendered_landmarks.sum()), 0)
        rendered_confidence = hw.render_hand_keypoints(
            image, hand, use_confidence=True, map_fn=lambda confidence: confidence
        )
        self.assertGreater(int(rendered_confidence.sum()), 0)

        body = np.c_[
            np.linspace(5, 58, 25),
            np.linspace(8, 52, 25),
            np.ones(25),
        ]
        rendered_body = hw.render_body_keypoints(image, body)
        self.assertEqual(rendered_body.shape, image.shape)
        self.assertGreater(int(rendered_body.sum()), 0)

    @TEST_TAG("standalone")
    def test_perspective_projection(self):
        points = torch.tensor([[[1.0, 2.0, 2.0], [0.0, 0.0, 1.0]]])
        projected = hw.perspective_projection(
            points,
            torch.eye(3).unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([2.0]),
            torch.tensor([[10.0, 20.0]]),
        )
        torch.testing.assert_close(projected[0, 0], torch.tensor([10.666666, 21.333334]), atol=1e-05, rtol=0)
        distorted = hw.perspective_projection(
            points, None, None, torch.tensor([2.0]), torch.zeros(1, 2), torch.zeros(1, 5)
        )
        torch.testing.assert_close(distorted[0, 0], torch.tensor([1.0, 2.0]))

    @TEST_TAG("standalone")
    def test_identity_rotation_conversions(self):
        identity = torch.eye(3).unsqueeze(0)
        torch.testing.assert_close(hw.avg_rot(identity.repeat(2, 1, 1)), identity[0])
        torch.testing.assert_close(hw.rot9d_to_rotmat(identity.reshape(1, 9)), identity)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(hw.quat_to_rotmat(quat), identity)
        torch.testing.assert_close(hw.quaternion_to_rotation_matrix(quat), identity)
        torch.testing.assert_close(hw.rotation_matrix_to_quaternion(identity), quat)
        torch.testing.assert_close(hw.rotation_matrix_to_angle_axis(identity), torch.zeros(1, 3))
        torch.testing.assert_close(hw.quaternion_to_angle_axis(quat), torch.zeros(1, 3))

    @TEST_TAG("standalone")
    def test_quaternion_angle_axis_rejects_invalid_inputs(self):
        with self.assertRaises(TypeError):
            hw.quaternion_to_angle_axis([1, 0, 0, 0])
        with self.assertRaises(ValueError):
            hw.quaternion_to_angle_axis(torch.ones(3))

    @TEST_TAG("standalone")
    def test_rotation_6d_conversions(self):
        identity = torch.eye(3).unsqueeze(0)
        self.assertEqual(hw.rotmat_to_rot6d(identity).shape, (1, 6))
        canonical_rot6 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        torch.testing.assert_close(hw.rot6d_to_rotmat(canonical_rot6), identity)

    @TEST_TAG("standalone")
    def test_angle_axis_and_quaternion_inverse(self):
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        angle_axis = torch.tensor([[0.0, 0.0, math.pi / 2], [0.0, 0.0, 0.0]])
        quats = hw.angle_axis_to_quaternion(angle_axis)
        mats = hw.angle_axis_to_rotation_matrix(angle_axis)
        torch.testing.assert_close(mats, hw.aa_to_rotmat(angle_axis), atol=1e-06, rtol=1e-06)
        product = hw.quaternion_mul(quats, hw.quaternion_inverse(quats))
        torch.testing.assert_close(product, quat.repeat(2, 1), atol=1e-06, rtol=1e-06)

    @TEST_TAG("standalone")
    def test_quaternion_slerp(self):
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        halfway = hw.quaternion_slerp(torch.tensor([0.5]), quat, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        self.assertAlmostEqual(float(halfway.norm()), 1.0, places=6)
        negative_path = hw.quaternion_slerp(
            torch.tensor([0.25]), quat, torch.tensor([[-math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]])
        )
        self.assertAlmostEqual(float(negative_path.norm()), 1.0, places=6)
        self.assertGreater(float(negative_path[0, 0]), 0.0)

    @TEST_TAG("standalone")
    def test_estimate_translation(self):
        rng = np.random.default_rng(4)
        points = rng.normal(size=(24, 3))
        points[:, 2] += 5
        translation = np.array([0.2, -0.1, 2.0])
        focal, size = (500.0, 224.0)
        camera = points + translation
        joints = camera[:, :2] / camera[:, 2:] * focal + size / 2
        estimated = hw.estimate_translation_np(points, joints, np.ones(24), focal, size)
        np.testing.assert_allclose(estimated, translation, atol=1e-05)
        batched = hw.estimate_translation(
            torch.tensor(points[None], dtype=torch.float32),
            torch.tensor(np.c_[joints, np.ones(24)][None], dtype=torch.float32),
            focal,
            size,
        )
        np.testing.assert_allclose(batched.numpy()[0], translation, atol=0.0002)

    @TEST_TAG("standalone")
    def test_parse_contiguous_frame_chunks(self):
        frames = np.array([0, 1, 2, 5, 6, 9, 10, 11])
        boxes = np.arange(40).reshape(8, 5)
        frame_chunks, box_chunks = hw.parse_chunks(frames, boxes, min_len=2)
        self.assertEqual([x.tolist() for x in frame_chunks], [[0, 1, 2], [5, 6], [9, 10, 11]])
        self.assertEqual([len(x) for x in box_chunks], [3, 2, 3])

    @TEST_TAG("standalone")
    def test_interpolate_missing_bboxes(self):
        missing = np.array([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [3, 4, 5, 6, 7]], dtype=float)
        filled = hw.interpolate_bboxes(missing)
        np.testing.assert_allclose(filled[0], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(filled[2], [2, 3, 4, 5, 6])
        self.assertIs(hw.interpolate_bboxes(filled), filled)


class HaworNetworksTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_normalization_layers(self):
        self.assertIsInstance(hw.normalization_layer("batch", 4), torch.nn.BatchNorm1d)
        self.assertIsInstance(hw.normalization_layer("layer", 4), torch.nn.LayerNorm)
        self.assertIsInstance(hw.normalization_layer(None, 4), torch.nn.Identity)
        with self.assertRaises(ValueError):
            hw.normalization_layer("bad", 4)
        with self.assertRaises(AssertionError):
            hw.normalization_layer("ada", 4)
        with self.assertRaises(ValueError):
            hw.AdaptiveLayerNorm1D(0, 2)

    @TEST_TAG("standalone")
    def test_adaptive_layer_norm_initialization(self):
        ada = hw.AdaptiveLayerNorm1D(4, 2)
        x = torch.randn(2, 3, 4)
        conditioned = ada(x, torch.ones(2, 2))
        torch.testing.assert_close(conditioned, torch.nn.functional.layer_norm(x, (4,)))

    @TEST_TAG("standalone")
    def test_frequency_embedding_shapes(self):
        embedded = hw.FrequencyEmbedder(2, 1)(torch.tensor([0.0, 1.0]))
        self.assertEqual(embedded.shape, (2, 5))
        self.assertEqual(hw.FrequencyEmbedder(2, 1)(torch.ones(2, 3)).shape, (2, 15))

    @TEST_TAG("standalone")
    def test_feed_forward_shape(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4)
        self.assertEqual(hw.FeedForward(4, 8)(x).shape, x.shape)

    @TEST_TAG("standalone")
    def test_attention_shapes(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4)
        self.assertEqual(hw.Attention(4, heads=1, dim_head=4)(x).shape, x.shape)
        self.assertEqual(hw.Attention(4, heads=2, dim_head=3)(x).shape, x.shape)
        self.assertEqual(
            hw.CrossAttention(4, context_dim=6, heads=2, dim_head=2)(x, torch.randn(2, 2, 6)).shape, x.shape
        )

    @TEST_TAG("standalone")
    def test_transformer_shapes_and_context_validation(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4)
        self.assertEqual(hw.Transformer(4, 2, 2, 2, 8)(x).shape, x.shape)
        cross = hw.TransformerCrossAttn(4, 2, 2, 2, 8, context_dim=6)
        self.assertEqual(cross(x, context=torch.randn(2, 2, 6)).shape, x.shape)
        with self.assertRaises(ValueError):
            cross(x, context_list=[torch.randn(2, 2, 6)])
        adaptive_transformer = hw.Transformer(4, 1, 1, 4, 8, norm="ada", norm_cond_dim=2)
        self.assertEqual(adaptive_transformer(x, torch.ones(2, 2)).shape, x.shape)

    @TEST_TAG("standalone")
    def test_token_dropout_training_and_evaluation(self):
        x = torch.ones(2, 4, 3)
        with self.assertRaises(ValueError):
            hw.DropTokenDropout(-0.1)
        drop = hw.DropTokenDropout(1.0)
        drop.train()
        self.assertEqual(drop(x).shape, (2, 0, 3))
        drop.eval()
        self.assertEqual(drop(x).shape, x.shape)
        zero = hw.ZeroTokenDropout(1.0)
        zero.train()
        self.assertEqual(int(torch.count_nonzero(zero(x.clone()))), 0)

    @TEST_TAG("standalone")
    def test_transformer_encoder_options(self):
        for location in ("input", "token", "token_afterpos"):
            encoder = hw.TransformerEncoder(4, 3, 4, 1, 1, 8, dim_head=4, emb_dropout_loc=location)
            self.assertEqual(encoder(torch.randn(2, 4, 3)).shape, (2, 4, 4))
        freq_encoder = hw.TransformerEncoder(4, 2, 4, 1, 1, 8, dim_head=4, token_pe_numfreq=2)
        self.assertEqual(freq_encoder(torch.randn(2, 4, 2)).shape, (2, 4, 4))
        with self.assertRaises(ValueError):
            hw.TransformerEncoder(4, 3, 4, 1, 1, 8, emb_dropout_type="bad")

    @TEST_TAG("standalone")
    def test_transformer_decoder_options(self):
        decoder = hw.TransformerDecoder(3, 4, 4, 1, 1, 8, dim_head=4, context_dim=5, emb_dropout_type="normal")
        self.assertEqual(decoder(torch.randn(2, 3, 4), context=torch.randn(2, 2, 5)).shape, (2, 3, 4))
        self.assertIsInstance(
            hw.TransformerDecoder(3, 4, 4, 1, 1, 8, skip_token_embedding=True).to_token_embedding, torch.nn.Identity
        )
        with self.assertRaises(ValueError):
            hw.TransformerDecoder(3, 3, 4, 1, 1, 8, skip_token_embedding=True)

    @TEST_TAG("standalone")
    def test_positional_encoding(self):
        pe = hw.PositionalEncoding(4, dropout=0.0, max_len=5)
        encoded = pe(torch.zeros(3, 2, 4))
        torch.testing.assert_close(encoded[0, :, 0::2], torch.zeros(2, 2))

    @TEST_TAG("standalone")
    def test_temporal_attention_shape(self):
        temporal = hw.temporal_attention(in_dim=4, out_dim=4, hdim=4, nlayer=1, nhead=1, residual=True).eval()
        self.assertEqual(temporal(torch.randn(2, 3, 4)).shape, (2, 3, 4))

    @TEST_TAG("standalone")
    def test_hawor_camera_helpers_without_model(self):
        model = hw.HAWOR.__new__(hw.HAWOR)
        torch.nn.Module.__init__(model)
        model.crop_size = 200
        center = torch.tensor([[100.0, 80.0]])
        scale = torch.tensor([1.0])
        focal = torch.tensor([500.0])
        image_center = torch.tensor([[90.0, 70.0]])
        camera = torch.tensor([[2.0, 0.1, -0.2]])
        trans = model.get_trans(camera, center, scale, focal, image_center)
        torch.testing.assert_close(trans, torch.tensor([[[0.15, -0.15, 2.5]]]))
        bbox = model.bbox_est(center, scale, focal, image_center)
        torch.testing.assert_close(bbox, torch.tensor([[0.056, 0.056, 2.6666667]]), atol=1e-6, rtol=1e-6)
        points = torch.tensor([[[0.0, 0.0, 0.0]]])
        full, crop = model.project(points, camera, center, scale, focal, image_center, return_full=True)
        self.assertEqual(full.shape, (1, 1, 2))
        self.assertEqual(crop.shape, (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
