import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_hand_action_compute_mapper import VideoHandActionComputeMapper
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoHandActionComputeMapperTest(DataJuicerTestCaseBase):

    def _make_sample(self, num_frames=10, with_left=True):
        """Create a synthetic sample with hand reconstruction and camera pose data."""
        # Generate dummy camera poses: identity with small translations
        cam_c2w = []
        for i in range(num_frames):
            mat = np.eye(4)
            mat[0, 3] = i * 0.01  # small x translation
            cam_c2w.append(mat.tolist())

        # Generate dummy right hand reconstruction data
        right_frame_ids = list(range(num_frames))
        right_transl = (np.random.randn(num_frames, 3) * 0.1).tolist()
        right_global_orient = (np.random.randn(num_frames, 3) * 0.1).tolist()  # axis-angle
        right_hand_pose = (np.random.randn(num_frames, 45) * 0.1).tolist()  # axis-angle

        hand_recon = {
            "fov_x": 0.75,
            "img_focal": 500.0,
            "right": {
                "frame_ids": right_frame_ids,
                "transl": right_transl,
                "global_orient": right_global_orient,
                "hand_pose": right_hand_pose,
                "betas": (np.zeros((num_frames, 10))).tolist(),
            },
            "left": {
                "frame_ids": [],
                "transl": [],
                "global_orient": [],
                "hand_pose": [],
                "betas": [],
            }
        }

        if with_left:
            left_frame_ids = list(range(0, num_frames, 2))  # every other frame
            n_left = len(left_frame_ids)
            hand_recon["left"] = {
                "frame_ids": left_frame_ids,
                "transl": (np.random.randn(n_left, 3) * 0.1).tolist(),
                "global_orient": (np.random.randn(n_left, 3) * 0.1).tolist(),
                "hand_pose": (np.random.randn(n_left, 45) * 0.1).tolist(),
                "betas": (np.zeros((n_left, 10))).tolist(),
            }

        camera_pose = {
            CameraCalibrationKeys.cam_c2w: cam_c2w,
        }

        sample = {
            'videos': ['dummy_video.mp4'],
            'text': 'pick up the cup',
            Fields.meta: {
                MetaKeys.hand_reconstruction_hawor_tags: [hand_recon],
                MetaKeys.video_camera_pose_tags: [camera_pose],
            }
        }
        return sample

    def test_both_hands(self):
        """Test computing actions for both hands."""
        sample = self._make_sample(num_frames=10, with_left=True)
        ds_list = [sample]

        op = VideoHandActionComputeMapper(
            hand_reconstruction_field=MetaKeys.hand_reconstruction_hawor_tags,
            camera_pose_field=MetaKeys.video_camera_pose_tags,
            tag_field_name=MetaKeys.hand_action_tags,
            hand_type="both",
        )

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 1)
        tag = res_list[0][Fields.meta][MetaKeys.hand_action_tags]
        self.assertIsInstance(tag, list)
        self.assertEqual(len(tag), 1)  # one video

        video_result = tag[0]
        # Should have both hand results
        self.assertIn('right', video_result)
        self.assertIn('left', video_result)

        # Right hand
        right = video_result['right']
        self.assertIn('states', right)
        self.assertIn('actions', right)
        self.assertIn('valid_frame_ids', right)

        states = np.array(right['states'])
        actions = np.array(right['actions'])
        self.assertEqual(states.shape[1], 8)  # 8-dim state
        self.assertEqual(actions.shape[1], 7)  # 7-dim action
        self.assertEqual(states.shape[0], actions.shape[0])

        # Left hand
        left = video_result['left']
        left_states = np.array(left['states'])
        left_actions = np.array(left['actions'])
        if len(left_states) > 0:
            self.assertEqual(left_states.shape[1], 8)
            self.assertEqual(left_actions.shape[1], 7)

    def test_right_hand_only(self):
        """Test computing actions for right hand only."""
        sample = self._make_sample(num_frames=10, with_left=False)
        ds_list = [sample]

        op = VideoHandActionComputeMapper(
            hand_type="right",
        )

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        video_result = res_list[0][Fields.meta][MetaKeys.hand_action_tags][0]
        self.assertIn('right', video_result)
        self.assertNotIn('left', video_result)

        right = video_result['right']
        states = np.array(right['states'])
        self.assertEqual(states.shape[1], 8)

    def test_insufficient_frames(self):
        """Test with only 1 frame (needs at least 2 for actions)."""
        sample = self._make_sample(num_frames=1, with_left=False)
        ds_list = [sample]

        op = VideoHandActionComputeMapper(hand_type="right")

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        video_result = res_list[0][Fields.meta][MetaKeys.hand_action_tags][0]
        right = video_result['right']
        # Should be empty when < 2 frames
        self.assertEqual(len(right['states']), 0)
        self.assertEqual(len(right['actions']), 0)

    def test_empty_hand_recon(self):
        """Test with empty hand reconstruction data."""
        sample = {
            'videos': ['dummy_video.mp4'],
            'text': 'test',
            Fields.meta: {
                MetaKeys.hand_reconstruction_hawor_tags: [],
                MetaKeys.video_camera_pose_tags: [],
            }
        }

        op = VideoHandActionComputeMapper(hand_type="both")

        dataset = Dataset.from_list([sample])
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        tag = res_list[0][Fields.meta][MetaKeys.hand_action_tags]
        self.assertEqual(len(tag), 0)

    def test_last_action_is_zero(self):
        """Test that the last frame's positional action is zero."""
        sample = self._make_sample(num_frames=5, with_left=False)
        ds_list = [sample]

        op = VideoHandActionComputeMapper(hand_type="right")

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        video_result = res_list[0][Fields.meta][MetaKeys.hand_action_tags][0]
        actions = np.array(video_result['right']['actions'])
        # Last action: position deltas should be zero
        np.testing.assert_array_almost_equal(actions[-1, :6], 0.0, decimal=5)

    def test_mul_proc(self):
        """Test with multiple processes."""
        samples = [
            self._make_sample(num_frames=8, with_left=True),
            self._make_sample(num_frames=6, with_left=True),
        ]

        op = VideoHandActionComputeMapper(hand_type="both")

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 2)
        for sample in res_list:
            tag = sample[Fields.meta][MetaKeys.hand_action_tags]
            self.assertGreater(len(tag), 0)


if __name__ == '__main__':
    unittest.main()
