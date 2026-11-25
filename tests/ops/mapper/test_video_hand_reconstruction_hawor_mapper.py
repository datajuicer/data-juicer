import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper import VideoExtractFramesMapper, VideoHandReconstructionHaworMapper
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


@unittest.skip('Users need to download MANO_RIGHT.pkl and MANO_LEFT.pkl.')
class VideoHandReconstructionHaworMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    def _build_ds_list(self):
        """Build dataset with pre-extracted frames and camera calibration in meta."""
        ds_list = [{
            'videos': [self.vid3_path],
            Fields.meta: {
                'camera_calibration': [{CameraCalibrationKeys.hfov: [0.76] * 6,}],
            }
        }, {
            'videos': [self.vid4_path],
            Fields.meta: {
                'camera_calibration': [{
                    CameraCalibrationKeys.hfov: [0.66] * 5,
                }],
            }
        }]

        extract_op = VideoExtractFramesMapper(
            frame_sampling_method='all_keyframes',
            output_format='bytes',
            legacy_split_by_text_token=False,
        )
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(extract_op.process, batched=True, batch_size=1)

        return dataset.to_list()

    def test_default(self):
        ds_list = self._build_ds_list()

        op = VideoHandReconstructionHaworMapper(
            hawor_model_path="hawor.ckpt",
            hawor_config_path="model_config.yaml",
            hawor_detector_path="detector.pt",
            tag_field_name=MetaKeys.hand_reconstruction_hawor_tags,
            mano_right_path='MANO_RIGHT.pkl',
            mano_left_path='MANO_LEFT.pkl',
            frame_field=MetaKeys.video_frames,
            camera_calibration_field='camera_calibration',
            thresh=0.2,
        )

        # Process each sample directly to avoid Arrow type inference
        # conflicts when hand detection results vary across samples
        # (empty list [] inferred as null vs list<list<double>>).
        res_list = []
        for sample in ds_list:
            result = op.process_single(sample)
            res_list.append(result)

        for sample in res_list:
            tag = sample[Fields.meta][MetaKeys.hand_reconstruction_hawor_tags]
            self.assertIsInstance(tag, list)
            self.assertGreater(len(tag), 0)

            for video_result in tag:
                # Check top-level keys
                self.assertIn('fov_x', video_result)
                self.assertIn('img_focal', video_result)
                self.assertIn('left', video_result)
                self.assertIn('right', video_result)

                # Check hand output structure (axis-angle format)
                for hand_type in ['left', 'right']:
                    hand = video_result[hand_type]
                    self.assertIn('frame_ids', hand)
                    self.assertIn('global_orient', hand)
                    self.assertIn('hand_pose', hand)
                    self.assertIn('betas', hand)
                    self.assertIn('transl', hand)

                    n_frames = len(hand['frame_ids'])
                    if n_frames > 0:
                        # global_orient: list of (3,) axis-angle
                        self.assertEqual(
                            np.array(hand['global_orient']).shape,
                            (n_frames, 3))
                        # hand_pose: list of (45,) axis-angle
                        self.assertEqual(
                            np.array(hand['hand_pose']).shape,
                            (n_frames, 45))
                        # betas: list of (10,)
                        self.assertEqual(
                            np.array(hand['betas']).shape,
                            (n_frames, 10))
                        # transl: list of (3,)
                        self.assertEqual(
                            np.array(hand['transl']).shape,
                            (n_frames, 3))


if __name__ == '__main__':
    unittest.main()
