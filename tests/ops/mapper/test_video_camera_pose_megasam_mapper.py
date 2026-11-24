import os
import unittest
import numpy as np
import cv2

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_camera_pose_megasam_mapper import VideoCameraPoseMegaSaMMapper
from data_juicer.ops.mapper.video_extract_frames_mapper import VideoExtractFramesMapper
from data_juicer.utils.constant import Fields, MetaKeys, CameraCalibrationKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


@unittest.skip(
    'Requires mega-sam conda environment with CUDA compiled extensions '
    '(droid_backends, lietorch).'
)
class VideoCameraPoseMegaSaMMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid12_path = os.path.join(data_path, 'video12.mp4')

    def _extract_frames_and_build_dataset(self):
        """Extract real frames from videos, then build dataset with
        dummy camera calibration (depth + intrinsics) to simulate
        output from VideoCameraCalibrationMogeMapper."""
        ds_list = [{
            'videos': [self.vid3_path]
        }, {
            'videos': [self.vid12_path]
        }]

        # Step 1: Extract real frames from videos
        # Use 'uniform' sampling to ensure enough frames for DROID-SLAM
        extract_op = VideoExtractFramesMapper(
            frame_sampling_method='uniform',
            frame_num=8,
            output_format='bytes',
            legacy_split_by_text_token=False,
        )
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(extract_op.process, batched=True, batch_size=1)

        # Step 2: Add dummy camera calibration data matching real frame dims
        res_list = dataset.to_list()
        for sample in res_list:
            video_frames = sample[MetaKeys.video_frames]
            calibration_list = []
            for frames_per_video in video_frames:
                num_frames = len(frames_per_video)
                # Read the first frame to get dimensions
                first_frame = frames_per_video[0]
                if isinstance(first_frame, bytes):
                    image_array = np.frombuffer(first_frame, dtype=np.uint8)
                    first_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                else:
                    first_frame = cv2.imread(first_frame)
                h, w, _ = first_frame.shape
                # Dummy intrinsics (3x3)
                focal = 500.0
                K = [[focal, 0, w / 2.0],
                     [0, focal, h / 2.0],
                     [0, 0, 1]]
                # Structured depth maps (N, H, W) simulating a scene
                # with smooth depth gradient (near=1m to far=5m)
                # Random noise causes DROID-SLAM factor graph to be empty
                base_depth = np.linspace(1.0, 5.0, h).reshape(h, 1)
                base_depth = np.broadcast_to(base_depth, (h, w))
                depth = np.stack([
                    base_depth + 0.1 * i for i in range(num_frames)
                ]).tolist()
                calibration_list.append({
                    CameraCalibrationKeys.depth: depth,
                    CameraCalibrationKeys.intrinsics: K,
                })
            sample[Fields.meta]['camera_calibration'] = calibration_list

        dataset = Dataset.from_list(res_list)
        return dataset

    def test_default(self):
        dataset = self._extract_frames_and_build_dataset()

        op = VideoCameraPoseMegaSaMMapper(
            tag_field_name=MetaKeys.video_camera_pose_tags,
            frame_field=MetaKeys.video_frames,
            camera_calibration_field='camera_calibration',
            max_frames=1000,
        )
        dataset = dataset.map(op.process)
        res_list = dataset.to_list()

        tgt = {
            "depths_ndim": 3,
            "intrinsic_shape": [3, 3],
            "cam_c2w_last_dim": 4,
        }

        for sample in res_list:
            tag_list = sample[Fields.meta][MetaKeys.video_camera_pose_tags]
            self.assertIsInstance(tag_list, list)
            self.assertGreater(len(tag_list), 0)

            for video_result in tag_list:
                # Check output keys
                self.assertIn(CameraCalibrationKeys.depth, video_result)
                self.assertIn(CameraCalibrationKeys.intrinsics, video_result)
                self.assertIn(CameraCalibrationKeys.cam_c2w, video_result)

                # Check shapes
                depths = np.array(video_result[CameraCalibrationKeys.depth])
                intrinsic = np.array(video_result[CameraCalibrationKeys.intrinsics])
                cam_c2w = np.array(video_result[CameraCalibrationKeys.cam_c2w])

                self.assertEqual(depths.ndim, tgt["depths_ndim"])
                self.assertEqual(list(intrinsic.shape), tgt["intrinsic_shape"])
                self.assertEqual(cam_c2w.shape[-1], tgt["cam_c2w_last_dim"])
                self.assertEqual(cam_c2w.shape[-2], 4)  # (N, 4, 4)


if __name__ == '__main__':
    unittest.main()
