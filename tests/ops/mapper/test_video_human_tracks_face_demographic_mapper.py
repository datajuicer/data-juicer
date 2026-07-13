import os
import pickle
import tempfile
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_human_tracks_face_demographic_mapper import (
    VideoHumantrackFaceDemographicMapper,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

try:
    from deepface import DeepFace  # noqa: F401
    _DEEPFACE_AVAILABLE = True
except (ImportError, RuntimeError):
    _DEEPFACE_AVAILABLE = False


def _make_dummy_track(video_path, temp_dir):
    """Create a dummy bbox pkl file for testing face demographic mapper.

    This is NOT a mock — it creates real pickle files that the operator
    reads. The operator's DeepFace analysis still runs on real video frames.
    """
    bbox_path = os.path.join(temp_dir, os.path.basename(video_path) + '_0.pkl')
    bbox_dict = {
        'frame': [0, 5, 10],
        'xy_bbox': [[10, 10, 80, 80], [12, 12, 82, 82], [14, 14, 84, 84]],
        'xys_bbox': {
            'x': [45.0, 47.0, 49.0],
            'y': [45.0, 47.0, 49.0],
            's': [35.0, 35.0, 35.0],
        },
    }
    with open(bbox_path, 'wb') as f:
        pickle.dump(bbox_dict, f)
    return {'bbox_path': bbox_path}


@unittest.skipUnless(_DEEPFACE_AVAILABLE, 'DeepFace runtime dependencies are not available.')
class VideoHumantrackFaceDemographicMapperTest(DataJuicerTestCaseBase):
    """Test for video_human_tracks_face_demographic_mapper, which detects
    face demographics (age, gender, race) for each tracked person using
    DeepFace.

    No mock is used — the real DeepFace model runs on actual video frames
    cropped from real video files.
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid4_path = os.path.join(data_path, 'video4.mp4')
    vid5_path = os.path.join(data_path, 'video5.mp4')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree(cls._temp_dir, ignore_errors=True)
        super().tearDownClass()

    def _run_mapper(self, ds_list, op, num_proc=1):
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res = dataset.flatten().select_columns([
            f'{Fields.meta}.{MetaKeys.video_facetrack_attribute_demographic}',
        ])
        return res[f'{Fields.meta}.{MetaKeys.video_facetrack_attribute_demographic}']

    def test_single_video(self):
        """Face demographic detection should produce results for each track."""
        track = _make_dummy_track(self.vid4_path, self._temp_dir)
        ds_list = [{
            'text': f'{SpecialTokens.video} A person is speaking. {SpecialTokens.eoc}',
            'videos': [self.vid4_path],
            Fields.meta: {MetaKeys.human_track_data_path: [[track]]},
        }]
        op = VideoHumantrackFaceDemographicMapper(
            original_data_save_path=self._temp_dir,
            detect_interval=5,
        )
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)

    def test_no_video(self):
        """Samples with no videos should not crash."""
        ds_list = [{
            'text': 'No video here.',
            'videos': [],
            Fields.meta: {MetaKeys.human_track_data_path: []},
        }]
        op = VideoHumantrackFaceDemographicMapper(
            original_data_save_path=self._temp_dir,
            detect_interval=5,
        )
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
