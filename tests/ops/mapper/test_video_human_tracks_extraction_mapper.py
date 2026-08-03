import os
import tempfile
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_human_tracks_extraction_mapper import (
    VideoHumanTracksExtractionMapper,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

S3FD_MODEL_PATH = './thirdparty/humanvbench_models/Light-ASD/model/faceDetector/s3fd/sfd_face.pth'
YOLOV8_MODEL_PATH = './thirdparty/humanvbench_models/YOLOv8_human/weights/best.pt'

_MODELS_AVAILABLE = os.path.exists(S3FD_MODEL_PATH) and os.path.exists(YOLOV8_MODEL_PATH)

try:
    from thirdparty.humanvbench_models.YOLOv8_human.dj import demo  # noqa: F401
    _MODULE_AVAILABLE = True
except ImportError:
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODELS_AVAILABLE and _MODULE_AVAILABLE,
                     'Thirdparty models or modules not found. '
                     'Follow thirdparty/humanvbench_models/README.md to set up.')
class VideoHumanTracksExtractionMapperTest(DataJuicerTestCaseBase):
    """Test for video_human_tracks_extraction_mapper, which extracts face
    and human bounding box tracks from videos using S3FD and YOLOv8.

    No mock is used — real S3FD and YOLOv8 models run on actual video files.
    Requires thirdparty model weights to be present.
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
            f'{Fields.meta}.{MetaKeys.human_track_data_path}',
            f'{Fields.meta}.{MetaKeys.number_people_in_video}',
        ])
        return res

    def test_single_video(self):
        """Track extraction should produce track data and people count."""
        ds_list = [{
            'text': f'{SpecialTokens.video} A person is speaking. {SpecialTokens.eoc}',
            'videos': [self.vid4_path],
            Fields.meta: {},
        }]
        op = VideoHumanTracksExtractionMapper(
            face_track_bbox_path=self._temp_dir,
            YOLOv8_human_model_path=YOLOV8_MODEL_PATH,
            face_detect_S3FD_model_path=S3FD_MODEL_PATH,
        )
        res = self._run_mapper(ds_list, op)
        tracks = res[f'{Fields.meta}.{MetaKeys.human_track_data_path}']
        people = res[f'{Fields.meta}.{MetaKeys.number_people_in_video}']
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(people), 1)
        # Each video should have at least one track (if a person is detected)
        if len(tracks[0]) > 0:
            # Each track should have a bbox_path
            for track in tracks[0]:
                self.assertIn('bbox_path', track)

    def test_multi_video(self):
        """Multiple videos should each get track results."""
        ds_list = [{
            'text': f'{SpecialTokens.video} Person 1. {SpecialTokens.eoc}'
            f'{SpecialTokens.video} Person 2.',
            'videos': [self.vid4_path, self.vid5_path],
            Fields.meta: {},
        }]
        op = VideoHumanTracksExtractionMapper(
            face_track_bbox_path=self._temp_dir,
            YOLOv8_human_model_path=YOLOV8_MODEL_PATH,
            face_detect_S3FD_model_path=S3FD_MODEL_PATH,
        )
        res = self._run_mapper(ds_list, op)
        tracks = res[f'{Fields.meta}.{MetaKeys.human_track_data_path}']
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]), 2)

    def test_no_video(self):
        """Samples with no videos should not crash."""
        ds_list = [{
            'text': 'No video here.',
            'videos': [],
            Fields.meta: {},
        }]
        op = VideoHumanTracksExtractionMapper(
            face_track_bbox_path=self._temp_dir,
            YOLOv8_human_model_path=YOLOV8_MODEL_PATH,
            face_detect_S3FD_model_path=S3FD_MODEL_PATH,
        )
        res = self._run_mapper(ds_list, op)
        tracks = res[f'{Fields.meta}.{MetaKeys.human_track_data_path}']
        self.assertEqual(tracks, [[]])


if __name__ == '__main__':
    unittest.main()
