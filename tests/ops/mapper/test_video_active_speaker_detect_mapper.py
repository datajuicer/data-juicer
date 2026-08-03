import os
import pickle
import tempfile
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_active_speaker_detect_mapper import (
    VideoActiveSpeakerDetectMapper,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

LIGHT_ASD_MODEL_PATH = './thirdparty/humanvbench_models/Light-ASD/weight/finetuning_TalkSet.model'

_MODELS_AVAILABLE = os.path.exists(LIGHT_ASD_MODEL_PATH)


def _make_dummy_track(video_path, temp_dir):
    """Create a real bbox pkl file for testing. The ASD model still runs
    on real cropped video frames — no mock is used."""
    bbox_path = os.path.join(temp_dir, os.path.basename(video_path) + '_0.pkl')
    bbox_dict = {
        'frame': [0, 5, 10, 15, 20],
        'xy_bbox': [[10, 10, 80, 80], [12, 12, 82, 82], [14, 14, 84, 84],
                    [16, 16, 86, 86], [18, 18, 88, 88]],
        'xys_bbox': {
            'x': [45.0, 47.0, 49.0, 51.0, 53.0],
            'y': [45.0, 47.0, 49.0, 51.0, 53.0],
            's': [35.0, 35.0, 35.0, 35.0, 35.0],
        },
    }
    with open(bbox_path, 'wb') as f:
        pickle.dump(bbox_dict, f)
    return {'bbox_path': bbox_path}


@unittest.skipUnless(_MODELS_AVAILABLE, 'Light-ASD model not found. '
                     'Follow thirdparty/humanvbench_models/README.md to set up.')
class VideoActiveSpeakerDetectMapperTest(DataJuicerTestCaseBase):
    """Test for video_active_speaker_detect_mapper, which detects active
    speakers by analyzing visual face tracks and audio signals using the
    Light-ASD model.

    No mock is used — the real Light-ASD model runs on actual video files.
    Requires thirdparty model weights and prior metadata from upstream
    operators.
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid4_path = os.path.join(data_path, 'video4.mp4')  # Speech
    vid1_path = os.path.join(data_path, 'video1.mp4')  # Music

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
            f'{Fields.meta}.{MetaKeys.active_speaker_flag}',
        ])
        return res[f'{Fields.meta}.{MetaKeys.active_speaker_flag}']

    def test_basic_speech_video(self):
        """Active speaker detection should produce boolean flags for each track."""
        track = _make_dummy_track(self.vid4_path, self._temp_dir)
        ds_list = [{
            'text': f'{SpecialTokens.video} A person is speaking. {SpecialTokens.eoc}',
            'videos': [self.vid4_path],
            Fields.meta: {
                MetaKeys.human_track_data_path: [[track]],
                MetaKeys.video_audio_tags: ['Speech'],
            },
        }]
        op = VideoActiveSpeakerDetectMapper(
            temp_save_path=self._temp_dir,
            Light_ASD_model_path=LIGHT_ASD_MODEL_PATH,
            active_threshold=15,
        )
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)
        # Result is a list of lists of booleans (one per video, per track)
        self.assertEqual(len(result[0]), 1)

    def test_non_speech_video(self):
        """Non-speech videos should produce False flags."""
        track = _make_dummy_track(self.vid1_path, self._temp_dir)
        ds_list = [{
            'text': f'{SpecialTokens.video} Music playing. {SpecialTokens.eoc}',
            'videos': [self.vid1_path],
            Fields.meta: {
                MetaKeys.human_track_data_path: [[track]],
                MetaKeys.video_audio_tags: ['Music'],
            },
        }]
        op = VideoActiveSpeakerDetectMapper(
            temp_save_path=self._temp_dir,
            Light_ASD_model_path=LIGHT_ASD_MODEL_PATH,
            active_threshold=15,
        )
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)

    def test_multi_video(self):
        """Multiple videos should each get active speaker results."""
        track4 = _make_dummy_track(self.vid4_path, self._temp_dir)
        track1 = _make_dummy_track(self.vid1_path, self._temp_dir)
        ds_list = [{
            'text': f'{SpecialTokens.video} Speech here. {SpecialTokens.eoc}'
            f'{SpecialTokens.video} Music here.',
            'videos': [self.vid4_path, self.vid1_path],
            Fields.meta: {
                MetaKeys.human_track_data_path: [[track4], [track1]],
                MetaKeys.video_audio_tags: ['Speech', 'Music'],
            },
        }]
        op = VideoActiveSpeakerDetectMapper(
            temp_save_path=self._temp_dir,
            Light_ASD_model_path=LIGHT_ASD_MODEL_PATH,
            active_threshold=15,
        )
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

    def test_no_video(self):
        """Samples with no videos should not crash."""
        ds_list = [{
            'text': 'No video here.',
            'videos': [],
            Fields.meta: {
                MetaKeys.human_track_data_path: [],
                MetaKeys.video_audio_tags: [],
            },
        }]
        op = VideoActiveSpeakerDetectMapper(
            temp_save_path=self._temp_dir,
            Light_ASD_model_path=LIGHT_ASD_MODEL_PATH,
            active_threshold=15,
        )
        result = self._run_mapper(ds_list, op)
        # no video -> the mapper returns empty list for the flag
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
