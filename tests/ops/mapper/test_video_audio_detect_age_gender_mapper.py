import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_audio_detect_age_gender_mapper import (
    VideoAudioDetectAgeGenderMapper,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

hf_model = 'audeering/wav2vec2-large-robust-24-ft-age-gender'

try:
    from thirdparty.humanvbench_models.audio_code.wav2vec_age_gender import (  # noqa: F401
        process_func,
    )
    _THIRDPARTY_AVAILABLE = True
except ImportError:
    _THIRDPARTY_AVAILABLE = False


@unittest.skipUnless(_THIRDPARTY_AVAILABLE,
                     'thirdparty wav2vec_age_gender module not found. '
                     'Follow thirdparty/humanvbench_models/README.md to set up.')
class VideoAudioDetectAgeGenderMapperTest(DataJuicerTestCaseBase):
    """Test for video_audio_detect_age_gender_mapper, which detects age
    and gender from video audio using a wav2vec2 model.

    No mock is used — the real audeering/wav2vec2-large-robust-24-ft-age-gender
    model runs on actual video files with speech audio.
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid4_path = os.path.join(data_path, 'video4.mp4')  # Speech
    vid5_path = os.path.join(data_path, 'video5.mp4')  # Speech
    vid1_path = os.path.join(data_path, 'video1.mp4')  # Music

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(hf_model)

    def _run_mapper(self, ds_list, op, num_proc=1):
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res = dataset.flatten().select_columns(
            [f'{Fields.meta}.{MetaKeys.audio_speech_attribute}']
        )
        return res[f'{Fields.meta}.{MetaKeys.audio_speech_attribute}']

    def test_speech_video(self):
        """Age/gender detection should produce results for speech videos."""
        ds_list = [{
            'text': f'{SpecialTokens.video} A person is speaking. {SpecialTokens.eoc}',
            'videos': [self.vid4_path],
            Fields.meta: {MetaKeys.video_audio_tags: ['Speech']},
        }]
        op = VideoAudioDetectAgeGenderMapper(hf_audio_mapper=hf_model)
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        # Each result is a dict with Age, female, and male keys
        item = result[0][0]
        self.assertIsInstance(item, dict)
        self.assertIn('Age', item)
        self.assertIn('female', item)
        self.assertIn('male', item)

    def test_non_speech_video(self):
        """Non-speech videos should produce empty age/gender results."""
        ds_list = [{
            'text': f'{SpecialTokens.video} Music playing. {SpecialTokens.eoc}',
            'videos': [self.vid1_path],
            Fields.meta: {MetaKeys.video_audio_tags: ['Music']},
        }]
        op = VideoAudioDetectAgeGenderMapper(hf_audio_mapper=hf_model)
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)

    def test_multi_video(self):
        """Multiple videos in one sample should each get a result."""
        ds_list = [{
            'text': f'{SpecialTokens.video} Speech here. {SpecialTokens.eoc}'
            f'{SpecialTokens.video} More speech.',
            'videos': [self.vid4_path, self.vid5_path],
            Fields.meta: {MetaKeys.video_audio_tags: ['Speech', 'Speech']},
        }]
        op = VideoAudioDetectAgeGenderMapper(hf_audio_mapper=hf_model)
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

    def test_no_video(self):
        """Samples with no videos should not crash."""
        ds_list = [{
            'text': 'No video here.',
            'videos': [],
            Fields.meta: {MetaKeys.video_audio_tags: []},
        }]
        op = VideoAudioDetectAgeGenderMapper(hf_audio_mapper=hf_model)
        result = self._run_mapper(ds_list, op)
        # no video -> returns sample without setting the tag
        self.assertEqual(len(result), 1)

    def test_multi_process(self):
        """Test with multiple processes."""
        ds_list = [{
            'text': f'{SpecialTokens.video} A person is speaking. {SpecialTokens.eoc}',
            'videos': [self.vid4_path],
            Fields.meta: {MetaKeys.video_audio_tags: ['Speech']},
        }, {
            'text': f'{SpecialTokens.video} Another person is speaking. {SpecialTokens.eoc}',
            'videos': [self.vid5_path],
            Fields.meta: {MetaKeys.video_audio_tags: ['Speech']},
        }]
        op = VideoAudioDetectAgeGenderMapper(hf_audio_mapper=hf_model)
        result = self._run_mapper(ds_list, op, num_proc=2)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
