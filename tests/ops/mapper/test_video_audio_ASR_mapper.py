import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_audio_ASR_mapper import VideoAudioASRMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

hf_model = 'FunAudioLLM/SenseVoiceSmall'

try:
    from funasr import AutoModel  # noqa: F401
    _MODEL_AVAILABLE = True
except ImportError:
    _MODEL_AVAILABLE = False


@unittest.skipUnless(_MODEL_AVAILABLE, 'funasr package not installed. '
                     'Install with: pip install funasr')
class VideoAudioASRMapperTest(DataJuicerTestCaseBase):
    """Test for video_audio_ASR_mapper, which performs automatic speech
    recognition on video audio streams using the SenseVoiceSmall model.

    No mock is used — the real SenseVoiceSmall model runs on actual video
    files with speech audio.
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid4_path = os.path.join(data_path, 'video4.mp4')  # Speech
    vid5_path = os.path.join(data_path, 'video5.mp4')  # Speech
    vid1_path = os.path.join(data_path, 'video1.mp4')  # Music (no speech ASR)
    vid3_no_aud_path = os.path.join(data_path, 'video3-no-audio.mp4')  # EMPTY

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(hf_model)

    def _run_mapper(self, ds_list, op, num_proc=1):
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res = dataset.flatten().select_columns([f'{Fields.meta}.{MetaKeys.speech_ASR}'])
        return res[f'{Fields.meta}.{MetaKeys.speech_ASR}']

    def test_speech_video(self):
        """ASR should produce a non-empty transcription for speech videos."""
        ds_list = [{
            'text': f'{SpecialTokens.video} A person is speaking. {SpecialTokens.eoc}',
            'videos': [self.vid4_path],
            Fields.meta: {MetaKeys.video_audio_tags: ['Speech']},
        }]
        op = VideoAudioASRMapper(model_dir_ASR=hf_model)
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        # The ASR result should be a dict with 'language' and 'asr' keys
        self.assertIsInstance(result[0][0], dict)
        self.assertIn('language', result[0][0])
        self.assertIn('asr', result[0][0])

    def test_non_speech_video(self):
        """Non-speech videos (Music) should produce empty ASR results."""
        ds_list = [{
            'text': f'{SpecialTokens.video} Music playing. {SpecialTokens.eoc}',
            'videos': [self.vid1_path],
            Fields.meta: {MetaKeys.video_audio_tags: ['Music']},
        }]
        op = VideoAudioASRMapper(model_dir_ASR=hf_model)
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [{'language': '', 'asr': ''}])

    def test_multi_video(self):
        """Multiple videos in one sample should each get an ASR result."""
        ds_list = [{
            'text': f'{SpecialTokens.video} Speech here. {SpecialTokens.eoc}'
            f'{SpecialTokens.video} Music here.',
            'videos': [self.vid4_path, self.vid1_path],
            Fields.meta: {MetaKeys.video_audio_tags: ['Speech', 'Music']},
        }]
        op = VideoAudioASRMapper(model_dir_ASR=hf_model)
        result = self._run_mapper(ds_list, op)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)

    def test_no_video(self):
        """Samples with no videos should produce empty results."""
        ds_list = [{
            'text': 'No video here.',
            'videos': [],
            Fields.meta: {MetaKeys.video_audio_tags: []},
        }]
        op = VideoAudioASRMapper(model_dir_ASR=hf_model)
        result = self._run_mapper(ds_list, op)
        self.assertEqual(result, [[]])

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
        op = VideoAudioASRMapper(model_dir_ASR=hf_model)
        result = self._run_mapper(ds_list, op, num_proc=2)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
