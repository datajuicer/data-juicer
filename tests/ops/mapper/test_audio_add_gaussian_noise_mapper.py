import os
import unittest

from data_juicer.utils.constant import Fields

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.audio_add_gaussian_noise_mapper import AudioAddGaussianNoiseMapper

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class AudioAddGaussianNoiseMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    audio1_path = os.path.join(data_path, 'audio1.wav')
    audio2_path = os.path.join(data_path, 'audio2.wav')
    audio3_path = os.path.join(data_path, 'audio3.ogg')

    def _run_mapper(self, op, source_list):
        dataset = Dataset.from_list(source_list)
        dataset = dataset.map(op.process)
        res_list = dataset.to_list()
        for source, res in zip(source_list, res_list):
            for src_path, res_path in zip(source[op.audio_key], res[op.audio_key]):
                self.assertNotEqual(src_path, res_path)
            self.assertIn(Fields.source_file, res)

    def test_single_audio(self):
        ds_list = [{
            'audios': [self.audio1_path]
        }, {
            'audios': [self.audio2_path]
        }, {
            'audios': [self.audio3_path]
        }]
        op = AudioAddGaussianNoiseMapper()
        self._run_mapper(op, ds_list)

    def test_multiple_audios(self):
        ds_list = [{
            'audios': [self.audio1_path, self.audio2_path]
        }, {
            'audios': [self.audio2_path, self.audio3_path]
        }, {
            'audios': [self.audio3_path, self.audio1_path]
        }]
        op = AudioAddGaussianNoiseMapper()
        self._run_mapper(op, ds_list)


if __name__ == '__main__':
    unittest.main()
