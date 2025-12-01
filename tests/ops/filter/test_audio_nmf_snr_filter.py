import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.audio_nmf_snr_filter import AudioNMFSNRFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class AudioNMFSNRFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    aud1_path = os.path.join(data_path, 'audio1.wav')  # about -7dB
    aud2_path = os.path.join(data_path, 'audio2.wav')  # about 6dB
    aud3_path = os.path.join(data_path, 'audio3.ogg')  # about 3dB

    def _run_audio_nmf_snr_filter(self,
                                  dataset: Dataset,
                                  target_list,
                                  op,
                                  np=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=np)
        dataset = dataset.filter(op.process, num_proc=np)
        dataset = dataset.select_columns(column_names=[op.audio_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_default_filter(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = AudioNMFSNRFilter(min_snr=-1000, max_snr=1000)
        self._run_audio_nmf_snr_filter(dataset, tgt_list, op)

    def test_filter_low_snr_audios(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud2_path]}, {'audios': [self.aud3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = AudioNMFSNRFilter(min_snr=0)
        self._run_audio_nmf_snr_filter(dataset, tgt_list, op)

    def test_filter_high_snr_audios(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud1_path]}, {'audios': [self.aud3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = AudioNMFSNRFilter(min_snr=-1000, max_snr=5)
        self._run_audio_nmf_snr_filter(dataset, tgt_list, op)

    def test_filter_audios_within_range(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = AudioNMFSNRFilter(min_snr=0, max_snr=5)
        self._run_audio_nmf_snr_filter(dataset, tgt_list, op)

    def test_any(self):

        ds_list = [{
            'audios': [self.aud1_path, self.aud2_path]
        }, {
            'audios': [self.aud2_path, self.aud3_path]
        }, {
            'audios': [self.aud1_path, self.aud3_path]
        }]
        tgt_list = [{
            'audios': [self.aud2_path, self.aud3_path]
        }, {
            'audios': [self.aud1_path, self.aud3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = AudioNMFSNRFilter(min_snr=0, max_snr=5, any_or_all='any')
        self._run_audio_nmf_snr_filter(dataset, tgt_list, op)

    def test_all(self):

        ds_list = [{
            'audios': [self.aud1_path, self.aud2_path]
        }, {
            'audios': [self.aud2_path, self.aud3_path]
        }, {
            'audios': [self.aud1_path, self.aud3_path]
        }]
        tgt_list = []
        dataset = Dataset.from_list(ds_list)
        op = AudioNMFSNRFilter(min_snr=0, max_snr=5, any_or_all='all')
        self._run_audio_nmf_snr_filter(dataset, tgt_list, op)

    def test_filter_in_parallel(self):

        ds_list = [{
            'audios': [self.aud1_path]
        }, {
            'audios': [self.aud2_path]
        }, {
            'audios': [self.aud3_path]
        }]
        tgt_list = [{'audios': [self.aud3_path]}]
        dataset = Dataset.from_list(ds_list)
        op = AudioNMFSNRFilter(min_snr=0, max_snr=5, any_or_all='any')
        self._run_audio_nmf_snr_filter(dataset, tgt_list, op, np=2)


if __name__ == '__main__':
    unittest.main()
