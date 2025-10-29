import os
import unittest
import json

from data_juicer.core.data import NestedDataset
from data_juicer.utils.ckpt_utils import CheckpointManager
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class CkptUtilsTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.temp_output_path = 'tmp/test_ckpt_utils/'

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')
        super().tearDown()

    def test_basic_func(self):
        ckpt_path = os.path.join(self.temp_output_path, 'ckpt_1')
        manager = CheckpointManager(ckpt_path, original_process_list=[
            {'test_op_1': {'test_key': 'test_value_1'}},
            {'test_op_2': {'test_key': 'test_value_2'}},
        ])
        self.assertEqual(manager.get_left_process_list(), [
            {'test_op_1': {'test_key': 'test_value_1'}},
            {'test_op_2': {'test_key': 'test_value_2'}},
        ])
        self.assertFalse(manager.ckpt_available)

        self.assertFalse(manager.check_ckpt())
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(ckpt_path, 'latest'), exist_ok=True)
        with open(os.path.join(ckpt_path, 'ckpt_op.json'), 'w') as fout:
            json.dump([
                {'test_op_1': {'test_key': 'test_value_1'}},
            ], fout)
        self.assertTrue(manager.check_ops_to_skip())

        manager = CheckpointManager(ckpt_path, original_process_list=[
            {'test_op_1': {'test_key': 'test_value_1'}},
            {'test_op_2': {'test_key': 'test_value_2'}},
        ])
        with open(os.path.join(ckpt_path, 'ckpt_op.json'), 'w') as fout:
            json.dump([
                {'test_op_1': {'test_key': 'test_value_1'}},
                {'test_op_2': {'test_key': 'test_value_2'}},
            ], fout)
        self.assertFalse(manager.check_ops_to_skip())

    def test_different_ops(self):
        ckpt_path = os.path.join(self.temp_output_path, 'ckpt_2')
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(os.path.join(ckpt_path, 'latest'), exist_ok=True)
        with open(os.path.join(ckpt_path, 'ckpt_op.json'), 'w') as fout:
            json.dump([
                {'test_op_2': {'test_key': 'test_value_2'}},
            ], fout)
        manager = CheckpointManager(ckpt_path, original_process_list=[
            {'test_op_1': {'test_key': 'test_value_1'}},
            {'test_op_2': {'test_key': 'test_value_2'}},
        ])
        self.assertFalse(manager.ckpt_available)

    def test_save_and_load_ckpt(self):
        ckpt_path = os.path.join(self.temp_output_path, 'ckpt_3')
        test_data = {
            'text': ['text1', 'text2', 'text3'],
        }
        dataset = NestedDataset.from_dict(test_data)
        manager = CheckpointManager(ckpt_path, original_process_list=[])
        self.assertFalse(os.path.exists(os.path.join(manager.ckpt_ds_dir, 'dataset_info.json')))
        manager.record({'test_op_1': {'test_key': 'test_value_1'}})
        manager.save_ckpt(dataset)
        self.assertTrue(os.path.exists(os.path.join(manager.ckpt_ds_dir, 'dataset_info.json')))
        self.assertTrue(os.path.exists(manager.ckpt_op_record))
        loaded_ckpt = manager.load_ckpt()
        self.assertDatasetEqual(dataset, loaded_ckpt)


if __name__ == '__main__':
    unittest.main()
