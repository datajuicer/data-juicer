import unittest

from data_juicer.ops.load import load_ops
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LoadOPsTest(DataJuicerTestCaseBase):
    """Test load_ops: instantiates operators from process config list."""

    def test_load_single_op(self):
        process_list = [
            {"clean_email_mapper": {}},
        ]
        ops = load_ops(process_list)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]._name, "clean_email_mapper")

    def test_load_multiple_ops(self):
        process_list = [
            {"clean_email_mapper": {}},
            {"fix_unicode_mapper": {}},
        ]
        ops = load_ops(process_list)
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0]._name, "clean_email_mapper")
        self.assertEqual(ops[1]._name, "fix_unicode_mapper")

    def test_op_with_args(self):
        process_list = [
            {"text_length_filter": {"min_len": 10, "max_len": 1000}},
        ]
        ops = load_ops(process_list)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].min_len, 10)
        self.assertEqual(ops[0].max_len, 1000)

    def test_op_cfg_stored(self):
        """Each op should have its config stored in _op_cfg."""
        cfg = {"clean_email_mapper": {"repl": "<EMAIL>"}}
        ops = load_ops([cfg])
        self.assertEqual(ops[0]._op_cfg, cfg)

    def test_empty_process_list(self):
        ops = load_ops([])
        self.assertEqual(ops, [])

    def test_op_order_preserved(self):
        process_list = [
            {"fix_unicode_mapper": {}},
            {"clean_email_mapper": {}},
            {"whitespace_normalization_mapper": {}},
        ]
        ops = load_ops(process_list)
        names = [op._name for op in ops]
        self.assertEqual(names, [
            "fix_unicode_mapper",
            "clean_email_mapper",
            "whitespace_normalization_mapper",
        ])


if __name__ == '__main__':
    unittest.main()

