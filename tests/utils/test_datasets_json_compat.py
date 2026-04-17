"""Unit tests for datasets_json_compat.py monkey patch."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

_ENV_FLAG = "DATA_JUICER_USE_STDLIB_JSON"


class DatasetsJsonCompatTest(DataJuicerTestCaseBase):
    """Tests for apply_stdlib_json_patch_for_datasets()."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        # Clean env before each test
        os.environ.pop(_ENV_FLAG, None)
        # Reset the module's _PATCHED flag
        import data_juicer.utils.datasets_json_compat as mod
        mod._PATCHED = False

    def tearDown(self):
        os.environ.pop(_ENV_FLAG, None)
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_patch_not_applied_without_env(self):
        """Without env flag, patch should not be applied."""
        from data_juicer.utils.datasets_json_compat import apply_stdlib_json_patch_for_datasets

        result = apply_stdlib_json_patch_for_datasets()
        self.assertFalse(result)

    def test_patch_applied_with_env(self):
        """With env flag, patch should be applied."""
        os.environ[_ENV_FLAG] = "1"
        from data_juicer.utils.datasets_json_compat import apply_stdlib_json_patch_for_datasets

        result = apply_stdlib_json_patch_for_datasets()
        self.assertTrue(result)

    def test_patch_is_idempotent(self):
        """Calling patch multiple times should be safe."""
        os.environ[_ENV_FLAG] = "1"
        from data_juicer.utils.datasets_json_compat import apply_stdlib_json_patch_for_datasets

        result1 = apply_stdlib_json_patch_for_datasets()
        result2 = apply_stdlib_json_patch_for_datasets()
        self.assertTrue(result1)
        self.assertTrue(result2)

    def test_both_modules_are_patched(self):
        """Both datasets.utils.json and datasets.packaged_modules.json.json should be patched."""
        os.environ[_ENV_FLAG] = "1"
        from data_juicer.utils.datasets_json_compat import apply_stdlib_json_patch_for_datasets
        apply_stdlib_json_patch_for_datasets()

        import datasets.utils.json as ds_json
        import importlib
        ds_json_loader = importlib.import_module("datasets.packaged_modules.json.json")

        self.assertEqual(ds_json.ujson_loads.__name__, "_stdlib_loads")
        self.assertEqual(ds_json_loader.ujson_loads.__name__, "_stdlib_loads")
        # Same function object
        self.assertIs(ds_json.ujson_loads, ds_json_loader.ujson_loads)

    def test_stdlib_loads_handles_bytes(self):
        """_stdlib_loads should accept bytes input."""
        os.environ[_ENV_FLAG] = "1"
        from data_juicer.utils.datasets_json_compat import apply_stdlib_json_patch_for_datasets
        apply_stdlib_json_patch_for_datasets()

        import datasets.utils.json as ds_json
        data = b'{"key": "value"}'
        result = ds_json.ujson_loads(data)
        self.assertEqual(result, {"key": "value"})

    def test_stdlib_loads_handles_str(self):
        """_stdlib_loads should accept string input."""
        os.environ[_ENV_FLAG] = "1"
        from data_juicer.utils.datasets_json_compat import apply_stdlib_json_patch_for_datasets
        apply_stdlib_json_patch_for_datasets()

        import datasets.utils.json as ds_json
        data = '{"key": "value"}'
        result = ds_json.ujson_loads(data)
        self.assertEqual(result, {"key": "value"})

    def test_stdlib_loads_handles_big_integer(self):
        """_stdlib_loads should handle big integers that ujson rejects."""
        os.environ[_ENV_FLAG] = "1"
        from data_juicer.utils.datasets_json_compat import apply_stdlib_json_patch_for_datasets
        apply_stdlib_json_patch_for_datasets()

        import datasets.utils.json as ds_json
        big_int = 9999999999999999999999999999999999999999
        data = f'{{"big": {big_int}}}'
        result = ds_json.ujson_loads(data)
        # stdlib json parses big ints as floats, but doesn't raise
        self.assertIn("big", result)

    def test_load_dataset_with_big_int_succeeds(self):
        """End-to-end: load_dataset should succeed with big integers when patch is applied."""
        os.environ[_ENV_FLAG] = "1"
        from data_juicer.utils.datasets_json_compat import apply_stdlib_json_patch_for_datasets
        apply_stdlib_json_patch_for_datasets()

        # Create a JSONL file with big integer
        jsonl_path = os.path.join(self.tmp_dir, "big_int.jsonl")
        big_int = 9999999999999999999999999999999999999999
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "hello", "big": big_int}) + "\n")

        from datasets import load_dataset
        ds = load_dataset("json", data_files=jsonl_path, split="train")
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
