import json
import os
import tempfile
import unittest

from data_juicer.utils.constant import Fields
from data_juicer.utils.jsonl_lenient_loader import (
    dataset_from_lenient_jsonl_files,
    iter_lenient_jsonl_records,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class JsonlLenientLoaderTest(DataJuicerTestCaseBase):
    """Tests for jsonl_lenient_loader module."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_iter_lenient_jsonl_skips_bad_lines(self):
        """iter_lenient_jsonl_records should skip invalid JSON and non-object lines."""
        jsonl_path = os.path.join(self.tmp_dir, "test.jsonl")
        huge_int = 2**65  # ujson may reject; stdlib ok
        lines = [
            json.dumps({"ok": 1, "id": huge_int}),
            "not json",
            json.dumps(["not", "an", "object"]),
            json.dumps({"ok": 2}),
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        rows = list(
            iter_lenient_jsonl_records(
                [(jsonl_path, ".jsonl")],
                add_suffix_column=False,
            )
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ok"], 1)
        self.assertEqual(rows[0]["id"], huge_int)
        self.assertEqual(rows[1]["ok"], 2)

    def test_iter_lenient_jsonl_adds_suffix(self):
        """iter_lenient_jsonl_records should add suffix column when requested."""
        jsonl_path = os.path.join(self.tmp_dir, "test.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"x": 1}) + "\n")

        rows = list(
            iter_lenient_jsonl_records(
                [(jsonl_path, ".jsonl")],
                add_suffix_column=True,
            )
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][Fields.suffix], ".jsonl")

    def test_dataset_from_lenient_jsonl_files(self):
        """dataset_from_lenient_jsonl_files should build a Dataset skipping bad lines."""
        jsonl_path = os.path.join(self.tmp_dir, "test.jsonl")
        body = (
            json.dumps({"a": 1}) + "\n" + "oops\n" + json.dumps({"a": 2}) + "\n"
        )
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(body)

        ds = dataset_from_lenient_jsonl_files(
            [(jsonl_path, ".jsonl")],
            add_suffix_column=False,
        )
        self.assertEqual(len(ds), 2)
        self.assertEqual(list(ds["a"]), [1, 2])

    def test_handles_big_integer(self):
        """Should handle big integers that ujson rejects."""
        jsonl_path = os.path.join(self.tmp_dir, "big_int.jsonl")
        big_int = 9999999999999999999999999999999999999999
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "hello", "big": big_int}) + "\n")

        rows = list(
            iter_lenient_jsonl_records(
                [(jsonl_path, ".jsonl")],
                add_suffix_column=False,
            )
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("big", rows[0])

    def test_handles_gzip_file(self):
        """Should handle .jsonl.gz files."""
        import gzip

        jsonl_path = os.path.join(self.tmp_dir, "test.jsonl.gz")
        with gzip.open(jsonl_path, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"x": 1}) + "\n")
            f.write(json.dumps({"x": 2}) + "\n")

        rows = list(
            iter_lenient_jsonl_records(
                [(jsonl_path, ".jsonl.gz")],
                add_suffix_column=False,
            )
        )
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
