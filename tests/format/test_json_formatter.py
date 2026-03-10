import os
import unittest
import gzip
import tempfile
import shutil

from data_juicer.format.json_formatter import JsonFormatter
from data_juicer.format.load import load_formatter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

try:
    import zstandard as zstd  # type: ignore

    HAS_ZSTD = True
except Exception:
    zstd = None
    HAS_ZSTD = False


class JsonFormatterTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        self._path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "structured")
        self._file = os.path.join(self._path, "demo-dataset.jsonl")
        print(self._file)
        # create compressed variants for testing
        # create a temp directory to hold generated compressed files
        self._temp_dir = tempfile.mkdtemp()
        with open(self._file, "rb") as f:
            raw = f.read()

        # .jsonl.gz
        self._jsonl_gz = os.path.join(self._temp_dir, "demo-dataset.jsonl.gz")
        with gzip.open(self._jsonl_gz, "wb") as f:
            f.write(raw)

        # .json.gz (same content, different suffix)
        self._json_gz = os.path.join(self._temp_dir, "demo-dataset.json.gz")
        with gzip.open(self._json_gz, "wb") as f:
            f.write(raw)

        # .json.zst and .jsonl.zst if zstandard available
        if HAS_ZSTD:
            self._jsonl_zst = os.path.join(self._temp_dir, "demo-dataset.jsonl.zst")
            self._json_zst = os.path.join(self._temp_dir, "demo-dataset.json.zst")
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(raw)
            with open(self._jsonl_zst, "wb") as f:
                f.write(compressed)
            with open(self._json_zst, "wb") as f:
                f.write(compressed)

    def test_json_file(self):
        formatter = JsonFormatter(self._file)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ["text", "meta"])

    def test_json_path(self):
        formatter = JsonFormatter(self._path)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ["text", "meta"])

    def test_load_formatter_with_file(self):
        """Test load_formatter with a direct file path"""
        formatter = load_formatter(self._file)
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ["text", "meta"])

    def test_load_formatter_with_specified_suffix(self):
        """Test load_formatter with specified suffixes"""
        formatter = load_formatter(self._path, suffixes=[".jsonl"])
        self.assertIsInstance(formatter, JsonFormatter)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ["text", "meta"])

    def tearDown(self):
        # cleanup temp dir and files
        if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
        super().tearDown()

    def test_jsonl_gz_file(self):
        formatter = JsonFormatter(self._jsonl_gz)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ["text", "meta"])

    def test_json_gz_file(self):
        formatter = JsonFormatter(self._json_gz)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ["text", "meta"])

    @unittest.skipUnless(HAS_ZSTD, "zstandard not installed")
    def test_json_zst_file(self):
        formatter = JsonFormatter(self._json_zst)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ["text", "meta"])

    @unittest.skipUnless(HAS_ZSTD, "zstandard not installed")
    def test_jsonl_zst_file(self):
        formatter = JsonFormatter(self._jsonl_zst)
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 6)
        self.assertEqual(list(ds.features.keys()), ["text", "meta"])


if __name__ == "__main__":
    unittest.main()
