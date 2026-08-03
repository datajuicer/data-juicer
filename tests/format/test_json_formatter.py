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


class JsonFormatterLenientTest(DataJuicerTestCaseBase):
    """Test JsonFormatter's lenient JSONL loading with real files."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.environ.pop("DATA_JUICER_JSONL_LENIENT", None)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def _write_jsonl(self, filename, lines):
        import json
        path = os.path.join(self.tmp_dir, filename)
        with open(path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return path

    def test_lenient_env_var(self):
        self._write_jsonl("data.jsonl", [
            {"text": "hello"},
            {"text": "world"},
        ])
        os.environ["DATA_JUICER_JSONL_LENIENT"] = "1"

        formatter = JsonFormatter(self.tmp_dir, text_keys=["text"])
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 2)

    def test_lenient_skips_bad_lines(self):
        import json
        path = os.path.join(self.tmp_dir, "data.jsonl")
        with open(path, "w") as f:
            f.write('{"text": "good1"}\n')
            f.write('bad json line\n')
            f.write('{"text": "good2"}\n')

        os.environ["DATA_JUICER_JSONL_LENIENT"] = "true"
        formatter = JsonFormatter(self.tmp_dir, text_keys=["text"])
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 2)
        texts = sorted(ds["text"])
        self.assertEqual(texts, ["good1", "good2"])

    def test_lenient_cfg_flag(self):
        from jsonargparse import Namespace
        self._write_jsonl("data.jsonl", [
            {"text": "test"},
        ])
        cfg = Namespace(load_jsonl_lenient=True)
        formatter = JsonFormatter(self.tmp_dir, text_keys=["text"])
        ds = formatter.load_dataset(global_cfg=cfg)
        self.assertEqual(len(ds), 1)

    def test_non_lenient_uses_default(self):
        self._write_jsonl("data.jsonl", [
            {"text": "normal"},
        ])
        formatter = JsonFormatter(self.tmp_dir, text_keys=["text"])
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["text"], "normal")

    def test_lenient_no_jsonl_files_falls_back(self):
        import json
        path = os.path.join(self.tmp_dir, "data.json")
        with open(path, "w") as f:
            json.dump([{"text": "from_json"}], f)

        os.environ["DATA_JUICER_JSONL_LENIENT"] = "1"
        formatter = JsonFormatter(self.tmp_dir, text_keys=["text"])
        ds = formatter.load_dataset()
        self.assertEqual(len(ds), 1)


if __name__ == "__main__":
    unittest.main()
