import gzip
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import find_files_with_suffix
from data_juicer.utils.jsonl_lenient_loader import (
    dataset_from_lenient_jsonl_files,
    iter_lenient_jsonl_records,
)


class JsonlLenientLoaderEncodingTest(unittest.TestCase):
    def test_utf8_with_and_without_bom(self):
        for suffix in (".jsonl", ".jsonl.gz", ".jsonl.zst"):
            for encoding in ("utf-8", "utf-8-sig"):
                with self.subTest(suffix=suffix, encoding=encoding), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / ("records" + suffix)
                    payload = '{"text": "first"}\nnot json\n{"text": "second"}\n'.encode(encoding)
                    if suffix == ".jsonl.gz":
                        payload = gzip.compress(payload)
                    elif suffix == ".jsonl.zst":
                        payload = zstd.ZstdCompressor().compress(payload)
                    path.write_bytes(payload)

                    rows = list(iter_lenient_jsonl_records([(str(path), suffix)], add_suffix_column=False))

                    self.assertEqual(rows, [{"text": "first"}, {"text": "second"}])

    def test_dataset_preserves_first_record_with_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"text": "first"}\n{"text": "second"}\n', encoding="utf-8-sig")

            dataset = dataset_from_lenient_jsonl_files([(str(path), ".jsonl")], add_suffix_column=False)

            self.assertEqual(dataset.to_list(), [{"text": "first"}, {"text": "second"}])

    def test_compressed_suffix_case_matches_discovery(self):
        formats = (
            (".JSONL.GZ", ".jsonl.gz", gzip.compress),
            (".Jsonl.Gz", ".jsonl.gz", gzip.compress),
            (".JSONL.ZST", ".jsonl.zst", zstd.ZstdCompressor().compress),
            (".JsonL.ZsT", ".jsonl.zst", zstd.ZstdCompressor().compress),
        )
        for suffix, extension, compress in formats:
            for encoding in ("utf-8", "utf-8-sig"):
                for add_suffix in (False, True):
                    with self.subTest(suffix=suffix, encoding=encoding, add_suffix=add_suffix):
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / ("MixedCase" + suffix)
                            raw = '{"text": "第一条"}\nnot json\n{"text": "second"}\n'.encode(encoding)
                            path.write_bytes(compress(raw))
                            discovered = find_files_with_suffix(path, [".jsonl.gz", ".jsonl.zst"])
                            self.assertEqual(discovered, {extension: [str(path)]})
                            pairs = [(name, ext) for ext, names in discovered.items() for name in names]

                            rows = list(iter_lenient_jsonl_records(pairs, add_suffix_column=add_suffix))

                            expected = [{"text": "第一条"}, {"text": "second"}]
                            if add_suffix:
                                for row in expected:
                                    row[Fields.suffix] = extension
                            self.assertEqual(rows, expected)

    def test_dataset_loads_discovered_uppercase_compressed_files(self):
        formats = (
            (".JSONL.GZ", gzip.compress),
            (".JSONL.ZST", zstd.ZstdCompressor().compress),
        )
        for suffix, compress in formats:
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ("MixedCase" + suffix)
                path.write_bytes(compress('{"text": "first"}\nnot json\n{"text": "second"}\n'.encode("utf-8-sig")))
                discovered = find_files_with_suffix(path, [".jsonl.gz", ".jsonl.zst"])
                pairs = [(name, ext) for ext, names in discovered.items() for name in names]

                dataset = dataset_from_lenient_jsonl_files(pairs, add_suffix_column=False)

                self.assertEqual(dataset.to_list(), [{"text": "first"}, {"text": "second"}])

    def test_discovered_uppercase_plain_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MixedCase.JSONL"
            path.write_text('{"text": "first"}\n', encoding="utf-8-sig")
            discovered = find_files_with_suffix(path, ".jsonl")
            self.assertEqual(discovered, {".jsonl": [str(path)]})
            pairs = [(name, ext) for ext, names in discovered.items() for name in names]

            self.assertEqual(list(iter_lenient_jsonl_records(pairs, add_suffix_column=False)), [{"text": "first"}])


if __name__ == "__main__":
    unittest.main()
