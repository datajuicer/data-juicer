import random
import unittest
from unittest.mock import patch

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator.document_deduplicator import DocumentDeduplicator
from data_juicer.ops.deduplicator.image_deduplicator import ImageDeduplicator
from data_juicer.ops.deduplicator.video_deduplicator import VideoDeduplicator
from data_juicer.utils.constant import HashKeys


class ExactDedupTraceCountsTest(unittest.TestCase):
    @staticmethod
    def make_op(kind, consider_text):
        if kind == "document":
            return DocumentDeduplicator(), HashKeys.hash
        if kind == "image":
            # The tests exercise process() with precomputed hashes, not image
            # decoding or hashing, so no optional imagededup model is needed.
            with patch("data_juicer.ops.deduplicator.image_deduplicator.get_hash_method"):
                return ImageDeduplicator(consider_text=consider_text), HashKeys.imagehash
        return VideoDeduplicator(consider_text=consider_text), HashKeys.videohash

    @staticmethod
    def expected(rows, key, kind, consider_text, show_num):
        groups = {}
        kept = []
        for row in rows:
            value = (row[key], row[HashKeys.hash]) if consider_text else row[key]
            if kind != "document" and not value:
                kept.append(row)
                continue
            if value not in groups:
                groups[value] = []
                kept.append(row)
            groups[value].append(row)
        ranked = sorted(groups, key=lambda value: len(groups[value]), reverse=True)
        duplicates = [value for value in ranked if len(groups[value]) > 1][:show_num]
        return kept, {value: groups[value][:2] for value in duplicates}

    def check_hashes(self, values):
        for kind, consider_text in (
            ("document", False),
            ("image", False),
            ("image", True),
            ("video", False),
            ("video", True),
        ):
            op, key = self.make_op(kind, consider_text)
            rows = []
            for index, value in enumerate(values):
                row = {"text": f"row {index}", "payload": {"id": index}, key: value}
                if consider_text:
                    row[HashKeys.hash] = str(index % 2) if value else ""
                rows.append(row)
            dataset = Dataset.from_list(rows)
            for limit in (0, 1, 2, 3, 50):
                with self.subTest(kind=kind, consider_text=consider_text, limit=limit):
                    expected_rows, expected_pairs = self.expected(rows, key, kind, consider_text, limit)
                    result, pairs = op.process(dataset, show_num=limit)
                    self.assertEqual(result.to_list(), expected_rows)
                    self.assertEqual(pairs, expected_pairs)

    def test_equal_counts_missing_hashes_and_trace_limits(self):
        self.check_hashes(["b", "a", "b", "c", "a", "c", "", "", None, None, "solo"])

    def test_mixed_frequencies(self):
        values = [f"hash-{i}" for i in range(19) for _ in range(i % 5 + 1)]
        random.Random(42).shuffle(values)
        self.check_hashes(values)

    def test_no_duplicates(self):
        self.check_hashes(["a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
