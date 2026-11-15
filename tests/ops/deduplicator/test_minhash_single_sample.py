import unittest

from datasets import Dataset

from data_juicer.ops.deduplicator.document_minhash_deduplicator import (
    DocumentMinhashDeduplicator,
    DocumentMinhashDeduplicatorWithUid,
)
from data_juicer.utils.constant import HashKeys


class MinhashSingleSampleTest(unittest.TestCase):

    def test_single_sample_drops_minhash_column(self):
        dataset = Dataset.from_dict(
            {
                "text": ["only one document"],
                HashKeys.minhash: [[b"x" * 64]],
            }
        )
        op = DocumentMinhashDeduplicator()
        result, _ = op.process(dataset)
        self.assertNotIn(HashKeys.minhash, result.column_names)

    def test_single_sample_with_uid_drops_minhash_column(self):
        dataset = Dataset.from_dict(
            {
                "text": ["only one document"],
                HashKeys.minhash: [[b"x" * 64]],
                HashKeys.uid: [1],
            }
        )
        op = DocumentMinhashDeduplicatorWithUid()
        result, _ = op.process(dataset)
        self.assertNotIn(HashKeys.minhash, result.column_names)


if __name__ == "__main__":
    unittest.main()
