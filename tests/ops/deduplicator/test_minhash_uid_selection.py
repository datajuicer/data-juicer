import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator.document_minhash_deduplicator import DocumentMinhashDeduplicatorWithUid
from data_juicer.utils.constant import HashKeys


class MinhashUidSelectionTest(unittest.TestCase):

    def setUp(self):
        # The first three rows form a transitive component through two bands;
        # the minimum UID is neither the first row nor the smallest row index.
        self.rows = [
            {"text": "first", HashKeys.uid: 900, HashKeys.minhash: [b"a", b"x"]},
            {"text": "minimum", HashKeys.uid: -40, HashKeys.minhash: [b"a", b"y"]},
            {"text": "bridge", HashKeys.uid: 70, HashKeys.minhash: [b"b", b"y"]},
            {"text": "second component", HashKeys.uid: 10, HashKeys.minhash: [b"c", b"z"]},
            {"text": "second minimum", HashKeys.uid: -5, HashKeys.minhash: [b"d", b"z"]},
            {"text": "singleton", HashKeys.uid: 200, HashKeys.minhash: [b"e", b"w"]},
        ]
        self.op = DocumentMinhashDeduplicatorWithUid(num_permutations=2, num_bands=2, num_rows_per_band=1)

    def clean(self, index):
        return {k: v for k, v in self.rows[index].items() if k != HashKeys.minhash}

    def test_components_and_trace(self):
        for show_num in (0, 1, 2, 10):
            with self.subTest(show_num=show_num):
                result, pairs = self.op.process(Dataset.from_list(self.rows), show_num=show_num)
                self.assertEqual(result.to_list(), [self.clean(i) for i in (1, 4, 5)])
                expected = {}
                if show_num > 0:
                    expected[1] = [self.clean(1), self.clean(0)]
                if show_num > 1:
                    expected[4] = [self.clean(4), self.clean(3)]
                self.assertEqual(pairs, expected)

    def test_selected_dataset_preserves_logical_order(self):
        dataset = Dataset.from_list(self.rows).select([5, 2, 4, 1, 0, 3])
        result, pairs = self.op.process(dataset, show_num=2)
        self.assertEqual(result.to_list(), [self.clean(i) for i in (5, 4, 1)])
        self.assertEqual(
            pairs,
            {
                3: [self.clean(1), self.clean(2)],
                2: [self.clean(4), self.clean(3)],
            },
        )

    def test_unique_rows_preserve_extra_columns(self):
        rows = [dict(self.rows[i], payload={"raw": b"bytes", "values": [1, 2]}) for i in (0, 3, 5)]
        result, pairs = self.op.process(Dataset.from_list(rows))
        self.assertEqual(result.to_list(), [{k: v for k, v in row.items() if k != HashKeys.minhash} for row in rows])
        self.assertEqual(pairs, {})

    def test_selection_across_uid_batches(self):
        count = 10003
        dataset = Dataset.from_dict(
            {
                "text": [str(i) for i in range(count)],
                HashKeys.uid: [count - i for i in range(count)],
                HashKeys.minhash: [[bytes([i % 7])] * 2 for i in range(count)],
            }
        )
        result, pairs = self.op.process(dataset)
        self.assertEqual(result["text"][:], [str(i) for i in range(count - 7, count)])
        self.assertEqual(result[HashKeys.uid][:], list(range(7, 0, -1)))
        self.assertEqual(pairs, {})


if __name__ == "__main__":
    unittest.main()
