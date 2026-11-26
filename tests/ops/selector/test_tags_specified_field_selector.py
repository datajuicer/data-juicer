import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.selector.tags_specified_field_selector import (
    TagsSpecifiedFieldSelector,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TagsSpecifiedFieldSelectorTest(DataJuicerTestCaseBase):

    def _run_tag_selector(self, dataset: Dataset, target_list, op):
        dataset = op.process(dataset)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_tag_select(self):
        ds_list = [
            {
                "text": "a",
                "meta": {
                    "sentiment": "happy",
                },
            },
            {
                "text": "b",
                "meta": {
                    "sentiment": "happy",
                },
            },
            {
                "text": "c",
                "meta": {
                    "sentiment": "sad",
                },
            },
            {
                "text": "d",
                "meta": {
                    "sentiment": "angry",
                },
            },
        ]
        tgt_list = [
            {
                "text": "a",
                "meta": {
                    "sentiment": "happy",
                },
            },
            {
                "text": "b",
                "meta": {
                    "sentiment": "happy",
                },
            },
            {
                "text": "c",
                "meta": {
                    "sentiment": "sad",
                },
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = TagsSpecifiedFieldSelector(field_key="meta.sentiment", target_tags=["happy", "sad"])
        self._run_tag_selector(dataset, tgt_list, op)

    def test_list_tags_match_any_and_preserve_order(self):
        rows = [
            {"id": 0, "tags": ["math", "code"]},
            {"id": 1, "tags": ["history"]},
            {"id": 2, "tags": []},
            {"id": 3, "tags": ["code", "code"]},
            {"id": 4, "tags": ["Math"]},
            {"id": 5, "tags": None},
        ]
        dataset = Dataset.from_list(rows)
        op = TagsSpecifiedFieldSelector(field_key="tags", target_tags=["math", "code"])
        self.assertEqual(op.process(dataset).to_list(), [rows[0], rows[3]])
        # Indexed datasets must follow their logical row order.
        self.assertEqual(op.process(dataset.select([3, 1, 0])).to_list(), [rows[3], rows[0]])

    def test_nested_list_field_and_empty_targets(self):
        rows = [{"meta": {"topics": ["math"]}}, {"meta": {"topics": ["code"]}}]
        dataset = Dataset.from_list(rows)
        op = TagsSpecifiedFieldSelector(field_key="meta.topics", target_tags=["code"])
        self.assertEqual(op.process(dataset).to_list(), [rows[1]])
        op = TagsSpecifiedFieldSelector(field_key="meta.topics", target_tags=[])
        self.assertEqual(op.process(dataset).to_list(), [])

    def test_list_numbers_and_null_elements(self):
        rows = [{"tags": [1, 2]}, {"tags": [3]}, {"tags": [None]}, {"tags": []}, {"tags": None}]
        dataset = Dataset.from_list(rows)
        for targets, indices in [([2], [0]), ([None], [2, 4]), ([99], [])]:
            with self.subTest(targets=targets):
                op = TagsSpecifiedFieldSelector(field_key="tags", target_tags=targets)
                self.assertEqual(op.process(dataset).to_list(), [rows[i] for i in indices])

    def test_scalar_numbers_and_null_are_unchanged(self):
        rows = [{"tag": 1}, {"tag": None}, {"tag": 2}]
        op = TagsSpecifiedFieldSelector(field_key="tag", target_tags=[1, None])
        self.assertEqual(op.process(Dataset.from_list(rows)).to_list(), rows[:2])

    def test_reject_nested_containers(self):
        for tags in ([["math"]], [{"topic": "math"}]):
            with self.subTest(tags=tags):
                dataset = Dataset.from_list([{"tags": tags}, {"tags": tags}])
                op = TagsSpecifiedFieldSelector(field_key="tags", target_tags=["math"])
                with self.assertRaisesRegex(AssertionError, "The 0 item"):
                    op.process(dataset)

    def test_existing_single_row_and_empty_field_noops(self):
        dataset = Dataset.from_list([{"tags": ["math"]}])
        op = TagsSpecifiedFieldSelector(field_key="tags", target_tags=["code"])
        self.assertEqual(op.process(dataset).to_list(), dataset.to_list())
        dataset = Dataset.from_list([{"tags": ["math"]}, {"tags": ["code"]}])
        op = TagsSpecifiedFieldSelector(target_tags=["code"])
        self.assertEqual(op.process(dataset).to_list(), dataset.to_list())


if __name__ == "__main__":
    unittest.main()
