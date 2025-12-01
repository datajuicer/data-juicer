import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.selector.tags_specified_field_selector import \
    TagsSpecifiedFieldSelector
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TagsSpecifiedFieldSelectorTest(DataJuicerTestCaseBase):

    def _run_tag_selector(self, dataset: Dataset, target_list, op):
        dataset = op.process(dataset)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_tag_select(self):
        ds_list = [{
            'text': 'a',
            'meta': {
                'sentiment': 'happy',
            }
        }, {
            'text': 'b',
            'meta': {
                'sentiment': 'happy',
            }
        }, {
            'text': 'c',
            'meta': {
                'sentiment': 'sad',
            }
        }, {
            'text': 'd',
            'meta': {
                'sentiment': 'angry',
            }
        }]
        tgt_list = [{
            'text': 'a',
            'meta': {
                'sentiment': 'happy',
            }
        }, {
            'text': 'b',
            'meta': {
                'sentiment': 'happy',
            }
        }, {
            'text': 'c',
            'meta': {
                'sentiment': 'sad',
            }
        }]
        dataset = Dataset.from_list(ds_list)
        op = TagsSpecifiedFieldSelector(
            field_key='meta.sentiment',
            target_tags=['happy', 'sad'])
        self._run_tag_selector(dataset, tgt_list, op)


if __name__ == '__main__':
    unittest.main()
