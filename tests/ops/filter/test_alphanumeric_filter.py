import unittest

from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class AlphanumericFilterTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone", "ray")
    def test_case(self):

        ds_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'a v s e e f g a qkc'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        tgt_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'a v s e e f g a qkc'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        dataset = self.generate_dataset(ds_list)
        op = AlphanumericFilter(min_ratio=0.2, max_ratio=0.9, batch_size=3, num_proc=1)
        result = self.run_single_op(dataset, op, ["text"])
        self.assertDatasetEqual(result, tgt_list)

    @TEST_TAG("standalone", "ray")
    def test_token_case(self):

        ds_list = [{
            'text': 'a=1\nb\nc=1+2+3+5\nd=6'
        }, {
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'a v s e e f g a qkc'
        }, {
            'text': '，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►'
        }, {
            'text': 'Do you need a cup of coffee?'
        }, {
            'text': 'emoji表情测试下😊，😸31231\n'
        }]
        tgt_list = [{
            'text':
            "Today is Sund Sund Sund Sunda and it's a happy day!\nYou know"
        }, {
            'text': 'Do you need a cup of coffee?'
        }]
        dataset = self.generate_dataset(ds_list)
        op = AlphanumericFilter(tokenization=True, min_ratio=1.5, batch_size=2, num_proc=1)
        result = self.run_single_op(dataset, op, ["text"])
        self.assertDatasetEqual(result, tgt_list)


if __name__ == '__main__':
    unittest.main()
