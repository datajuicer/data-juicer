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


    def test_compute_stats_batched_no_tokenization(self):
        """Directly call compute_stats_batched for non-tokenization path."""
        from data_juicer.utils.constant import Fields, StatsKeys

        op = AlphanumericFilter(tokenization=False, min_ratio=0.25, batch_size=2)
        samples = {
            'text': [
                'hello world 123',       # 13 alnum / 15 total = 0.8667
                '!@#$%^&*()',            # 0 alnum / 10 total = 0.0
                '',                      # empty -> 0.0
            ],
            Fields.stats: [{}, {}, {}],
        }
        result = op.compute_stats_batched(samples)
        self.assertAlmostEqual(
            result[Fields.stats][0][StatsKeys.alnum_ratio], 13 / 15, places=5)
        self.assertAlmostEqual(
            result[Fields.stats][1][StatsKeys.alnum_ratio], 0.0)
        self.assertAlmostEqual(
            result[Fields.stats][2][StatsKeys.alnum_ratio], 0.0)

    def test_process_batched_directly(self):
        """Directly call process_batched to verify filtering."""
        from data_juicer.utils.constant import Fields, StatsKeys

        op = AlphanumericFilter(tokenization=False, min_ratio=0.25, max_ratio=0.9, batch_size=2)
        samples = {
            Fields.stats: [
                {StatsKeys.alnum_ratio: 0.8},   # within range -> keep
                {StatsKeys.alnum_ratio: 0.1},   # below min -> filtered
                {StatsKeys.alnum_ratio: 0.95},  # above max -> filtered
            ],
        }
        keep_flags = list(op.process_batched(samples))
        self.assertEqual(keep_flags, [True, False, False])

    def test_compute_stats_batched_skips_existing(self):
        """Already computed stats are not recomputed."""
        from data_juicer.utils.constant import Fields, StatsKeys

        op = AlphanumericFilter(tokenization=False, min_ratio=0.25, batch_size=2)
        samples = {
            'text': ['hello'],
            Fields.stats: [{StatsKeys.alnum_ratio: 0.99}],  # pre-set
        }
        result = op.compute_stats_batched(samples)
        # Should preserve existing value, not recompute
        self.assertAlmostEqual(
            result[Fields.stats][0][StatsKeys.alnum_ratio], 0.99)


if __name__ == '__main__':
    unittest.main()
