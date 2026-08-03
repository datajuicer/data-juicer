import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.optimize_qa_mapper import OptimizeQAMapper
from data_juicer.utils.constant import DEFAULT_API_MODEL
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork

# @unittest.skip('unknown vllm connection error')
@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class OptimizeQAMapperTest(DataJuicerTestCaseBase):

    def _run_op(self, model="Qwen/Qwen2.5-7B-Instruct", enable_vllm=False, is_hf_model=True, sampling_params=None, num_proc=1):

        op = OptimizeQAMapper(
            api_or_hf_model=model,
            enable_vllm=enable_vllm,
            is_hf_model=is_hf_model,
            sampling_params=sampling_params)

        samples = [{
            'query':
            '鱼香肉丝怎么做？',
            'response':
            '鱼香肉丝是将猪肉丝与胡萝卜、青椒、木耳炒制，调入调味料如酱油、醋和辣豆瓣酱，快速翻炒而成的美味佳肴。'
        }, {
            'query': '什么是蚂蚁上树？',
            'response': '蚂蚁上树是一道中国菜。'
        }]
        dataset = Dataset.from_list(samples)
        results = dataset.map(op.process, num_proc=num_proc, with_rank=True)

        for row in results:
            logger.info(f'Output results: {row}')
            self.assertNotEqual(row['query'], '')
            self.assertNotEqual(row['response'], '')

    def test(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(sampling_params=sampling_params)

    def test_api(self):
        sampling_params = {'max_new_tokens': 200, 'enable_thinking': False}
        self._run_op(model=DEFAULT_API_MODEL, is_hf_model=False, sampling_params=sampling_params)

    # def test_multi_process(self):
    #     sampling_params = {'max_new_tokens': 200}
    #     self._run_op(sampling_params=sampling_params, num_proc=2)

    # def test_vllm(self):
    #     sampling_params = {'max_tokens': 200}
    #     self._run_op(enable_vllm=True, sampling_params=sampling_params)


class ParseOutputTest(DataJuicerTestCaseBase):
    """Test parse_output method without requiring model initialization."""

    def _create_op(self):
        """Create an OptimizeQAMapper with API mode (no model download)."""
        return OptimizeQAMapper(
            api_or_hf_model='test-model',
            is_hf_model=False,
        )

    def test_parse_output_normal(self):
        op = self._create_op()
        raw_output = '【问题】优化后的问题【回答】优化后的回答'
        result = op.parse_output(raw_output)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], '优化后的问题')
        self.assertEqual(result[1], '优化后的回答')

    def test_parse_output_empty(self):
        op = self._create_op()
        raw_output = ''
        result = op.parse_output(raw_output)
        self.assertEqual(result, (None, None))


class EdgeCaseTest(DataJuicerTestCaseBase):
    """Test edge cases without requiring model or API access."""

    def test_empty_query(self):
        """Test behavior when query is empty string."""
        op = OptimizeQAMapper(
            api_or_hf_model='test-model',
            is_hf_model=False,
        )
        sample = {
            'query': '',
            'response': '这是一个回答',
        }
        # build_input should still work with empty query
        input_prompt = op.build_input(sample)
        self.assertIn('【问题】', input_prompt)
        self.assertIn('【回答】', input_prompt)


if __name__ == '__main__':
    unittest.main()
