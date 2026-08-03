import os
import math
import unittest

from datasets import load_dataset
from data_juicer.core.data import NestedDataset
from data_juicer.ops.mapper.optimize_prompt_mapper import OptimizePromptMapper
from data_juicer.utils.constant import DEFAULT_API_MODEL
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork

@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class OptimizePromptMapperTest(DataJuicerTestCaseBase):
    prompt_key = 'prompt'
    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..')
    test_data_path = os.path.join(root_path, 'demos/data/auto-prompt-optim/demo-dataset-prompts.jsonl')

    def _run_op(self, model="Qwen/Qwen2.5-7B-Instruct", enable_vllm=False, is_hf_model=True, sampling_params=None, num_proc=1):
        gen_num = 3
        batch_size = 2
        op = OptimizePromptMapper(
            api_or_hf_model=model,
            gen_num=gen_num,
            max_example_num=3,
            enable_vllm=enable_vllm,
            is_hf_model=is_hf_model,
            sampling_params=sampling_params,
        )

        dataset = NestedDataset(load_dataset("json", data_files=self.test_data_path, split='train'))

        results = dataset.map(op.process, num_proc=num_proc, with_rank=True, batched=True, batch_size=batch_size)

        num_batches = math.ceil(len(dataset) / batch_size)
        self.assertEqual(len(results), len(dataset) + num_batches * gen_num)

    def test(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(sampling_params=sampling_params)

    def test_api_model(self):
        sampling_params = {'max_new_tokens': 200, 'enable_thinking': False}
        self._run_op(model=DEFAULT_API_MODEL, is_hf_model=False, sampling_params=sampling_params)


class ParseOutputTest(DataJuicerTestCaseBase):
    """Test parse_output method without requiring model initialization."""

    def _create_op(self):
        """Create an OptimizePromptMapper with API mode (no model download)."""
        return OptimizePromptMapper(
            api_or_hf_model='test-model',
            is_hf_model=False,
        )

    def test_parse_output_normal(self):
        op = self._create_op()
        raw_output = '【提示词】这是一个优化后的提示词【分析】这是分析'
        result = op.parse_output(raw_output)
        self.assertEqual(result, '这是一个优化后的提示词')

    def test_parse_output_no_match(self):
        op = self._create_op()
        raw_output = '这里没有任何匹配的模式'
        result = op.parse_output(raw_output)
        self.assertEqual(result, '')

    def test_parse_output_empty(self):
        op = self._create_op()
        raw_output = ''
        result = op.parse_output(raw_output)
        self.assertEqual(result, '')


class EdgeCaseTest(DataJuicerTestCaseBase):
    """Test edge cases without requiring model or API access."""

    def test_missing_prompt_key(self):
        """Test that process_batched returns samples unchanged when
        prompt_key is missing from the samples dict."""
        op = OptimizePromptMapper(
            api_or_hf_model='test-model',
            is_hf_model=False,
        )
        # Samples without the 'prompt' key
        samples = {
            'text': ['sample1', 'sample2'],
        }
        result = op.process_batched(samples)
        self.assertEqual(result, samples)


@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class OptimizePromptMapperAPITest(DataJuicerTestCaseBase):
    """Additional API-based tests for OptimizePromptMapper."""

    root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', '..', '..')
    test_data_path = os.path.join(
        root_path,
        'demos/data/auto-prompt-optim/demo-dataset-prompts.jsonl')

    def test_keep_original_false(self):
        gen_num = 3
        batch_size = 2
        op = OptimizePromptMapper(
            api_or_hf_model=DEFAULT_API_MODEL,
            gen_num=gen_num,
            max_example_num=3,
            keep_original_sample=False,
            is_hf_model=False,
            sampling_params={'enable_thinking': False},
        )

        dataset = NestedDataset(load_dataset(
            'json', data_files=self.test_data_path, split='train'))

        results = dataset.map(op.process, num_proc=1,
                              with_rank=True, batched=True,
                              batch_size=batch_size)

        # With keep_original_sample=False, output count per batch should be
        # at most gen_num (no originals kept).
        num_batches = math.ceil(len(dataset) / batch_size)
        self.assertLessEqual(len(results), num_batches * gen_num)

    def test_gen_num_1(self):
        gen_num = 1
        batch_size = 2
        op = OptimizePromptMapper(
            api_or_hf_model=DEFAULT_API_MODEL,
            gen_num=gen_num,
            max_example_num=3,
            keep_original_sample=True,
            is_hf_model=False,
            sampling_params={'enable_thinking': False},
        )

        dataset = NestedDataset(load_dataset(
            'json', data_files=self.test_data_path, split='train'))

        results = dataset.map(op.process, num_proc=1,
                              with_rank=True, batched=True,
                              batch_size=batch_size)

        num_batches = math.ceil(len(dataset) / batch_size)
        self.assertEqual(len(results), len(dataset) + num_batches * gen_num)


if __name__ == '__main__':
    unittest.main()
