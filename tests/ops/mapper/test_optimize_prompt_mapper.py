import os
import math
import unittest

from datasets import load_dataset
from data_juicer.core.data import NestedDataset
from data_juicer.ops.mapper.optimize_prompt_mapper import OptimizePromptMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
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
        sampling_params = {'max_new_tokens': 200}
        self._run_op(model="qwen2.5-72b-instruct", is_hf_model=False, sampling_params=sampling_params)


if __name__ == '__main__':
    unittest.main()
