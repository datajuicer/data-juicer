import unittest
import json
from data_juicer.ops.mapper.text_tagging_by_prompt_mapper import TextTaggingByPromptMapper, DEFAULT_CLASSIFICATION_PROMPT, DEFAULT_CLASSIFICATION_LIST
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

def check_string_in_list(string_list, output):
    if not string_list: 
        assert False, "输入的列表不能是空的"
    
    for string in string_list:
        if string in output:
            return
        
    assert False, f"没有字符串在输出中"

# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class TextTaggingByPromptTest(DataJuicerTestCaseBase):
    text_key = 'text'

    def _run_tagging(self, samples, enable_vllm=False, sampling_params={}, **kwargs):
        op = TextTaggingByPromptMapper(
            hf_model='Qwen/Qwen2-7B-Instruct',
            prompt=DEFAULT_CLASSIFICATION_PROMPT,
            enable_vllm=enable_vllm,
            sampling_params=sampling_params,
            **kwargs
            )
        for sample in samples:
            result = op.process(sample)
            out_tag = result[Fields.text_tags]
            print(f'Output tag: {out_tag}')

            # test one output qa sample
            check_string_in_list(DEFAULT_CLASSIFICATION_LIST, out_tag)

    def test_tagging(self):
        samples = [
            {
            self.text_key: """{\n"instruction": "找出方程 x2 - 3x = 0 的根。",\n"input": "",\n"output": "该方程可以写成 x(x-3)=0。\n\n根据乘法原理，x = 0或x - 3 = 0。\n\n因此，x1 = 0和x2 = 3是方程 x2 - 3x = 0 的两个根。"\n}"""
            }]
        self._run_tagging(samples)

    def test_tagging_vllm(self):
        samples = [
            {
            self.text_key: """{\n"instruction": "找出方程 x2 - 3x = 0 的根。",\n"input": "",\n"output": "该方程可以写成 x(x-3)=0。\n\n根据乘法原理，x = 0或x - 3 = 0。\n\n因此，x1 = 0和x2 = 3是方程 x2 - 3x = 0 的两个根。"\n}"""
            }]
        self._run_tagging(
            samples, 
            enable_vllm=True,
            max_model_len=1024,
            max_num_seqs=16,
            sampling_params={'temperature': 0.1, 'top_p': 0.95, 'max_tokens': 256})


if __name__ == '__main__':
    unittest.main()
