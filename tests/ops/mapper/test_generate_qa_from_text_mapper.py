import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper import generate_qa_from_text_mapper as qa_module
from data_juicer.ops.mapper.generate_qa_from_text_mapper import \
    GenerateQAFromTextMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

# @unittest.skip('unknown vllm connection error')
class GenerateQAFromTextMapperTest(DataJuicerTestCaseBase):
    text_key = 'text'

    def _run_op(self,
                enable_vllm=False,
                model_params=None,
                sampling_params=None,
                num_proc=1,
                max_num=None):

        op = GenerateQAFromTextMapper(enable_vllm=enable_vllm,
                                      model_params=model_params,
                                      sampling_params=sampling_params,
                                      max_num=max_num)

        samples = [{
            self.text_key:
            '蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n'
        }, {
            self.text_key:
            '四大名著是指《水浒传》《三国演义》《西游记》《红楼梦》四部长篇小说，作者分别是施耐庵、罗贯中、吴承恩、曹雪芹。'
        }]

        dataset = Dataset.from_list(samples)
        results = dataset.map(op.process, num_proc=num_proc, with_rank=True)

        if max_num is not None:
            self.assertLessEqual(len(results), len(samples)*max_num)

        for row in results:
            logger.info(row)
            self.assertIn(op.query_key, row)
            self.assertIn(op.response_key, row)

    def test(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(sampling_params=sampling_params)

    def test_max_num(self):
        sampling_params = {'max_new_tokens': 200}
        self._run_op(sampling_params=sampling_params, max_num=1)

    # def test_multi_process(self):
    #     sampling_params = {'max_new_tokens': 200}
    #     self._run_op(sampling_params=sampling_params, num_proc=2)

    # def test_vllm(self):
    #     model_params = {'max_model_len': 1024, 'max_num_seqs': 16}
    #     sampling_params = {
    #         'temperature': 0.9,
    #         'top_p': 0.95,
    #         'max_tokens': 200
    #     }
    #     self._run_op(enable_vllm=True,
    #                  model_params=model_params,
    #                  sampling_params=sampling_params)


class ParseOutputTest(DataJuicerTestCaseBase):
    """Test parse_output method without requiring model initialization."""

    def _create_op(self):
        """Create a GenerateQAFromTextMapper with a dummy model name.
        Since prepare_model for huggingface type only stores a partial
        (no download), this is safe for unit testing."""
        return GenerateQAFromTextMapper(hf_model='test-model')

    def test_parse_output_normal(self):
        op = self._create_op()
        raw_output = ('Human: question1 Assistant: answer1 '
                      'Human: question2 Assistant: answer2')
        result = op.parse_output(raw_output)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 'question1')
        self.assertEqual(result[0][1], 'answer1')
        self.assertEqual(result[1][0], 'question2')
        self.assertEqual(result[1][1], 'answer2')

    def test_parse_output_single_pair(self):
        op = self._create_op()
        raw_output = 'Human: single question Assistant: single answer'
        result = op.parse_output(raw_output)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 'single question')
        self.assertEqual(result[0][1], 'single answer')

    def test_parse_output_empty(self):
        op = self._create_op()
        raw_output = ''
        result = op.parse_output(raw_output)
        self.assertEqual(result, [])

    def test_parse_output_no_match(self):
        op = self._create_op()
        raw_output = 'This text has no QA pattern at all.'
        result = op.parse_output(raw_output)
        self.assertEqual(result, [])

    def test_parse_output_answer_can_contain_human(self):
        op = self._create_op()
        self.assertEqual(
            op.parse_output("Human: Who contributes? Assistant: Human reviewers."),
            [("Who contributes?", "Human reviewers.")],
        )


class GenerateQAFromTextBackendsTest(DataJuicerTestCaseBase):
    def test_api_backend_uses_shared_client_and_preserves_row_alignment(self):
        client = Mock(
            side_effect=[
                "Human: First? Assistant: One. Human: Second? Assistant: Two. Human: Third? Assistant: Three.",
                "",
            ]
        )
        with patch.object(qa_module, "get_model", return_value=client) as get_model, patch.object(
            qa_module, "update_sampling_params"
        ) as local_sampling:
            op = GenerateQAFromTextMapper(
                api_model="test-chat",
                api_endpoint="/chat/custom",
                response_path="data.text",
                sampling_params={"max_tokens": 100},
                max_num=2,
                text_key="body",
                query_key="question",
                response_key="answer",
            )
            result = op.process(
                {
                    "body": ["source 1", "source 2"],
                    "id": [1, 2],
                    "question": ["old q1", "old q2"],
                    "answer": ["old a1", "old a2"],
                }
            )

        self.assertEqual(
            result,
            {
                "body": ["source 1", "source 1"],
                "id": [1, 1],
                "question": ["First?", "Second?"],
                "answer": ["One.", "Two."],
            },
        )
        self.assertEqual(op.accelerator, "cpu")
        self.assertEqual(op.model_key.func.__name__, "prepare_api_model")
        self.assertEqual(
            op.model_key.keywords,
            {
                "model": "test-chat",
                "endpoint": "/chat/custom",
                "response_path": "data.text",
            },
        )
        get_model.assert_called_once_with(op.model_key, None, False)
        local_sampling.assert_not_called()
        self.assertEqual(client.call_count, 2)
        client.assert_any_call(
            [
                {"role": "system", "content": op.DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "source 1"},
            ],
            max_tokens=100,
        )

    def test_local_backend_system_prompt_is_opt_in(self):
        model = Mock(return_value=[{"generated_text": "Human: What? Assistant: This."}])
        op = GenerateQAFromTextMapper(hf_model="test-model", sampling_params={"max_new_tokens": 100})
        with patch.object(qa_module, "get_model", return_value=(model, None)):
            self.assertEqual(op.process({"text": ["source"]})["query"], ["What?"])
        model.assert_called_once_with(
            [{"role": "user", "content": "source"}],
            return_full_text=False,
            max_new_tokens=100,
        )

    def test_local_vllm_backend_uses_custom_prompt_and_pattern(self):
        output = SimpleNamespace(outputs=[SimpleNamespace(text="Q: What? A: This.")])
        model = Mock()
        model.chat.return_value = [output]
        with patch.object(qa_module, "vllm", SimpleNamespace(SamplingParams=lambda **params: params)), patch.object(
            qa_module, "get_model", return_value=(model, None)
        ), patch.object(qa_module, "update_sampling_params", return_value={"max_tokens": 100}):
            op = GenerateQAFromTextMapper(
                hf_model="test-model",
                enable_vllm=True,
                system_prompt="Reply with Q: and A:.",
                output_pattern=r"Q:(.*?)A:(.*)",
            )
            self.assertEqual(op.process({"text": ["source"]})["response"], ["This."])
        model.chat.assert_called_once_with(
            [
                {"role": "system", "content": "Reply with Q: and A:."},
                {"role": "user", "content": "source"},
            ],
            {"max_tokens": 100},
        )

    def test_api_and_vllm_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot be used together"):
            GenerateQAFromTextMapper(api_model="test-chat", enable_vllm=True)

    def test_api_options_require_api_model(self):
        with self.assertRaisesRegex(ValueError, "require api_model"):
            GenerateQAFromTextMapper(api_endpoint="/chat/custom")


if __name__ == '__main__':
    unittest.main()
