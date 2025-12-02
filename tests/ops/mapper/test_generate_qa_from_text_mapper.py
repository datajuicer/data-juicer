import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
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


if __name__ == '__main__':
    unittest.main()
