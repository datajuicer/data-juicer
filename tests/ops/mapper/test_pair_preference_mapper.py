import unittest

from loguru import logger

from data_juicer.ops.mapper.pair_preference_mapper import PairPreferenceMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class PairPreferenceMapperTest(DataJuicerTestCaseBase):

    def _run_op(self, op, samples):
        for sample in samples:
            result = op.process(sample)
            logger.info(f'Output results: {result}')
            self.assertNotEqual(result['rejected_response'], '')
            self.assertNotEqual(result['reason'], '')

    def test(self):
        # before running this test, set below environment variables:
        # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
        # export OPENAI_API_KEY=your_key

        reference = '王八十娘：小远城王八十的娘亲，李莲花刚到小远城时被方多病偷掉钱袋找小乞丐问路时，刚好发现王八十娘被另一个小乞丐撞到便将她扶起，结识了王八十。\n朴二黄：灵山派管家，方多病小厮旺福的父亲。真实身份是金鸳盟的奔雷手辛雷，离开金鸳盟后，用假名朴二黄在灵山派当管家。因害怕王青山看穿他的身份，设计杀死了灵山派的王青山。被捕后识破了李莲花的真实身份，最后在攻击李莲花的时候被方多病情急之下杀死。'  # noqa: E501
        samples = [{
            'text': reference,
            'query': '李莲花，你认识方多病吗?',
            'response': '方多病啊，那可是我的好友。'
        }]
        op = PairPreferenceMapper(api_model='qwen2.5-72b-instruct')
        self._run_op(op, samples)

    def test_no_reference(self):
        samples = [{'query': '李莲花，你认识方多病吗?', 'response': '方多病啊，那可是我的好友。'}]
        system_prompt = ('修改问答对中的回答，在语言风格、事实性、人物身份、立场等任一方面与原回答相反。'
                         '必须按照以下标记格式输出，不要输出其他多余内容。\n'
                         '【回答】\n'
                         '生成的新回答\n'
                         '【原因】\n'
                         '生成该回答的原因')
        input_template = ('以下是原始问答对：\n'
                          '【问题】\n'
                          '{query}\n'
                          '【回答】\n'
                          '{response}')

        op = PairPreferenceMapper(api_model='qwen2.5-72b-instruct',
                                  system_prompt=system_prompt,
                                  input_template=input_template)
        self._run_op(op, samples)


if __name__ == '__main__':
    unittest.main()
