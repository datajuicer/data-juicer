import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.extract_nickname_mapper import ExtractNicknameMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK
from data_juicer.utils.constant import Fields, MetaKeys

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class ExtractNicknameMapperTest(DataJuicerTestCaseBase):


    def _run_op(self, api_model, response_path=None):

        op = ExtractNicknameMapper(api_model=api_model,
                               response_path=response_path)

        raw_text = """△李莲花又指出刚才门框上的痕迹。
△李莲花：门框上也是人的掌痕和爪印。指力能嵌入硬物寸余，七分力道主上，三分力道垫下，还有辅以的爪式，看样子这还有昆仑派的外家功夫。
方多病看着李莲花，愈发生疑os：通过痕迹就能判断出功夫和门派，这绝对只有精通武艺之人才能做到，李莲花你到底是什么人？！
笛飞声环顾四周：有朝月派，还有昆仑派，看来必是一群武林高手在这发生了决斗！
李莲花：如果是武林高手过招，为何又会出现如此多野兽的痕迹。方小宝，你可听过江湖上有什么门派是驯兽来斗？方小宝？方小宝？
方多病回过神：不、不曾听过。
李莲花：还有这些人都去了哪里？
笛飞声：打架不管是输是赢，自然是打完就走。
李莲花摇头：就算打完便走，但这里是客栈，为何这么多年一直荒在这里，甚至没人来收拾一下？
笛飞声：闹鬼？这里死过这么多人，楼下又画了那么多符，所以不敢进来？
△这时，梁上又出现有东西移动的声响，李莲花、笛飞声都猛然回头看去。
"""
        samples = [{
            'text': raw_text,
        }]

        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)
        self.assertIn(MetaKeys.nickname, dataset[0][Fields.meta])
        result = dataset[0][Fields.meta][MetaKeys.nickname]
        result = [(
            d[MetaKeys.source_entity],
            d[MetaKeys.target_entity],
            d[MetaKeys.relation_description])
            for d in result]
        logger.info(f'result: {result}')
        self.assertIn(("李莲花","方多病","方小宝"), result)

    def test(self):
        # before running this test, set below environment variables:
        # export OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
        # export OPENAI_API_KEY=your_key
        self._run_op('qwen2.5-72b-instruct')


if __name__ == '__main__':
    unittest.main()
