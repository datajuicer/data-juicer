import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.extract_nickname_mapper import ExtractNicknameMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork
from data_juicer.utils.constant import DEFAULT_API_MODEL, Fields, MetaKeys

@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class ExtractNicknameMapperTest(DataJuicerTestCaseBase):


    def _run_op(self, api_model, response_path=None, sampling_params=None):

        op = ExtractNicknameMapper(api_model=api_model,
                               response_path=response_path,
                               sampling_params=sampling_params)

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
        self._run_op(DEFAULT_API_MODEL, sampling_params={'enable_thinking': False})


class ParseOutputTest(DataJuicerTestCaseBase):
    """Tests for parse_output that do not require API access."""

    def _create_op(self):
        return ExtractNicknameMapper(api_model='fake')

    def test_parse_output_normal(self):
        op = self._create_op()
        raw_string = (
            '### 称呼方式1\n'
            '- **说话人**：李莲花\n'
            '- **被称呼人**：方多病\n'
            '- **李莲花对方多病的昵称**：方小宝\n'
            '### 称呼方式2\n'
            '- **说话人**：方多病\n'
            '- **被称呼人**：笛飞声\n'
            '- **方多病对笛飞声的昵称**：阿飞\n'
        )
        results = op.parse_output(raw_string)
        self.assertEqual(len(results), 2)
        sources = [r[MetaKeys.source_entity] for r in results]
        targets = [r[MetaKeys.target_entity] for r in results]
        descs = [r[MetaKeys.relation_description] for r in results]
        self.assertIn('李莲花', sources)
        self.assertIn('方多病', sources)
        self.assertIn('方多病', targets)
        self.assertIn('笛飞声', targets)
        self.assertIn('方小宝', descs)
        self.assertIn('阿飞', descs)
        for r in results:
            self.assertEqual(r[MetaKeys.relation_keywords], ['nickname'])
            self.assertIsNone(r[MetaKeys.relation_strength])

    def test_parse_output_double_check_mismatch(self):
        op = self._create_op()
        # role1 is '李莲花' but role1_tmp (in the nickname line) is '方多病'
        # This should be discarded by the double-check logic
        raw_string = (
            '### 称呼方式1\n'
            '- **说话人**：李莲花\n'
            '- **被称呼人**：方多病\n'
            '- **方多病对方多病的昵称**：方小宝\n'
        )
        results = op.parse_output(raw_string)
        self.assertEqual(len(results), 0)

    def test_parse_output_nickname_equals_entity(self):
        op = self._create_op()
        # nickname == role2, should be filtered out
        raw_string = (
            '### 称呼方式1\n'
            '- **说话人**：李莲花\n'
            '- **被称呼人**：方多病\n'
            '- **李莲花对方多病的昵称**：方多病\n'
        )
        results = op.parse_output(raw_string)
        self.assertEqual(len(results), 0)

    def test_parse_output_empty(self):
        op = self._create_op()
        results = op.parse_output('')
        self.assertEqual(len(results), 0)


class EdgeCaseTest(DataJuicerTestCaseBase):
    """Tests for edge cases that do not require API access."""

    def test_already_generated_skip(self):
        op = ExtractNicknameMapper(api_model='fake')
        existing_nicknames = [
            {
                MetaKeys.source_entity: '李莲花',
                MetaKeys.target_entity: '方多病',
                MetaKeys.relation_description: '方小宝',
                MetaKeys.relation_keywords: ['nickname'],
                MetaKeys.relation_strength: None,
            }
        ]
        sample = {
            'text': '一些文本',
            Fields.meta: {
                MetaKeys.nickname: existing_nicknames,
            },
        }
        result = op.process_single(sample)
        # Should return unchanged (early return)
        self.assertEqual(
            result[Fields.meta][MetaKeys.nickname], existing_nicknames)


@skip_if_from_fork(
    "Skipping API-based test because running from a fork repo")
class ExtractNicknameMapperAPITest(DataJuicerTestCaseBase):
    """Additional API-based tests for drop_text and custom key."""

    def test_drop_text(self):
        op = ExtractNicknameMapper(
            api_model=DEFAULT_API_MODEL,
            drop_text=True,
            sampling_params={'enable_thinking': False},
        )
        raw_text = (
            '李莲花：方小宝，你可听过江湖上有什么门派是驯兽来斗？\n'
            '方多病回过神：不、不曾听过。\n'
        )
        sample = {'text': raw_text, Fields.meta: {}}
        result = op.process_single(sample)
        self.assertNotIn('text', result)
        self.assertIn(MetaKeys.nickname, result[Fields.meta])

    def test_custom_key(self):
        op = ExtractNicknameMapper(
            api_model=DEFAULT_API_MODEL,
            nickname_key='my_nickname',
            sampling_params={'enable_thinking': False},
        )
        raw_text = (
            '李莲花：方小宝，你可听过江湖上有什么门派是驯兽来斗？\n'
            '方多病回过神：不、不曾听过。\n'
        )
        samples = [{'text': raw_text}]
        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)
        result = dataset[0]
        self.assertIn('my_nickname', result[Fields.meta])


if __name__ == '__main__':
    unittest.main()
