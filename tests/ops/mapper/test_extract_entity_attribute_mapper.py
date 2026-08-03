import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.extract_entity_attribute_mapper import ExtractEntityAttributeMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork
from data_juicer.utils.constant import DEFAULT_API_MODEL, Fields, MetaKeys

@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class ExtractEntityAttributeMapperTest(DataJuicerTestCaseBase):


    def _run_op(self, api_model, response_path=None, sampling_params=None):

        query_entities = ["李莲花", "方多病"]
        query_attributes = ["语言风格", "角色性格"]
        
        op = ExtractEntityAttributeMapper(
            api_model=api_model, 
            query_entities=query_entities,
            query_attributes=query_attributes,                 
            response_path=response_path,
            sampling_params=sampling_params)

        raw_text = """△笛飞声独自坐在莲花楼屋顶上。李莲花边走边悠闲地给马喂草。方多病则走在一侧，却总不时带着怀疑地盯向楼顶的笛飞声。
方多病走到李莲花身侧：我昨日分明看到阿飞神神秘秘地见了一人，我肯定他有什么瞒着我们。阿飞的来历我必须去查清楚！
李莲花继续悠然地喂草：放心吧，我认识他十几年了，对他一清二楚。
方多病：认识十几年？你上次才说是一面之缘？
李莲花忙圆谎：见得不多，但知根知底。哎，这老马吃得也太多了。
方多病一把夺过李莲花手中的草料：别转移话题！——快说！
李莲花：阿飞啊，脾气不太好，他......这十年也没出过几次门，所以见识短，你不要和他计较。还有他是个武痴，武功深藏不露，你平时别惹他。
方多病：呵，阿飞武功高？编瞎话能不能用心点？
李莲花：可都是大实话啊。反正，我和他彼此了解得很。你就别瞎操心了。
方多病很是质疑：(突然反应过来)等等！你说你和他认识十几年？你们彼此了解？！这么说，就我什么都不知道？！
△李莲花一愣，意外方多病是如此反应。
方多病很是不爽：不行，你们现在投奔我，我必须对我的手下都了解清楚。现在换我来问你，你，李莲花究竟籍贯何处？今年多大？家里还有什么人？平时都有些什么喜好？还有，可曾婚配？
△此时的笛飞声正坐在屋顶，从他的位置远远地向李莲花和方多病二人看去，二人声音渐弱。
李莲花：鄙人李莲花，有个兄弟叫李莲蓬，莲花山莲花镇莲花村人，曾经订过亲，但媳妇跟人跑子。这一辈子呢，没什么抱负理想，只想种种萝卜、逗逗狗，平时豆花爱吃甜的，粽子要肉的......
方多病：没一句实话。
"""
        samples = [{
            'text': raw_text,
        }]

        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)
        for sample in dataset:
            self.assertIn(MetaKeys.main_entities, sample[Fields.meta])
            self.assertIn(MetaKeys.attributes, sample[Fields.meta])
            self.assertIn(MetaKeys.attribute_descriptions, sample[Fields.meta])
            self.assertIn(MetaKeys.attribute_support_texts, sample[Fields.meta])
            ents = sample[Fields.meta][MetaKeys.main_entities]
            attrs = sample[Fields.meta][MetaKeys.attributes]
            descs = sample[Fields.meta][MetaKeys.attribute_descriptions]
            sups = sample[Fields.meta][MetaKeys.attribute_support_texts]
            for ent, attr, desc, sup in zip(ents, attrs, descs, sups):
                logger.info(f'{ent} {attr}: {desc}')
                self.assertNotEqual(desc, '')
                self.assertNotEqual(len(sup), 0)

    def test(self):
        # before running this test, set below environment variables:
        # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
        # export OPENAI_API_KEY=your_dashscope_key
        self._run_op(DEFAULT_API_MODEL, sampling_params={'enable_thinking': False})


class ParseOutputTest(DataJuicerTestCaseBase):
    """Tests for parse_output that do not require API access."""

    def _create_op(self):
        return ExtractEntityAttributeMapper(
            api_model='fake',
            query_entities=['李莲花'],
            query_attributes=['语言风格'],
        )

    def test_parse_output_with_demos(self):
        op = self._create_op()
        raw_string = (
            '# 李莲花\n'
            '## 语言风格：\n'
            '李莲花说话幽默风趣，常以轻松口吻化解尴尬局面。\n'
            '### 代表性示例摘录1：\n'
            '```\n'
            '放心吧，我认识他十几年了，对他一清二楚。\n'
            '```\n'
            '### 代表性示例摘录2：\n'
            '```\n'
            '哎，这老马吃得也太多了。\n'
            '```\n'
        )
        attribute, demos = op.parse_output(raw_string, '语言风格')
        self.assertIn('幽默风趣', attribute)
        self.assertEqual(len(demos), 2)
        self.assertIn('认识他十几年', demos[0])
        self.assertIn('老马吃得也太多了', demos[1])

    def test_parse_output_no_match(self):
        op = self._create_op()
        raw_string = 'This string has no matching patterns at all.'
        attribute, demos = op.parse_output(raw_string, '语言风格')
        self.assertEqual(attribute, '')
        self.assertEqual(len(demos), 0)

    def test_parse_output_empty(self):
        op = self._create_op()
        attribute, demos = op.parse_output('', '语言风格')
        self.assertEqual(attribute, '')
        self.assertEqual(len(demos), 0)


class EdgeCaseTest(DataJuicerTestCaseBase):
    """Tests for edge cases that do not require API access."""

    def test_already_generated_skip(self):
        op = ExtractEntityAttributeMapper(
            api_model='fake',
            query_entities=['李莲花'],
            query_attributes=['语言风格'],
        )
        sample = {
            'text': '一些文本',
            Fields.meta: {
                MetaKeys.main_entities: ['李莲花'],
                MetaKeys.attributes: ['语言风格'],
                MetaKeys.attribute_descriptions: ['幽默风趣'],
                MetaKeys.attribute_support_texts: [['示例']],
            },
        }
        result = op.process_single(sample)
        # Should return the sample unchanged (early return)
        self.assertEqual(
            result[Fields.meta][MetaKeys.main_entities], ['李莲花'])
        self.assertEqual(
            result[Fields.meta][MetaKeys.attributes], ['语言风格'])
        self.assertEqual(
            result[Fields.meta][MetaKeys.attribute_descriptions], ['幽默风趣'])
        self.assertEqual(
            result[Fields.meta][MetaKeys.attribute_support_texts], [['示例']])

    def test_empty_query_entities(self):
        op = ExtractEntityAttributeMapper(
            api_model='fake',
            query_entities=[],
            query_attributes=['语言风格'],
        )
        sample = {
            'text': '一些文本',
            Fields.meta: {},
        }
        result = op.process_single(sample)
        # With no entities, the nested loop produces empty lists
        self.assertEqual(result[Fields.meta][MetaKeys.main_entities], [])
        self.assertEqual(result[Fields.meta][MetaKeys.attributes], [])
        self.assertEqual(
            result[Fields.meta][MetaKeys.attribute_descriptions], [])
        self.assertEqual(
            result[Fields.meta][MetaKeys.attribute_support_texts], [])


@skip_if_from_fork(
    "Skipping API-based test because running from a fork repo")
class ExtractEntityAttributeMapperAPITest(DataJuicerTestCaseBase):
    """Additional API-based tests for drop_text and custom keys."""

    def test_drop_text(self):
        op = ExtractEntityAttributeMapper(
            api_model=DEFAULT_API_MODEL,
            query_entities=['李莲花'],
            query_attributes=['语言风格'],
            drop_text=True,
            sampling_params={'enable_thinking': False},
        )
        raw_text = (
            '李莲花：放心吧，我认识他十几年了，对他一清二楚。\n'
            '方多病：认识十几年？你上次才说是一面之缘？\n'
        )
        sample = {'text': raw_text, Fields.meta: {}}
        result = op.process_single(sample)
        self.assertNotIn('text', result)
        self.assertIn(MetaKeys.main_entities, result[Fields.meta])

    def test_custom_keys(self):
        op = ExtractEntityAttributeMapper(
            api_model=DEFAULT_API_MODEL,
            query_entities=['李莲花'],
            query_attributes=['语言风格'],
            entity_key='my_entity',
            attribute_key='my_attr',
            attribute_desc_key='my_desc',
            support_text_key='my_support',
            sampling_params={'enable_thinking': False},
        )
        raw_text = (
            '李莲花：放心吧，我认识他十几年了，对他一清二楚。\n'
            '方多病：认识十几年？你上次才说是一面之缘？\n'
        )
        samples = [{'text': raw_text}]
        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)
        result = dataset[0]
        self.assertIn('my_entity', result[Fields.meta])
        self.assertIn('my_attr', result[Fields.meta])
        self.assertIn('my_desc', result[Fields.meta])
        self.assertIn('my_support', result[Fields.meta])


if __name__ == '__main__':
    unittest.main()
