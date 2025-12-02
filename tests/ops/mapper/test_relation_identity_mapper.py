import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.relation_identity_mapper import RelationIdentityMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK
from data_juicer.utils.constant import Fields, MetaKeys

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class RelationIdentityMapperTest(DataJuicerTestCaseBase):

    # before running this test, set below environment variables:
    # export OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    # export OPENAI_API_KEY=your_key

    def _run_op(self, api_model, output_key=MetaKeys.role_relation):

        op = RelationIdentityMapper(api_model=api_model,
                                    source_entity="李莲花",
                                    target_entity="方多病",
                                    output_key=output_key)

        raw_text = """李莲花原名李相夷，十五岁战胜西域天魔，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。
在与金鸳盟盟主笛飞声的对决中，李相夷中毒重伤，沉入大海，十年后在莲花楼醒来，过起了市井生活。他帮助肉铺掌柜解决家庭矛盾，表现出敏锐的洞察力。
李莲花与方多病合作，解决了灵山派掌门王青山的假死案，揭露了朴管家的罪行。
随后，他与方多病和笛飞声一起调查了玉秋霜的死亡案，最终揭露了玉红烛的阴谋。在朴锄山，李莲花和方多病调查了七具无头尸事件，发现男童的真实身份是笛飞声。
李莲花利用飞猿爪偷走男童手中的观音垂泪，导致笛飞声恢复内力，但李莲花巧妙逃脱。李莲花与方多病继续合作，调查了少师剑被盗案，揭露了静仁和尚的阴谋。
在采莲庄，他解决了新娘溺水案，找到了狮魂的线索，并在南门园圃挖出单孤刀的药棺。在玉楼春的案件中，李莲花和方多病揭露了玉楼春的阴谋，救出了被拐的清儿。
在石寿村，他们发现了柔肠玉酿的秘密，并救出了被控制的武林高手。李莲花与方多病在白水园设下机关，救出方多病的母亲何晓惠，并最终在云隐山找到了治疗碧茶之毒的方法。
在天机山庄，他揭露了单孤刀的野心，救出了被控制的大臣。在皇宫，李莲花与方多病揭露了魔僧和单孤刀的阴谋，成功解救了皇帝。
最终，李莲花在东海之滨与笛飞声的决斗中未出现，留下一封信，表示自己已无法赴约。
一年后，方多病在东海畔的柯厝村找到了李莲花，此时的李莲花双目失明，右手残废，但心态平和，过着简单的生活。
方多病 (称呼:方小宝、方大少爷)百川院刑探，单孤刀之子，李相夷的徒弟。方多病通过百川院的考核，成为刑探，并在百川院内展示了自己是李相夷的弟子，获得暂时的录用。
他接到任务前往嘉州调查金鸳盟的余孽，期间与李莲花相识并合作破案。方多病在调查过程中逐渐了解到自己的身世，发现自己的生父是单孤刀。
他与李莲花、笛飞声等人多次合作，共同对抗金鸳盟和单孤刀的阴谋。方多病在一系列案件中展现了出色的推理能力和武艺，逐渐成长为一名优秀的刑探。
最终，方多病在天机山庄和皇宫的斗争中发挥了关键作用，帮助李莲花等人挫败了单孤刀的野心。在李莲花中毒后，方多病决心为他寻找解毒之法，展现了深厚的友情。
"""
        samples = [{
            'text': raw_text,
        }]

        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)
        for data in dataset:
            for k in data:
                logger.info(f"{k}: {data[k]}")
            self.assertIn(output_key, data[Fields.meta])
            self.assertNotEqual(data[Fields.meta][output_key], '')

    def test_default(self):
        self._run_op('qwen2.5-72b-instruct')

    def test_rename_key(self):
        self._run_op('qwen2.5-72b-instruct', output_key='output')


if __name__ == '__main__':
    unittest.main()
