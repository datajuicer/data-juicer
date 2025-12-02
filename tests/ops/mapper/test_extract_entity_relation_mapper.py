import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.extract_entity_relation_mapper import ExtractEntityRelationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields, MetaKeys

class ExtractEntityRelationMapperTest(DataJuicerTestCaseBase):


    def _run_op(self, op):

        raw_text = """△芩婆走到中间，看着众人。
芩婆：当年，我那老鬼漆木山与李相夷之父乃是挚交。原本李家隐世而居，一日为了救人，得罪附近山匪，夜里便遭了山匪所袭，唯有二子生还，流落街头。
封磬震惊：二子？不是只有一个儿子吗？
芩婆：我和漆木山得知这个噩耗后，到处寻找李家那两个孩子的下落。只可惜等我们找他们时，李家长子李相显已经病死。
李莲花似回忆起了什么：李相显......
芩婆：我们只从乞丐堆里带回了年纪尚且未满四岁的李相夷，以及，(看向单孤刀)二个一直护着李相夷，与李相显年纪相仿的小乞丐......
闪回/
李相显将李且给他的玉佩塞给单孤刀，恳切托付：我没什么值钱的东西，这个玉佩是我唯一的家当了、送给你，我弟弟、相夷......求你照顾他一阵......
△李相显还想再说什么已气绝而亡，小相夷唤着哥哥大哭，单孤刀愕然看着手里的玉佩有点不知所措。
△话刚说完，哐当一声破庙门倒进来，几个其他少年乞丐进来。少年乞丐老大：这地儿不错，诶，你俩，出去！
△单孤刀把小相夷护在身后，抓住靠在墙边的木棍。单孤刀：这儿，是我，和我弟弟的。
乞丐们要抢李相夷的馒头，小李相夷哭着死死护住自馒头不放。
乞丐甲野蛮地抢：给我拿来！
小单孤刀：放开他！
△单孤刀用力撞向几个乞丐，救下小李相夷。乞丐甲：小子，活腻了！
△几个乞丐围攻小单孤刀，小单孤刀和众乞丐厮打到一起。突然其中一个乞丐掏出一把生锈的刀就朝单孤刀砍去、一个点燃火把棍戳他。单孤刀侧手一挡，火把棍在他手腕上烫出一道伤口，身后几根棍子打得他痛苦倒地！
/闪回结束
△单孤刀拿着自己手里的玉佩看着，又看看自己手上的印记，不肯相信。单孤刀：胡说！全都是胡说！这些事我为何不知道？都是你在信口雌黄！
芩婆：那我问你，我们将你带回云隐山之前的事你又记得多少？
△单孤刀突然愣住，他意识到那之前的事自己竟都想不起来。
芩婆：怎么？都想不起来了？(拽起单孤刀手腕，露出他的伤痕)你当日被你师父找到时，手腕上就受了伤，也正因为这处伤，高烧不退，醒来后便忘记了不少从前的事。
△单孤刀呆住。
芩婆：而相夷当年不过孩童，尚未到记事的年纪，很多事自然不知道。
△李莲花得知真相，闭目叹息。
△封磬震惊地看看单孤刀，又看看李莲花，终于想明白了一切，颓然、懊恼。
封磬：自萱公主之子下落不明后，这近百年来我们整个家族都一直在不遗余力地寻找萱公主的子嗣后代，直到二十几年前终于让我寻得了线索，知道萱公主的曾孙被漆木山夫妇收为徒，但......我只知道萱公主之孙有一年约十岁的儿子，却不知......原来竟还有一幼子！我......我凭着南胤皇族的玉佩、孩子的年纪和他身上的印记来与主上相认，可没想到......这竟是一个错误！全错了！
△封磬神情复杂地看向李莲花，封磬：你，你才是我的主上......
△封磬颓然地跪倒下来。
△李莲花对眼前的一切有些意外、无措。
笛飞声冷声：怪不得单孤刀的血对业火独毫无作用，李莲花的血才能毁掉这东西。
△笛飞声不禁冷笑一下。
"""
        samples = [{
            'text': raw_text,
        }]

        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)
        sample = dataset[0]
        self.assertIn(MetaKeys.entity, sample[Fields.meta])
        self.assertIn(MetaKeys.relation, sample[Fields.meta])
        self.assertNotEqual(len(sample[Fields.meta][MetaKeys.entity]), 0)
        self.assertNotEqual(len(sample[Fields.meta][MetaKeys.relation]), 0)
        logger.info(f"entitis: {sample[Fields.meta][MetaKeys.entity]}")
        logger.info(f"relations: {sample[Fields.meta][MetaKeys.relation]}")

    def test_default(self):
        # before running this test, set below environment variables:
        # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
        # export OPENAI_API_KEY=your_dashscope_key
        op = ExtractEntityRelationMapper(api_model='qwen2.5-72b-instruct')
        self._run_op(op)
    
    def test_entity_types(self):
        op = ExtractEntityRelationMapper(
            api_model='qwen2.5-72b-instruct',
            entity_types=['人物', '组织', '地点', '物件', '武器', '武功'],
        )
        self._run_op(op)
    
    def test_max_gleaning(self):
        op = ExtractEntityRelationMapper(
            api_model='qwen2.5-72b-instruct',
            entity_types=['人物', '组织', '地点', '物件', '武器', '武功'],
            max_gleaning=5,
        )
        self._run_op(op)


if __name__ == '__main__':
    unittest.main()
