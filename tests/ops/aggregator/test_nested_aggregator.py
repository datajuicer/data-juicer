import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.aggregator import NestedAggregator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

from data_juicer.utils.constant import Fields, MetaKeys

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class NestedAggregatorTest(DataJuicerTestCaseBase):

    def _run_helper(self, op, samples, output_key=MetaKeys.event_description):

        # before running this test, set below environment variables:
        # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
        # export OPENAI_API_KEY=your_dashscope_key

        dataset = Dataset.from_list(samples)
        new_dataset = op.run(dataset)

        for data in new_dataset:
            for k in data:
                logger.info(f"{k}: {data[k]}")
            self.assertIn(output_key, data[Fields.batch_meta])
            self.assertNotEqual(data[Fields.batch_meta][output_key], '')

        self.assertEqual(len(new_dataset), len(samples))

    def test_default_aggregator(self):
        samples = [
            {
                Fields.meta: [
                    {MetaKeys.event_description: "十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。"},
                    {MetaKeys.event_description: "有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。"},
                    {MetaKeys.event_description: '在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。'},
                    {MetaKeys.event_description: '十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。'},
                    {MetaKeys.event_description: '小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。'}
                ]
            },
        ]
        op = NestedAggregator(
            api_model='qwen2.5-72b-instruct'
        )
        self._run_helper(op, samples)
    
    def test_input_output(self):
        samples = [
            {
                Fields.meta: [
                    {'sub_docs': "十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。"},
                    {'sub_docs': "有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。"},
                    {'sub_docs': '在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。'},
                    {'sub_docs': '十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。'},
                    {'sub_docs': '小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。'}
                ]
            },
        ]
        op = NestedAggregator(
            api_model='qwen2.5-72b-instruct',
            input_key='sub_docs',
            output_key='text'
        )
        self._run_helper(op, samples, output_key='text')

    def test_max_token_num_1(self):
        samples = [
            {
                Fields.meta: [
                    {MetaKeys.event_description: "十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。"},
                    {MetaKeys.event_description: "有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。"},
                    {MetaKeys.event_description: '在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。'},
                    {MetaKeys.event_description: '十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。'},
                    {MetaKeys.event_description: '小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。'}
                ]
            },
        ]
        op = NestedAggregator(
            api_model='qwen2.5-72b-instruct',
            max_token_num=2
        )
        self._run_helper(op, samples)

    def test_max_token_num_2(self):
        samples = [
            {
                Fields.meta: [
                    {MetaKeys.event_description: "十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。"},
                    {MetaKeys.event_description: "有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。"},
                    {MetaKeys.event_description: '在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。'},
                    {MetaKeys.event_description: '十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。'},
                    {MetaKeys.event_description: '小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。'}
                ]
            },
        ]
        op = NestedAggregator(
            api_model='qwen2.5-72b-instruct',
            max_token_num=90
        )
        self._run_helper(op, samples)

    def test_max_token_num_3(self):
        samples = [
            {
                Fields.meta: [
                    {MetaKeys.event_description: "十年前，李相夷十五岁战胜西域天魔成为天下第一高手，十七岁建立四顾门，二十岁问鼎武林盟主，成为传奇人物。"},
                    {MetaKeys.event_description: "有人视李相夷为中原武林的希望，但也有人以战胜他为目标，包括魔教金鸳盟盟主笛飞声。笛飞声设计加害李相夷的师兄单孤刀，引得李相夷与之一战。"},
                    {MetaKeys.event_description: '在东海的一艘船上，李相夷独自一人对抗金鸳盟的高手，最终击败了大部分敌人。笛飞声突然出现，两人激战，李相夷在战斗中中毒，最终被笛飞声重伤，船只爆炸，李相夷沉入大海。'},
                    {MetaKeys.event_description: '十年后，李莲花在一个寒酸的莲花楼内醒来，表现出与李相夷截然不同的性格。他以神医的身份在小镇上行医，但生活贫困。'},
                    {MetaKeys.event_description: '小镇上的皮影戏摊讲述李相夷和笛飞声的故事，孩子们争论谁赢了。风火堂管事带着人来找李莲花，要求他救治一个“死人”。'}
                ]
            },
        ]
        op = NestedAggregator(
            api_model='qwen2.5-72b-instruct',
            max_token_num=200
        )
        self._run_helper(op, samples)

if __name__ == '__main__':
    unittest.main()