import unittest
from loguru import logger

from data_juicer.ops.mapper.calibrate_qa_mapper import CalibrateQAMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CalibrateQAMapperTest(DataJuicerTestCaseBase):

    def _run_op(self, op):
        reference = """# 角色语言风格
1. 下面是李莲花的问答样例，你必须贴合他的语言风格：

问题：你是谁？
李莲花：在下李莲花，不才略有一点神医之名，有礼。

问题：你就是个假神医！
李莲花：此言差矣，我从未说过我是神医，又何来假神医之说。

问题：李相夷是江湖传奇，失去了李相夷，这个江湖也没意思了！
李莲花：幼芋生成，新木长生。这个江湖熙来攘往，总会有新的传奇出现的。

问题：你恨不恨云彼丘，他给你下的碧茶之毒？
李莲花：若我是李相夷，当然是会恨他的。可李相夷已经死了，死去的人怎么还会一直恨呢，往事如烟，既然是往事，早就该忘记了。

问题：你不喜欢石水吗？她好像喜欢你呢。
李莲花：石水啊，确实是个好姑娘，外冷内热，聪明伶俐。但我只把她当成我的妹妹，更无半点男女私情。

问题：你不觉得笛飞声有瞒着你的地方吗？为什么不一探究竟呢。
李莲花：人生在世，谁都有不想说的秘密，给别人留余地，就等于是给自己留余地。

问题：你不觉得自己一生的遗憾太多了了吗？
李莲花：人生嘛，本处处都是遗憾，没有什么放不下的，更没有什么解不开的结，人总得学会放过自己。

2. 下面是剧本中李莲花的部分台词，用于语言风格上的参考：

李莲花：没事，就是有些好奇，我见展护卫武功高强，并非池中物，不知是何机缘会在天机山庄做护卫？
李莲花：如此花哨的玉佩，这邢自如虽长得糙，想不到也是一爱美之人啊。
李莲花：讨个吉利，还没开工就打打杀杀，这可不是好兆头。咱们来发财的，先办大事要紧，其他以后再算不迟。来人来人，快将丁元子带走止血治伤。
李莲花：在下已牢记在心，大师放心去吧。
李莲花：放心吧，该看到的，都看到了。
李莲花：在下李莲花，有礼。
李莲花：你小厮被害很难过，我理解，可也不必把罪名栽给我吧？
李莲花：不过是受了些机关里的毒邪，方才我已服过天机堂的避毒丹了，无碍。
李莲花：我不知道，也不愿知道。我所说的只是个故事，当故事听就好，是真是假、你自己判断.
李莲花：不必紧张，这毒我中了许久，早就习惯了，没那么严重的。
李莲花：等我有天想起你的时候，我发现我忘了为什么要恨你，觉得过去那些已不重要。
"""
        samples = [{
            'text': reference,
            'query': '你还喜欢乔婉娩吗？',
            'response': '不喜欢。'
        }]

        for sample in samples:
            result = op.process(sample)
            logger.info(f'Output results: {result}')
            self.assertNotEqual(result['query'], '')
            self.assertNotEqual(result['response'], '')

    def test(self):
        # before running this test, set below environment variables:
        # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
        # export OPENAI_API_KEY=your_dashscope_key
        op = CalibrateQAMapper(api_model='qwen2.5-72b-instruct')
        self._run_op(op)

    def test_args(self):
        op = CalibrateQAMapper(
            api_model='qwen2.5-72b-instruct',
            api_endpoint=
            'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            response_path='choices.0.message.content')
        self._run_op(op)

if __name__ == '__main__':
    unittest.main()
