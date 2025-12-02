import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.dialog_sentiment_intensity_mapper import DialogSentimentIntensityMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK
from data_juicer.utils.constant import Fields, MetaKeys

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class TestDialogSentimentIntensityMapper(DataJuicerTestCaseBase):
    # before running this test, set below environment variables:
    # export OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    # export OPENAI_API_KEY=your_key

    def _run_op(self, op, samples, target_len, intensities_key=None, analysis_key=None):
        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)
        intensities_key = intensities_key or MetaKeys.dialog_sentiment_intensity
        analysis_key = analysis_key or MetaKeys.dialog_sentiment_intensity_analysis
        intensity_list = dataset[0][Fields.meta][intensities_key]
        analysis_list = dataset[0][Fields.meta][analysis_key]

        for analysis, intensity in zip(analysis_list, intensity_list):
            logger.info(f'分析：{analysis}')
            logger.info(f'情绪：{intensity}')
            self.assertNotEqual(analysis, '')
        
        self.assertEqual(len(analysis_list), target_len)
        self.assertEqual(len(intensity_list), target_len)
        
    def test_default(self):
        
        samples = [{
            'history': [
                (
                    '李莲花有口皆碑',
                    '「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。'
                ),
                (
                    '是的，你确实是一个普通大夫，没什么值得夸耀的。',
                    '「委屈」你这话说的，我也是尽心尽力治病救人了。'
                ),
                (
                    '你自己说的呀，我现在说了，你又不高兴了。',
                    'or of of of of or or and or of of of of of of of,,, '
                ),
                (
                    '你在说什么我听不懂。',
                    '「委屈」我也没说什么呀，就是觉得你有点冤枉我了'
                )
            ]
        }]

        op = DialogSentimentIntensityMapper(api_model='qwen2.5-72b-instruct')
        self._run_op(op, samples, 4)
    
    def test_max_round(self):

        samples = [{
            'history': [
                (
                    '李莲花有口皆碑',
                    '「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。'
                ),
                (
                    '是的，你确实是一个普通大夫，没什么值得夸耀的。',
                    '「委屈」你这话说的，我也是尽心尽力治病救人了。'
                ),
                (
                    '你自己说的呀，我现在说了，你又不高兴了。',
                    'or of of of of or or and or of of of of of of of,,, '
                ),
                (
                    '你在说什么我听不懂。',
                    '「委屈」我也没说什么呀，就是觉得你有点冤枉我了'
                )
            ]
        }]

        op = DialogSentimentIntensityMapper(api_model='qwen2.5-72b-instruct',
                                            max_round=1)
        self._run_op(op, samples, 4)

    def test_max_round_zero(self):

        samples = [{
            'history': [
                (
                    '李莲花有口皆碑',
                    '「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。'
                ),
                (
                    '是的，你确实是一个普通大夫，没什么值得夸耀的。',
                    '「委屈」你这话说的，我也是尽心尽力治病救人了。'
                ),
                (
                    '你自己说的呀，我现在说了，你又不高兴了。',
                    'or of of of of or or and or of of of of of of of,,, '
                ),
                (
                    '你在说什么我听不懂。',
                    '「委屈」我也没说什么呀，就是觉得你有点冤枉我了'
                )
            ]
        }]

        op = DialogSentimentIntensityMapper(api_model='qwen2.5-72b-instruct',
                                            max_round=0)
        self._run_op(op, samples, 4)

    def test_query(self):

        samples = [{
            'history': [
                (
                    '李莲花有口皆碑',
                    '「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。'
                ),
                (
                    '是的，你确实是一个普通大夫，没什么值得夸耀的。',
                    '「委屈」你这话说的，我也是尽心尽力治病救人了。'
                ),
                (
                    '你自己说的呀，我现在说了，你又不高兴了。',
                    'or of of of of or or and or of of of of of of of,,, '
                )
            ],
            'query': '你在说什么我听不懂。',
            'response': '「委屈」我也没说什么呀，就是觉得你有点冤枉我了'
        }]

        op = DialogSentimentIntensityMapper(api_model='qwen2.5-72b-instruct',
                                            max_round=1)
        self._run_op(op, samples, 4)

    def test_rename_keys(self):
        
        samples = [{
            'history': [
                (
                    '李莲花有口皆碑',
                    '「微笑」过奖了，我也就是个普通大夫，没什么值得夸耀的。'
                ),
                (
                    '是的，你确实是一个普通大夫，没什么值得夸耀的。',
                    '「委屈」你这话说的，我也是尽心尽力治病救人了。'
                ),
                (
                    '你自己说的呀，我现在说了，你又不高兴了。',
                    'or of of of of or or and or of of of of of of of,,, '
                ),
                (
                    '你在说什么我听不懂。',
                    '「委屈」我也没说什么呀，就是觉得你有点冤枉我了'
                )
            ]
        }]

        intensities_key = 'my_intensity'
        analysis_key = 'my_analysis'
        op = DialogSentimentIntensityMapper(api_model='qwen2.5-72b-instruct',
                                            intensities_key=intensities_key,
                                            analysis_key=analysis_key)
        self._run_op(op, samples, 4, intensities_key, analysis_key)


if __name__ == '__main__':
    unittest.main()
