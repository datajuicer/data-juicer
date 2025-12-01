import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.dialog_topic_detection_mapper import DialogTopicDetectionMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK
from data_juicer.utils.constant import Fields, MetaKeys

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class TestDialogTopicDetectionMapper(DataJuicerTestCaseBase):
    # before running this test, set below environment variables:
    # export OPENAI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    # export OPENAI_API_KEY=your_key

    def _run_op(self, op, samples, target_len, labels_key=None, analysis_key=None):
        dataset = Dataset.from_list(samples)
        dataset = op.run(dataset)
        
        labels_key = labels_key or MetaKeys.dialog_topic_labels
        analysis_key = analysis_key or MetaKeys.dialog_topic_labels_analysis
        labels_list = dataset[0][Fields.meta][labels_key]
        analysis_list = dataset[0][Fields.meta][analysis_key]

        for analysis, labels in zip(analysis_list, labels_list):
            logger.info(f'分析：{analysis}')
            logger.info(f'话题：{labels}')
            self.assertNotEqual(analysis, '')
            self.assertNotEqual(labels, '')
        
        self.assertEqual(len(analysis_list), target_len)
        self.assertEqual(len(labels_list), target_len)
        
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

        op = DialogTopicDetectionMapper(api_model='qwen2.5-72b-instruct')
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

        op = DialogTopicDetectionMapper(api_model='qwen2.5-72b-instruct',
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

        op = DialogTopicDetectionMapper(api_model='qwen2.5-72b-instruct',
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

        op = DialogTopicDetectionMapper(api_model='qwen2.5-72b-instruct',
                                            max_round=1)
        self._run_op(op, samples, 4)

    def test_topic_candidates(self):
        
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

        op = DialogTopicDetectionMapper(api_model='qwen2.5-72b-instruct',
                                topic_candidates=['评价', '沟通', '闲聊', '其他'])
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

        labels_key = 'my_label'
        analysis_key = 'my_analysis'
        op = DialogTopicDetectionMapper(api_model='qwen2.5-72b-instruct',
                                        labels_key=labels_key,
                                        analysis_key=analysis_key)
        self._run_op(op, samples, 4, labels_key, analysis_key)


if __name__ == '__main__':
    unittest.main()
