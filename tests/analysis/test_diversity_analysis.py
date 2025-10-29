import os
import unittest
import pandas as pd

from data_juicer.core.data import NestedDataset
from data_juicer.analysis.diversity_analysis import find_root_verb_and_its_dobj_in_string, get_diversity, DiversityAnalysis
from data_juicer.utils.model_utils import prepare_model, get_model

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class FindRootVerbAndItsDobjInStringTest(DataJuicerTestCaseBase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = get_model(prepare_model('spacy', lang='en'))
        super().setUpClass()

    def test_basic_func(self):
        test_data = {
            'Sam is playing football.': ('play', 'football'),
            'Today is a sunny day': (None, None),  # no verb
            'Lily is reading': ('read', None),  # only verb
            '': (None, None)  # no sentence
        }
        for data, truth in test_data.items():
            res = find_root_verb_and_its_dobj_in_string(self.nlp, data)
            self.assertEqual(truth, res)

    def test_first_sentence_is_false(self):
        test_data = {
            'Sam is playing football. He is running.': ('play', 'football'),  # the first sentence is valid
            'Today is a sunny day. Sam is playing football.': ('play', 'football'),  # the first sentence is invalid
            'Today is a sunny day. Tomorrow it will be raining.': (None, None),  # no valid sentence
        }
        for data, truth in test_data.items():
            res = find_root_verb_and_its_dobj_in_string(self.nlp, data, first_sent=False)
            self.assertEqual(truth, res)

class GetDiversityTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()

        self.test_data = [
            {'verb': 'play', 'noun': 'football'},
            {'verb': 'read', 'noun': 'book'},
            {'verb': 'watch', 'noun': 'movie'},
            {'verb': 'read', 'noun': 'book'},
            {'verb': 'watch', 'noun': 'movie'},
            {'verb': 'play', 'noun': 'basketball'},
            {'verb': 'watch', 'noun': 'movie'},
            {'verb': 'read', 'noun': None},  # invalid
            {'verb': None, 'noun': None},  # invalid
        ]

    @staticmethod
    def list_of_dict_equal(l1, l2):
        return set(tuple(d.items()) for d in l1) == set(tuple(d.items()) for d in l2)

    def test_basic_func(self):
        res_data = [
            {'verb': 'play', 'noun': 'basketball', 'count': 1},
            {'verb': 'play', 'noun': 'football', 'count': 1},
            {'verb': 'read', 'noun': 'book', 'count': 2},
            {'verb': 'watch', 'noun': 'movie', 'count': 3},
        ]
        df = pd.DataFrame(self.test_data)
        res = get_diversity(df)
        self.assertTrue(self.list_of_dict_equal(res.to_dict(orient='records'), res_data))

    def test_top_k_verbs(self):
        res_data = [
            {'verb': 'play', 'noun': 'basketball', 'count': 1},
            {'verb': 'play', 'noun': 'football', 'count': 1},
            {'verb': 'watch', 'noun': 'movie', 'count': 3},
        ]
        df = pd.DataFrame(self.test_data)
        # only keep the top 2 verb groups
        res = get_diversity(df, top_k_verbs=2)
        self.assertTrue(self.list_of_dict_equal(res.to_dict(orient='records'), res_data))

    def test_top_k_nouns(self):
        res_data_1 = [{'verb': 'play', 'noun': 'basketball', 'count': 1},
                      {'verb': 'read', 'noun': 'book', 'count': 2},
                      {'verb': 'watch', 'noun': 'movie', 'count': 3},]
        res_data_2 = [{'verb': 'play', 'noun': 'football', 'count': 1},
                      {'verb': 'read', 'noun': 'book', 'count': 2},
                      {'verb': 'watch', 'noun': 'movie', 'count': 3},]
        df = pd.DataFrame(self.test_data)
        # only keep the top 1 noun for each verb group
        res = get_diversity(df, top_k_nouns=1)
        self.assertTrue(self.list_of_dict_equal(res.to_dict(orient='records'), res_data_1) or
                        self.list_of_dict_equal(res.to_dict(orient='records'), res_data_2))

class DiversityAnalysisTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()

        data_dict_en = {
            'text': [
                'Sam is playing football.',
                'Today is a sunny day',
                'Lily is reading',
                '',
                'Sam is playing football. He is running.',
                'Today is a sunny day. Sam is playing football.',
                'Today is a sunny day. Tomorrow it will be raining.',
            ]
        }

        data_dict_zh = {
            'content': [
                '山姆在踢足球',
                '今天是个晴朗的一天',
                '莉莉正在阅读',
                '',
                '山姆在踢足球。他正在跑动。',
                '今天是个晴朗的一天。山姆在踢足球。',
                '今天是个晴朗的一天。明天会下雨。',
            ]
        }

        self.test_data_en = NestedDataset.from_dict(data_dict_en)
        self.test_data_zh = NestedDataset.from_dict(data_dict_zh)

        self.temp_output_path = 'tmp/test_diversity_analysis/'

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')

        super().tearDown()

    def test_analyze(self):
        diversity_analysis = DiversityAnalysis(self.test_data_en, self.temp_output_path)
        df_en = diversity_analysis.analyze()
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'diversity.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'diversity.md')))
        self.assertEqual(df_en.to_dict(orient='records'), [
            {'verb': 'play', 'noun': 'football', 'count': 2},
        ])

    def test_analyze_zh(self):
        spacy_model = get_model(prepare_model('spacy', lang='zh'))
        diversity_analysis_zh = DiversityAnalysis(self.test_data_zh, self.temp_output_path, lang_or_model=spacy_model)
        df_zh = diversity_analysis_zh.analyze(column_name='content')
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'diversity.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_output_path, 'diversity.md')))
        self.assertEqual(df_zh.to_dict(orient='records'), [
            {'verb': '踢', 'noun': '足球', 'count': 2},
        ])

    def test_invalid_input(self):
        invalid_data = NestedDataset.from_dict({
            'text': [1, 2, 3]
        })
        diversity_analysis = DiversityAnalysis(invalid_data, self.temp_output_path)
        df = diversity_analysis.analyze()
        self.assertEqual(df.to_dict(orient='records'), [])


if __name__ == '__main__':
    unittest.main()
