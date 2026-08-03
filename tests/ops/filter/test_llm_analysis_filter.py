import json
import unittest

import numpy as np
from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.llm_analysis_filter import LLMAnalysisFilter
from data_juicer.utils.constant import DEFAULT_API_MODEL, Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork


class LLMAnalysisFilterParseOutputTest(DataJuicerTestCaseBase):
    """Tests for parse_output and static helper methods (no API needed)."""

    def _make_op(self):
        return LLMAnalysisFilter(api_or_hf_model='fake')

    # -- parse_output ---

    def test_parse_output_none(self):
        op = self._make_op()
        score, record, tags = op.parse_output(None)
        self.assertEqual(score, 0.0)
        self.assertIsNone(record)
        self.assertIsNone(tags)

    def test_parse_output_empty_string(self):
        op = self._make_op()
        score, record, tags = op.parse_output('')
        self.assertEqual(score, 0.0)
        self.assertIsNone(record)
        self.assertIsNone(tags)

    def test_parse_output_whitespace_only(self):
        op = self._make_op()
        score, record, tags = op.parse_output('   ')
        self.assertEqual(score, 0.0)
        self.assertIsNone(record)
        self.assertIsNone(tags)

    def test_parse_output_no_json(self):
        op = self._make_op()
        score, record, tags = op.parse_output('no json here at all')
        self.assertEqual(score, 0.0)
        self.assertIsNone(record)
        self.assertIsNone(tags)

    def test_parse_output_invalid_json(self):
        op = self._make_op()
        score, record, tags = op.parse_output('{invalid json}')
        self.assertEqual(score, 0.0)
        self.assertIsNone(record)
        self.assertIsNone(tags)

    def test_parse_output_valid_full(self):
        op = self._make_op()
        raw = json.dumps({
            'dimension_scores': {
                'clarity': 4, 'relevance': 5,
                'usefulness': 3, 'fluency': 4,
            },
            'tags': {'topic': 'AI', 'style': 'Technical'},
            'flags': [],
            'rationale': 'Good quality.',
            'recommendation': ['keep'],
        })
        score, record, tags = op.parse_output(raw)
        # (4+5+3+4) / 4 / 5 = 16/20 = 0.8
        self.assertAlmostEqual(score, 0.8)
        self.assertIsNotNone(record)
        self.assertEqual(tags, {'topic': 'AI', 'style': 'Technical'})

    def test_parse_output_missing_dimension_scores(self):
        op = self._make_op()
        raw = json.dumps({
            'tags': {'topic': 'Test'},
            'recommendation': 'review',
        })
        score, record, tags = op.parse_output(raw)
        self.assertEqual(score, 0.0)
        self.assertIsNotNone(record)

    def test_parse_output_json_with_surrounding_text(self):
        """parse_output should extract JSON even if surrounded by text."""
        op = self._make_op()
        raw = 'Here is the result:\n' + json.dumps({
            'dimension_scores': {
                'clarity': 3, 'relevance': 3,
                'usefulness': 3, 'fluency': 3,
            },
            'recommendation': ['review'],
        }) + '\nEnd of response.'
        score, record, tags = op.parse_output(raw)
        self.assertAlmostEqual(score, 0.6)
        self.assertIsNotNone(record)

    def test_parse_output_custom_dim_keys(self):
        op = LLMAnalysisFilter(
            api_or_hf_model='fake',
            dim_required_keys=['clarity', 'fluency'],
        )
        raw = json.dumps({
            'dimension_scores': {
                'clarity': 5, 'relevance': 1,
                'usefulness': 1, 'fluency': 5,
            },
            'recommendation': [],
        })
        score, record, tags = op.parse_output(raw)
        # Only clarity(5) + fluency(5) => 10/2/5 = 1.0
        self.assertAlmostEqual(score, 1.0)

    # -- _normalize_recommendation_to_str_list ---

    def test_normalize_recommendation_none(self):
        result = LLMAnalysisFilter._normalize_recommendation_to_str_list(None)
        self.assertEqual(result, [])

    def test_normalize_recommendation_string(self):
        result = LLMAnalysisFilter._normalize_recommendation_to_str_list(
            'keep')
        self.assertEqual(result, ['keep'])

    def test_normalize_recommendation_empty_string(self):
        result = LLMAnalysisFilter._normalize_recommendation_to_str_list('  ')
        self.assertEqual(result, [])

    def test_normalize_recommendation_list(self):
        result = LLMAnalysisFilter._normalize_recommendation_to_str_list(
            ['keep', 'review'])
        self.assertEqual(result, ['keep', 'review'])

    def test_normalize_recommendation_list_with_none(self):
        result = LLMAnalysisFilter._normalize_recommendation_to_str_list(
            ['keep', None, 'review'])
        self.assertEqual(result, ['keep', 'review'])

    def test_normalize_recommendation_numpy_array(self):
        arr = np.array(['keep', 'review'])
        result = LLMAnalysisFilter._normalize_recommendation_to_str_list(arr)
        self.assertEqual(result, ['keep', 'review'])

    def test_normalize_recommendation_other_type(self):
        result = LLMAnalysisFilter._normalize_recommendation_to_str_list(42)
        self.assertEqual(result, ['42'])

    # -- _normalize_tags_to_str ---

    def test_normalize_tags_to_str_none(self):
        result = LLMAnalysisFilter._normalize_tags_to_str(None)
        self.assertEqual(result, '')

    def test_normalize_tags_to_str_string(self):
        result = LLMAnalysisFilter._normalize_tags_to_str('already a string')
        self.assertEqual(result, 'already a string')

    def test_normalize_tags_to_str_dict(self):
        tags = {'topic': 'AI', 'style': 'Technical'}
        result = LLMAnalysisFilter._normalize_tags_to_str(tags)
        parsed = json.loads(result)
        self.assertEqual(parsed, tags)

    def test_normalize_tags_to_str_list(self):
        tags = ['tag1', 'tag2']
        result = LLMAnalysisFilter._normalize_tags_to_str(tags)
        parsed = json.loads(result)
        self.assertEqual(parsed, tags)

    # -- _normalize_tags_to_dict ---

    def test_normalize_tags_to_dict_none(self):
        result = LLMAnalysisFilter._normalize_tags_to_dict(None)
        self.assertEqual(result, {})

    def test_normalize_tags_to_dict_dict(self):
        tags = {'topic': 'AI', 'count': 5}
        result = LLMAnalysisFilter._normalize_tags_to_dict(tags)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['topic'], 'AI')
        # Non-string values are coerced to string
        self.assertEqual(result['count'], '5')

    def test_normalize_tags_to_dict_string(self):
        result = LLMAnalysisFilter._normalize_tags_to_dict('plain tag')
        self.assertEqual(result, {'tags': 'plain tag'})

    def test_normalize_tags_to_dict_list(self):
        result = LLMAnalysisFilter._normalize_tags_to_dict(['tag1', 'tag2'])
        self.assertEqual(result, {'tags': 'tag1, tag2'})

    # -- _normalize_record ---

    def test_normalize_record_none(self):
        result = LLMAnalysisFilter._normalize_record(None)
        self.assertEqual(result, '')

    def test_normalize_record_dict(self):
        record = {'dimension_scores': {'clarity': 4}, 'tags': {'topic': 'AI'}}
        result = LLMAnalysisFilter._normalize_record(record)
        parsed = json.loads(result)
        self.assertEqual(parsed, record)

    # -- build_input ---

    def test_build_input_default(self):
        op = self._make_op()
        sample = {'text': 'Hello world'}
        result = op.build_input(sample)
        self.assertIn('Hello world', result)
        self.assertIn('Text', result)

    def test_build_input_multi_field(self):
        op = LLMAnalysisFilter(
            api_or_hf_model='fake',
            input_keys=['text', 'answer'],
            field_names=['Query', 'Answer'],
        )
        sample = {'text': 'What is AI?', 'answer': 'Artificial Intelligence'}
        result = op.build_input(sample)
        self.assertIn('What is AI?', result)
        self.assertIn('Artificial Intelligence', result)
        self.assertIn('Query', result)
        self.assertIn('Answer', result)

    def test_build_input_missing_key(self):
        op = LLMAnalysisFilter(
            api_or_hf_model='fake',
            input_keys=['text', 'missing_key'],
            field_names=['Text', 'Missing'],
        )
        sample = {'text': 'Hello'}
        # Should not raise; only available keys are formatted
        result = op.build_input(sample)
        self.assertIn('Hello', result)
        self.assertNotIn('Missing', result)

    # -- process_single ---

    def test_process_single_zero_score_returns_true(self):
        """When LLM analysis failed (score=0.0), the filter should not
        discard the sample."""
        op = self._make_op()
        sample = {Fields.stats: {StatsKeys.llm_analysis_score: 0.0}}
        result = op.process_single(sample)
        self.assertTrue(result)

    def test_process_single_within_range(self):
        op = LLMAnalysisFilter(
            api_or_hf_model='fake', min_score=0.5, max_score=1.0)
        sample = {Fields.stats: {StatsKeys.llm_analysis_score: 0.7}}
        result = op.process_single(sample)
        self.assertTrue(result)

    def test_process_single_below_range(self):
        op = LLMAnalysisFilter(
            api_or_hf_model='fake', min_score=0.5, max_score=1.0)
        sample = {Fields.stats: {StatsKeys.llm_analysis_score: 0.3}}
        result = op.process_single(sample)
        self.assertFalse(result)

    def test_preferred_output_lang(self):
        """Verify that preferred_output_lang appends language instructions."""
        op = LLMAnalysisFilter(
            api_or_hf_model='fake', preferred_output_lang='zh')
        self.assertIn(
            LLMAnalysisFilter.DEFAULT_SYSTEM_PROMPT, op.system_prompt)
        # The prompt should be longer than default because of language appendix
        self.assertGreater(
            len(op.system_prompt),
            len(LLMAnalysisFilter.DEFAULT_SYSTEM_PROMPT))


@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class LLMAnalysisFilterTest(DataJuicerTestCaseBase):
    api_or_hf_model = DEFAULT_API_MODEL

    def _run_test(self, dataset: Dataset, op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                       column=[{}] * dataset.num_rows)

        dataset = dataset.map(
            op.compute_stats,
            batch_size=op.batch_size
        )
        logger.info(dataset.to_list())
        
        scores = [d[Fields.stats].get(StatsKeys.llm_analysis_score) for d in dataset]
        for i in range(len(scores)-1):
            self.assertLess(scores[i], scores[i+1])

        for d in dataset:
            stats = d[Fields.stats]
            # Tags are now stored under a single fixed key as JSON string
            tags_str = stats.get(StatsKeys.llm_analysis_tags, "")
            self.assertIsInstance(tags_str, str)
        
        dataset = dataset.filter(op.process, batch_size=op.batch_size)
        dataset_test = dataset.select_columns(column_names=['text'])
        res_list = dataset_test.to_list()
        self.assertLess(len(res_list), len(scores))
        return dataset

    def test_default_case(self):
        ds_list = [{
            'text': "cat dog run jump very fast and happy today weather good."
        }, {
            'text': "The research paper presents findings in quantum computing. It shows a new way to handle qubits that helps reduce some problems. The writing could be clearer and more detailed."
        }, {
            'text': "This comprehensive study examines the impact of climate change on global ecosystems, providing detailed analysis supported by extensive data collection over a decade. The research methodology includes rigorous statistical analysis and peer reviews from leading experts in environmental science."
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMAnalysisFilter(
            api_or_hf_model=self.api_or_hf_model,
            sampling_params={"enable_thinking": False},
        )
        dataset = self._run_test(dataset, op)

    def test_rft_data(self):
        ds_list = [{
            "text": "What is the fastest land animal?",
            "analysis": "Fish is the fastest land animal because it swims in the ocean, flies above trees, and every animal is the same speed.",
            "answer": "Fish."
        }, {
            "text": "Why do leaves change color in autumn?",
            "analysis": "As days get shorter, trees stop replacing chlorophyll. The green color fades and yellow or orange pigments that were already in the leaves become visible, though this skips some details such as red pigments.",
            "answer": "Shorter daylight reduces chlorophyll, revealing other pigments in the leaves."
        }, {
            "text": "How does photosynthesis work?",
            "analysis": "Photosynthesis is the biochemical process by which green plants convert light energy into chemical energy stored in glucose. Chlorophyll in chloroplasts absorbs photons, driving the light-dependent reactions that produce ATP and NADPH. These then fuel the Calvin cycle, fixing CO2 into glyceraldehyde-3-phosphate, which is subsequently converted to glucose. Oxygen is released as a byproduct from water splitting.",
            "answer": "Plants use chlorophyll to absorb sunlight, converting carbon dioxide and water into glucose and oxygen through light-dependent reactions and the Calvin cycle."
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMAnalysisFilter(
            api_or_hf_model=self.api_or_hf_model,
            input_keys=['text', 'analysis', 'answer'],
            field_names=['Query', 'Analysis', 'Answer'],
            min_score=0.7,
            sampling_params={"enable_thinking": False},
        )
        dataset = self._run_test(dataset, op)

    def test_custom_dimension_keys(self):
        ds_list = ds_list = [{
            'text': "text very bad grammar unclear meaning hard read understand what say."
        }, {
            'text': "This text reads okay but has some minor grammar mistakes and unclear parts."
        }, {
            'text': "The sentence structure is clear and the grammar is impeccable, making it easy to understand the concept."
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMAnalysisFilter(
            api_or_hf_model=self.api_or_hf_model,
            dim_required_keys=["clarity", "fluency"],
            sampling_params={"enable_thinking": False},
        )
        dataset = self._run_test(dataset, op)

if __name__ == '__main__':
    unittest.main()
