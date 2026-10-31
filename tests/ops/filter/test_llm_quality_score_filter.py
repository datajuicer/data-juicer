import json
import unittest
from unittest.mock import patch

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.llm_quality_score_filter import LLMQualityScoreFilter
from data_juicer.utils.constant import DEFAULT_API_MODEL, Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork


class LLMQualityScoreFilterNormalizationTest(unittest.TestCase):
    """Deterministic inherited-parser and threshold tests without model calls."""

    def _make_op(self, **kwargs):
        with patch('data_juicer.ops.filter.llm_analysis_filter.prepare_model',
                   return_value='mock-model'):
            return LLMQualityScoreFilter(api_or_hf_model='fake', **kwargs)

    def test_average_4_5_uses_normalized_threshold(self):
        for dimensions, scores in [
            (None, {'accuracy': 5, 'grammar': 4,
                    'informativeness': 4, 'coherence': 5}),
            (['accuracy', 'coherence'], {'accuracy': 5, 'grammar': 1,
                                        'informativeness': 1, 'coherence': 4}),
        ]:
            for threshold, keep in [(0.9, True), (0.91, False), (4.5, False)]:
                with self.subTest(dimensions=dimensions, threshold=threshold):
                    op = self._make_op(dim_required_keys=dimensions,
                                       min_score=threshold)
                    score, record, _ = op.parse_output(json.dumps({
                        'dimension_scores': scores,
                    }))
                    self.assertAlmostEqual(score, 0.9)
                    self.assertEqual(record['dimension_scores'], scores)
                    sample = {Fields.stats: {StatsKeys.llm_quality_score: score}}
                    self.assertEqual(op.process_single(sample), keep)

    def test_standard_range_and_unclamped_custom_scores(self):
        op = self._make_op()
        for dimension_score, expected in [(1, 0.2), (5, 1.0), (6, 1.2)]:
            with self.subTest(dimension_score=dimension_score):
                score, _, _ = op.parse_output(json.dumps({
                    'dimension_scores': {
                        key: dimension_score for key in op.dim_required_keys
                    },
                }))
                self.assertAlmostEqual(score, expected)


@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class LLMQualityScoreFilterTest(DataJuicerTestCaseBase):
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
        scores = [d[Fields.stats][StatsKeys.llm_quality_score] for d in dataset]
        for i in range(len(scores)-1):
            self.assertLess(scores[i], scores[i+1])
        dataset = dataset.filter(op.process, batch_size=op.batch_size)
        dataset_test = dataset.select_columns(column_names=['text'])
        res_list = dataset_test.to_list()
        self.assertLess(len(res_list), len(scores))
        return dataset

    def test_default_case(self):

        ds_list = [{
            'text':
            "The cat fly in sky and eat stars. It very happy because stars is tasty."
        }, {
            'text':
            "Cats are known for their agility and independence. They often sleep during the day and are active at night. Some people believe cats bring good luck."
        }, {
            'text':
            "Cats are domesticated animals known for their agility, intelligence, and independent nature. Research shows that they spend approximately 70% of their lives sleeping, which helps conserve energy for hunting. Unlike dogs, cats are obligate carnivores, meaning their diet must consist primarily of meat to meet nutritional needs."
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMQualityScoreFilter(
            api_or_hf_model=self.api_or_hf_model,
            sampling_params={'enable_thinking': False},
        )
        dataset= self._run_test(dataset, op)

    def test_rft_data(self):
        ds_list = [{
            "text": "What is the fastest animal?",
            "analysis": "The fastest animal is fish because they swim very fast in water.",
            "answer": "Fish."
        }, {
            "text": "Why do leaves change color in autumn?",
            "analysis": "Leaves change color because of the decrease in sunlight and temperature. Chlorophyll breaks down, revealing other pigments like yellow and orange.",
            "answer": "Due to less sunlight and colder temperatures, chlorophyll breaks down, showing other colors."
        }, {
            "text": "How does photosynthesis work?",
            "analysis": "Photosynthesis is the process by which plants convert light energy into chemical energy. Chlorophyll absorbs sunlight, which drives the conversion of carbon dioxide and water into glucose and oxygen. This process occurs in the chloroplasts of plant cells and is essential for life on Earth.",
            "answer": "Plants use chlorophyll to absorb sunlight, converting carbon dioxide and water into glucose and oxygen through a process that occurs in chloroplasts."
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMQualityScoreFilter(
            api_or_hf_model=self.api_or_hf_model,
            input_keys=['text', 'analysis', 'answer'],
            field_names=['Query', 'Analysis', 'Answer'],
            sampling_params={'enable_thinking': False},
        )
        dataset= self._run_test(dataset, op)

    # def test_vllm_case(self):

    #     ds_list = [{
    #         'text':
    #         "The cat fly in sky and eat stars. It very happy because stars is tasty."
    #     }, {
    #         'text':
    #         "Cats are known for their agility and independence. They often sleep during the day and are active at night. Some people believe cats bring good luck."
    #     }, {
    #         'text':
    #         "Cats are domesticated animals known for their agility, intelligence, and independent nature. Research shows that they spend approximately 70% of their lives sleeping, which helps conserve energy for hunting. Unlike dogs, cats are obligate carnivores, meaning their diet must consist primarily of meat to meet nutritional needs."
    #     }]
    #     dataset = Dataset.from_list(ds_list)
    #     op = LLMQualityScoreFilter(
    #           api_or_hf_model=self.api_or_hf_model,
    #           enable_vllm=True,
    #           accelerator='cuda'
    #       )
    #     dataset= self._run_test(dataset, op)

if __name__ == '__main__':
    unittest.main()
