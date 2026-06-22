import unittest
from loguru import logger
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.llm_analysis_filter import LLMAnalysisFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, skip_if_from_fork

@skip_if_from_fork("Skipping API-based test because running from a fork repo")
class LLMAnalysisFilterTest(DataJuicerTestCaseBase):
    api_or_hf_model = 'qwen3.7-max'

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
