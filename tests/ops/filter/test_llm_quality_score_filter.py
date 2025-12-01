import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.llm_quality_score_filter import LLMQualityScoreFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class LLMQualityScoreFilterTest(DataJuicerTestCaseBase):
    api_or_hf_model = 'qwen2.5-72b-instruct'

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
        op = LLMQualityScoreFilter(api_or_hf_model=self.api_or_hf_model)
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
