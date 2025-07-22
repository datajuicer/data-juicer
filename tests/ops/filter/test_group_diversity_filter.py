import os
import unittest
from loguru import logger
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.group_diversity_filter import GroupDiversityFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping the test because running from a fork repo")
class GroupDiversityFilterTest(DataJuicerTestCaseBase):
    # before running this test, set below environment variables:
    # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
    # export OPENAI_API_KEY=your_dashscope_key
    api_model_name = 'text-embedding-v3'
    api_ebd_dim = 512

    # For local Hugging Face model test
    hf_model_path = 'iic/gte_Qwen2-1.5B-instruct'
    def setUp(self):
        self.ds_list = [{
            'text': "A cute cat is playing in the garden."
        }, {
            'text': "A lovely dog is running on the grass."
        }, {
            'text': "A beautiful bird is singing on the tree."
        }, {
            'text': "Quantum computing is a complex field of physics." # The outlier
        }]
        self.dataset = Dataset.from_list(self.ds_list)

    def test_api_based_diversity_logic(self):
        if not os.getenv('OPENAI_API_KEY'):
            self.skipTest("OPENAI_API_KEY environment variable is not set. "
                          "Skipping API-based integration test.")

        logger.info(f"Running diversity test with API model: {self.api_model_name}")
        op = GroupDiversityFilter(
            api_or_hf_model=self.api_model_name,
            is_hf_model=False,
            ebd_dim=self.api_ebd_dim
        )
        self._run_and_assert_diversity(op)

    def test_hf_based_diversity_logic(self):
        logger.info(f"Running diversity test with HF model: {self.hf_model_path}")
        op = GroupDiversityFilter(
            api_or_hf_model=self.hf_model_path,
            is_hf_model=True,
        )
        self._run_and_assert_diversity(op)

    def _run_and_assert_diversity(self, op: GroupDiversityFilter):
        dataset = self.dataset.add_column(name=Fields.stats, column=[{}] * len(self.dataset))
        dataset = dataset.map(op.compute_stats_batched,
                              with_rank=True,
                              batched=True,
                              batch_size=len(self.dataset))
        
        stats_list = dataset.to_list()
        for sample in stats_list:
            logger.info(f"Text: '{sample['text']}', "
                        f"Score: {sample[Fields.stats].get(StatsKeys.text_ebd_diversity_score, 'N/A')}")

        scores = [d[Fields.stats][StatsKeys.text_ebd_diversity_score] for d in stats_list]
        
        outlier_score = scores[-1]
        other_scores = scores[:-1]
        
        self.assertTrue(all(outlier_score > score for score in other_scores),
                        "The outlier sample did not receive the highest diversity score.")
        
        logger.info("Test passed: The outlier sample correctly received the highest diversity score.")


if __name__ == '__main__':
    unittest.main()
