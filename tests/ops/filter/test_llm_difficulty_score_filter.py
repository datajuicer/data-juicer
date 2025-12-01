import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.llm_difficulty_score_filter import LLMDifficultyScoreFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class LLMDifficultyScoreFilterTest(DataJuicerTestCaseBase):
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
        scores = [d[Fields.stats][StatsKeys.llm_difficulty_score] for d in dataset]
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
            "The cat sleeps on the mat. It is a sunny day, and the cat feels warm. Cats like to rest when they are not playing."
        }, {
            'text':
            "Photosynthesis is a process where plants convert sunlight into energy. Chlorophyll absorbs light, which triggers chemical reactions that produce glucose. This mechanism sustains ecosystems by providing food for herbivores and indirectly for carnivores."
        }, {
            'text':
            "In quantum field theory, renormalization addresses infinities arising from loop integrals in Feynman diagrams. By redefining parameters such as mass and charge, physicists ensure finite predictions align with experimental observations. However, this procedure raises philosophical questions about whether these adjustments reflect physical reality or merely mathematical conveniences."
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMDifficultyScoreFilter(api_or_hf_model=self.api_or_hf_model)
        dataset= self._run_test(dataset, op)

    def test_rft_data(self):
        ds_list = [{
            "text": "What is the capital of France?",
            "analysis": "The question asks for a factual piece of information about the capital city of France. The answer is straightforward and does not require any specialized knowledge or complex reasoning.",
            "answer": "The capital of France is Paris."
        }, {
            "text": "How does photosynthesis contribute to the carbon cycle?",
            "analysis": "This question requires an understanding of both photosynthesis and the carbon cycle. It involves explaining how plants absorb carbon dioxide during photosynthesis, convert it into organic matter, and release oxygen, thus influencing atmospheric carbon levels. The explanation requires linking two scientific concepts.",
            "answer": "Photosynthesis contributes to the carbon cycle by absorbing carbon dioxide from the atmosphere. Plants use sunlight to convert this carbon dioxide into glucose and other organic compounds, storing carbon in their biomass. When plants die or are consumed, the carbon is released back into the environment through decomposition or respiration, completing the cycle."
        }, {
            "text": "What are the implications of Gödel's incompleteness theorems for formal systems in mathematics?",
            "analysis": "This question delves into advanced theoretical mathematics and philosophy. Gödel's incompleteness theorems state that in any sufficiently powerful formal system, there will always be true statements that cannot be proven within the system. This has profound implications for the limits of mathematical reasoning and the nature of truth. To address this, one must understand formal logic, axiomatic systems, and the philosophical debates surrounding mathematical foundations.",
            "answer": "Gödel's incompleteness theorems imply that no formal system capable of expressing arithmetic can be both consistent and complete. There will always exist true statements that cannot be derived from the axioms of the system. This challenges the notion of a single, all-encompassing formal system for mathematics and highlights the inherent limitations of logical reasoning. Philosophically, it raises questions about whether mathematical truth is independent of human construction or if it is inherently incomplete."
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMDifficultyScoreFilter(
            api_or_hf_model=self.api_or_hf_model,
            input_keys=['text', 'analysis', 'answer'],
            field_names=['Query', 'Analysis', 'Answer'],
        )
        dataset= self._run_test(dataset, op)

    # def test_vllm_case(self):

    #     ds_list = [{
    #         'text':
    #         "The cat sleeps on the mat. It is a sunny day, and the cat feels warm. Cats like to rest when they are not playing."
    #     }, {
    #         'text':
    #         "In quantum field theory, renormalization addresses infinities arising from loop integrals in Feynman diagrams. By redefining parameters such as mass and charge, physicists ensure finite predictions align with experimental observations. However, this procedure raises philosophical questions about whether these adjustments reflect physical reality or merely mathematical conveniences."
    #     }, {
    #        'text':
    #        "In quantum field theory, renormalization addresses infinities arising from loop integrals in Feynman diagrams. By redefining parameters such as mass and charge, physicists ensure finite predictions align with experimental observations. However, this procedure raises philosophical questions about whether these adjustments reflect physical reality or merely mathematical conveniences."
    #     }]
    #     dataset = Dataset.from_list(ds_list)
    #     op = LLMDifficultyScoreFilter(
    #           api_or_hf_model=self.api_or_hf_model,
    #           enable_vllm=True,
    #           accelerator='cuda'
    #       )
    #     dataset= self._run_test(dataset, op)

if __name__ == '__main__':
    unittest.main()
