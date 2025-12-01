import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.llm_task_relevance_filter import LLMTaskRelevanceFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class LLMTaskRelevanceFilterTest(DataJuicerTestCaseBase):
    api_or_hf_model = 'qwen2.5-72b-instruct'

    def _run_test(self, dataset: Dataset, op, tgt_list):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(
            op.compute_stats,
            batch_size=op.batch_size
            )
        logger.info(dataset.to_list())
        dataset = dataset.filter(op.process, batch_size=op.batch_size)
        dataset_test = dataset.select_columns(column_names=['text'])
        res_list = dataset_test.to_list()
        self.assertEqual(res_list, tgt_list)

    def test_default_case(self):
        ds_list = [
            {"text": "It is challenging to train a large language model."},
            {"text": "Q: What is the capital of France? A: The question asks for a factual piece of information about the capital city of France. The answer is straightforward and does not require any specialized knowledge or complex reasoning. The capital of France is Paris."}, # noqa: E501
            {"text": "Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year? A: He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 6*2=<<6*2=12>>12 pages every week That means he writes 12*52=<<12*52=624>>624 pages a year #### 624"} # noqa: E501
        ]
        vs_list = [
            {"text": "Q: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden? A: There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers. So in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers. Purple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers. That means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers. So in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden. #### 35"}, # noqa: E501
            {"text": "Q: A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn? A: From the details given, the car has traveled 5 meters at the 1st turn + 8 meters after the 2nd turn + 0 meters after the 4th turn = <<5+8+0=13>>13 meters around the ring. It must therefore have driven 23 total meters – 13 calculated meters = 10 meters after the 3rd turn. #### 10"} # noqa: E501
        ]
        tgt_list = [
            {"text": "Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year? A: He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 6*2=<<6*2=12>>12 pages every week That means he writes 12*52=<<12*52=624>>624 pages a year #### 624"} # noqa: E501
        ]
        dataset = Dataset.from_list(ds_list)
        valid_dataset = Dataset.from_list(vs_list)
        task_desc = "To solve high school-level math problems."
        op = LLMTaskRelevanceFilter(
            api_or_hf_model=self.api_or_hf_model,
        )
        op.prepare_valid_feature(valid_dataset, task_desc)
        self._run_test(dataset, op, tgt_list)

    def test_rft_data(self):
        ds_list = [
            {
                "text": "It is challenging to train a large language model.",
                "analysis": "",
                "answer": ""
            },
            {
                "text": "What is the capital of France?",
                "analysis": "The question asks for a factual piece of information about the capital city of France. The answer is straightforward and does not require any specialized knowledge or complex reasoning.",
                "answer": "The capital of France is Paris."
            },
            {
                "text": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
                "analysis": "He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 6*2=<<6*2=12>>12 pages every week That means he writes 12*52=<<12*52=624>>624 pages a year",
                "answer": "624"
            }
        ]
        vs_list = [
            {
                "text": "Q: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
                "analysis": "There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers. So in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers. Purple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers. That means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers. So in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.",
                "answer": "35"
            },
            {
                "text": "A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn?",
                "analysis": "From the details given, the car has traveled 5 meters at the 1st turn + 8 meters after the 2nd turn + 0 meters after the 4th turn = <<5+8+0=13>>13 meters around the ring. It must therefore have driven 23 total meters – 13 calculated meters = 10 meters after the 3rd turn.",
                "answer": "10"
            }
        ]
        tgt_list = [
            {
                "text": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
            }
        ]
        dataset = Dataset.from_list(ds_list)
        valid_dataset = Dataset.from_list(vs_list)
        task_desc = "To solve high school-level math problems."
        op = LLMTaskRelevanceFilter(
            api_or_hf_model=self.api_or_hf_model,
            min_score=0.5,
            input_keys=['text', 'analysis', 'answer'],
            field_names=['Query', 'Analysis', 'Answer'],
        )
        op.prepare_valid_feature(valid_dataset, task_desc)
        self._run_test(dataset, op, tgt_list)


if __name__ == '__main__':
    unittest.main()
