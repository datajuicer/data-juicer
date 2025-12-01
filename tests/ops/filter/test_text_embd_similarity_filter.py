import unittest

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.text_embd_similarity_filter import TextEmbdSimilarityFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, FROM_FORK

@unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
class TextEmbdSimilarityFilterTest(DataJuicerTestCaseBase):

    _hf_model = "Qwen/Qwen3-Embedding-0.6B"
    # _hf_model = "/your/local/path/to/Qwen3-Embedding-0.6B"

    text_key = "text"

    def _run_filter(self, dataset: Dataset, op, tgt_list, num_proc=1):
        if Fields.stats not in dataset.features:
            # this is a temp solution, only add stats when calling filter op
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)

        dataset = dataset.map(op.compute_stats, num_proc=num_proc, with_rank=True)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.select_columns(column_names=[self.text_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, tgt_list)

    def test_api(self):

        ds_list = [
            {"text": "There is a lovely cat."},
            {"text": "It is challenging to train a large language model."}
        ]
        vs_list = [
            {"text": "There is a cute cat."},
            {"text": "The cat there is lovely."}
        ]
        tgt_list = [
            {"text": "There is a lovely cat."},
        ]

        dataset, valid_dataset = Dataset.from_list(ds_list), Dataset.from_list(vs_list)
        op = TextEmbdSimilarityFilter(
            api_or_hf_model="text-embedding-v4", # Based on Qwen3-Embedding-8B
            is_hf_model=False,
            min_score=0.7,
            max_score=1.0,
            ebd_dim=2048
        )
        op.prepare_valid_feature(valid_dataset)
        self._run_filter(dataset, op, tgt_list)

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
        op = TextEmbdSimilarityFilter(
            api_or_hf_model="text-embedding-v4",  # Based on Qwen3-Embedding-8B
            is_hf_model=False,
            min_score=0.2,
            max_score=1.0,
            ebd_dim=2048,
            input_template="{text} {analysis} {answer}",
        )
        op.prepare_valid_feature(valid_dataset)
        self._run_filter(dataset, op, tgt_list)

    def test_hf_model(self):

        ds_list = [
            {"text": "There is a lovely cat."},
            {"text": "It is challenging to train a large language model."}
        ]
        vs_list = [
            {"text": "There is a cute cat."},
            {"text": "The cat there is lovely."}
        ]
        tgt_list = [
            {"text": "There is a lovely cat."},
        ]

        dataset, valid_dataset = Dataset.from_list(ds_list), Dataset.from_list(vs_list)
        op = TextEmbdSimilarityFilter(
            api_or_hf_model=self._hf_model,
            is_hf_model=True,
            min_score=0.97,
            max_score=1.0,
        )
        op.prepare_valid_feature(valid_dataset)
        self._run_filter(dataset, op, tgt_list)

    def test_hf_model_mean_pooling(self):

        ds_list = [
            {"text": "There is a lovely cat."},
            {"text": "It is challenging to train a large language model."}
        ]
        vs_list = [
            {"text": "There is a cute cat."},
            {"text": "The cat there is lovely."}
        ]
        tgt_list = [
            {"text": "There is a lovely cat."},
        ]

        dataset, valid_dataset = Dataset.from_list(ds_list), Dataset.from_list(vs_list)
        op = TextEmbdSimilarityFilter(
            api_or_hf_model=self._hf_model,
            is_hf_model=True,
            min_score=0.97,
            max_score=1.0,
            pooling="mean"
        )
        op.prepare_valid_feature(valid_dataset)
        self._run_filter(dataset, op, tgt_list)

    def test_hf_model_weighted_mean_pooling(self):
        ds_list = [
            {"text": "There is a lovely cat."},
            {"text": "It is challenging to train a large language model."}
        ]
        vs_list = [
            {"text": "There is a cute cat."},
            {"text": "The cat there is lovely."}
        ]
        tgt_list = [
            {"text": "There is a lovely cat."},
        ]

        dataset, valid_dataset = Dataset.from_list(ds_list), Dataset.from_list(vs_list)
        op = TextEmbdSimilarityFilter(
            api_or_hf_model=self._hf_model,
            is_hf_model=True,
            min_score=0.99,
            max_score=1.0,
            pooling="weighted_mean"
        )
        op.prepare_valid_feature(valid_dataset)
        self._run_filter(dataset, op, tgt_list)


if __name__ == '__main__':
    unittest.main()
