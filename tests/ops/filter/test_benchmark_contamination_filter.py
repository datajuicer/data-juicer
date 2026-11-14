import gzip
import json
import os
import shutil
import tempfile
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.benchmark_contamination_filter import (
    BenchmarkContaminationFilter,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

BENCH_EN = (
    "Natalia sold clips to 48 of her friends in April, and then she "
    "sold half as many clips in May. How many clips did Natalia sell "
    "altogether in April and May? The answer is 72."
)
BENCH_ZH = (
    "某商店第一季度销售额为120万元，第二季度比第一季度增长了百分之二十五，"
    "请问第二季度的销售额是多少万元？答案是150万元。"
)
UNRELATED_EN = (
    "Photosynthesis is the process by which green plants convert "
    "light energy into chemical energy stored in glucose "
    "molecules, releasing oxygen as a byproduct of the reaction."
)
UNRELATED_ZH = (
    "长江发源于青藏高原的唐古拉山脉，自西向东流经十一个省级行政区，" "最终在上海附近注入东海，全长约六千三百公里。"
)
LONG_TAIL = (
    "Quantum entanglement describes a correlation between particles "
    "that cannot be explained by local hidden variables, and it was "
    "famously criticized by Einstein as spooky action at a distance, "
    "yet Bell inequalities and subsequent experiments confirmed the "
    "reality of the phenomenon beyond any reasonable doubt."
)


class BenchmarkContaminationFilterTest(DataJuicerTestCaseBase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.bench_path = os.path.join(self.tmp_dir, "bench.jsonl")
        with open(self.bench_path, "w", encoding="utf-8") as fout:
            for text in [BENCH_EN, BENCH_ZH]:
                record = json.dumps({"text": text}, ensure_ascii=False)
                fout.write(record + "\n")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run_filter(self, ds_list, op):
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            # only add the stats column when a filter OP is used
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, batch_size=op.batch_size)
        dataset = dataset.filter(op.process, batch_size=op.batch_size)
        dataset = dataset.select_columns(column_names=["text"])
        return dataset.to_list()

    def test_exact_copy_is_filtered(self):
        ds_list = [
            {"text": BENCH_EN},
            {"text": BENCH_ZH},
            {"text": UNRELATED_EN},
            {"text": UNRELATED_ZH},
        ]
        tgt_list = [
            {"text": UNRELATED_EN},
            {"text": UNRELATED_ZH},
        ]
        op = BenchmarkContaminationFilter(benchmark_path=self.bench_path, ngram_size=13, tokenization="char")
        self.assertEqual(self._run_filter(ds_list, op), tgt_list)

    def test_short_text_is_kept(self):
        # texts shorter than ngram_size carry no n-gram, they must not crash
        ds_list = [
            {"text": "hi"},
            {"text": ""},
            {"text": BENCH_ZH},
        ]
        tgt_list = [
            {"text": "hi"},
            {"text": ""},
        ]
        op = BenchmarkContaminationFilter(benchmark_path=self.bench_path)
        self.assertEqual(self._run_filter(ds_list, op), tgt_list)

    def test_threshold_controls_partial_overlap(self):
        # a document that quotes the benchmark and then drifts away from it
        partial = BENCH_EN + " " + LONG_TAIL
        ds_list = [{"text": partial}]

        # any overlap is enough to drop the sample
        strict = BenchmarkContaminationFilter(benchmark_path=self.bench_path, threshold=0.0)
        self.assertEqual(self._run_filter(ds_list, strict), [])

        # a loose threshold keeps the partially overlapping sample ...
        loose = BenchmarkContaminationFilter(benchmark_path=self.bench_path, threshold=0.9)
        self.assertEqual(self._run_filter(ds_list, loose), [{"text": partial}])

        # ... but still drops a verbatim copy, whose ratio is 1.0
        copy_ds = [{"text": BENCH_EN}]
        self.assertEqual(self._run_filter(copy_ds, loose), [])

    def test_stats_are_recorded(self):
        partial = BENCH_EN + " " + LONG_TAIL
        dataset = Dataset.from_list([{"text": partial}])
        dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        op = BenchmarkContaminationFilter(benchmark_path=self.bench_path)
        dataset = dataset.map(op.compute_stats, batch_size=1)
        stats = dataset[0][Fields.stats]
        self.assertIn("contamination_ratio", stats)
        self.assertIn("contamination_ngram_count", stats)
        self.assertGreater(stats["contamination_ratio"], 0.0)
        self.assertLess(stats["contamination_ratio"], 1.0)
        self.assertGreater(stats["contamination_ngram_count"], 0)

    def test_word_tokenization(self):
        op = BenchmarkContaminationFilter(benchmark_path=self.bench_path, ngram_size=8, tokenization="word")
        ds_list = [
            {"text": BENCH_EN},
            {"text": UNRELATED_EN},
        ]
        self.assertEqual(self._run_filter(ds_list, op), [{"text": UNRELATED_EN}])

    def test_benchmark_text_key(self):
        bench_path = os.path.join(self.tmp_dir, "bench_qa.jsonl")
        with open(bench_path, "w", encoding="utf-8") as fout:
            fout.write(json.dumps({"question": BENCH_EN, "answer": "72"}) + "\n")
        op = BenchmarkContaminationFilter(benchmark_path=bench_path, benchmark_text_key="question")
        self.assertEqual(self._run_filter([{"text": BENCH_EN}], op), [])

    def test_json_and_jsonl_gz_formats(self):
        json_path = os.path.join(self.tmp_dir, "bench.json")
        with open(json_path, "w", encoding="utf-8") as fout:
            json.dump([{"text": BENCH_EN}], fout)

        gz_path = os.path.join(self.tmp_dir, "bench.jsonl.gz")
        with gzip.open(gz_path, "wt", encoding="utf-8") as fout:
            fout.write(json.dumps({"text": BENCH_EN}) + "\n")

        txt_path = os.path.join(self.tmp_dir, "bench.txt")
        with open(txt_path, "w", encoding="utf-8") as fout:
            fout.write(BENCH_EN + "\n")

        for path in [json_path, gz_path, txt_path]:
            op = BenchmarkContaminationFilter(benchmark_path=path)
            self.assertEqual(self._run_filter([{"text": BENCH_EN}], op), [])

    def test_missing_benchmark_keeps_every_sample(self):
        # an empty benchmark_path must not break config validation
        op = BenchmarkContaminationFilter(benchmark_path=None)
        ds_list = [{"text": BENCH_EN}, {"text": UNRELATED_EN}]
        self.assertEqual(self._run_filter(ds_list, op), ds_list)

    def test_unsupported_format_raises(self):
        bad_path = os.path.join(self.tmp_dir, "bench.csv")
        with open(bad_path, "w", encoding="utf-8") as fout:
            fout.write("text\nhello\n")
        with self.assertRaises(ValueError):
            BenchmarkContaminationFilter(benchmark_path=bad_path)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            BenchmarkContaminationFilter(benchmark_path=self.bench_path, tokenization="sentence")
        with self.assertRaises(ValueError):
            BenchmarkContaminationFilter(benchmark_path=self.bench_path, ngram_size=0)


if __name__ == "__main__":
    unittest.main()
