import gzip
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

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

    def test_second_benchmark_does_not_reuse_stats(self):
        """Two benchmarks in one recipe must not share their stats fields.

        The base filter reuses a stats field that is already present, so a
        second contamination filter would otherwise read the ratio computed
        against the first benchmark and keep a sample that is contaminated
        against its own benchmark.
        """
        bench_a = os.path.join(self.tmp_dir, "bench_a.jsonl")
        bench_b = os.path.join(self.tmp_dir, "bench_b.jsonl")
        with open(bench_a, "w", encoding="utf-8") as fout:
            fout.write(json.dumps({"text": LONG_TAIL}, ensure_ascii=False) + "\n")
        with open(bench_b, "w", encoding="utf-8") as fout:
            fout.write(json.dumps({"text": BENCH_EN}, ensure_ascii=False) + "\n")

        dataset = Dataset.from_list([{"text": BENCH_EN}])
        dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)

        # benchmark A is unrelated: the sample is kept and its ratio is 0
        op_a = BenchmarkContaminationFilter(benchmark_path=bench_a)
        dataset = dataset.map(op_a.compute_stats, batch_size=1)
        self.assertEqual(dataset[0][Fields.stats]["contamination_ratio"], 0.0)

        # benchmark B contains the sample verbatim: it must be dropped,
        # which only works when B writes to its own stats fields
        op_b = BenchmarkContaminationFilter(
            benchmark_path=bench_b,
            contamination_ratio_key="contamination_ratio_b",
            contamination_ngram_count_key="contamination_ngram_count_b",
        )
        dataset = dataset.map(op_b.compute_stats, batch_size=1)
        stats = dataset[0][Fields.stats]
        self.assertEqual(stats["contamination_ratio_b"], 1.0)
        self.assertGreater(stats["contamination_ngram_count_b"], 0)
        # the first benchmark's stats are left untouched
        self.assertEqual(stats["contamination_ratio"], 0.0)

        filtered = dataset.filter(op_b.process, batch_size=1)
        self.assertEqual(filtered.num_rows, 0)

    def test_suite_scores_match_independent_filters(self):
        other_path = os.path.join(self.tmp_dir, "science.txt")
        with open(other_path, "w", encoding="utf-8") as fout:
            fout.write(LONG_TAIL)

        suites = {"math": self.bench_path, "science": other_path}
        records = [{"text": text} for text in [BENCH_EN, LONG_TAIL, UNRELATED_ZH, ""]]
        op = BenchmarkContaminationFilter(benchmark_suites=suites, threshold=0.5, auto_op_parallelism=False)
        dataset = Dataset.from_list(records)
        scored = op.run(dataset, reduce=False)
        for row in scored:
            stats = row[Fields.stats]
            scores = stats["contamination_breakdown"]["suites"]
            self.assertEqual(set(scores), set(suites))
            for suite, path in suites.items():
                single = BenchmarkContaminationFilter(benchmark_path=path)
                expected = single.compute_stats_single({"text": row["text"], Fields.stats: {}})[Fields.stats]
                self.assertEqual(
                    scores[suite],
                    {
                        "ratio": expected["contamination_ratio"],
                        "ngram_count": expected["contamination_ngram_count"],
                    },
                )
            self.assertEqual(stats["contamination_ratio"], max(score["ratio"] for score in scores.values()))
        self.assertEqual(op.run(dataset).select_columns("text").to_list(), records[2:])

    def test_suite_hashes_once_and_refreshes_when_reference_changes(self):
        opts = {"benchmark_suites": {"a": self.bench_path, "b": self.bench_path}}
        op = BenchmarkContaminationFilter(**opts)
        row = {"text": BENCH_EN, Fields.stats: {"contamination_ratio": 0.0}}
        with patch.object(op, "_ngram_hashes", wraps=op._ngram_hashes) as hashes:
            op.compute_stats_single(row)
            op.compute_stats_single(row)
            hashes.assert_called_once()
        self.assertEqual(row[Fields.stats]["contamination_ratio"], 1.0)

        with open(self.bench_path, "w", encoding="utf-8") as fout:
            fout.write(json.dumps({"text": UNRELATED_ZH}) + "\n")
        revised = BenchmarkContaminationFilter(**opts)
        revised.compute_stats_single(row)
        self.assertEqual(row[Fields.stats]["contamination_ratio"], 0.0)

    def test_invalid_suite_settings_fail_early(self):
        for sources in ({}, {"": self.bench_path}, {"broken": ""}):
            with self.subTest(sources=sources), self.assertRaisesRegex(ValueError, "benchmark_suites"):
                BenchmarkContaminationFilter(benchmark_suites=sources)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            BenchmarkContaminationFilter(benchmark_path=self.bench_path, benchmark_suites={"a": self.bench_path})
        with self.assertRaisesRegex(ValueError, "distinct"):
            BenchmarkContaminationFilter(
                benchmark_suites={"a": self.bench_path},
                contamination_breakdown_key="contamination_ratio",
            )

    def test_suite_stats_refresh_after_text_rewrite(self):
        op = BenchmarkContaminationFilter(
            benchmark_suites={"math": self.bench_path},
            threshold=0.5,
            auto_op_parallelism=False,
            num_proc=None,
        )
        clean = Dataset.from_list([{"text": UNRELATED_EN}])
        scored = op.run(clean, reduce=False)
        self.assertEqual(scored[0][Fields.stats]["contamination_ratio"], 0.0)
        rewritten = scored.map(lambda _: {"text": BENCH_EN})
        rescored = op.run(rewritten, reduce=False)
        self.assertEqual(rescored[0][Fields.stats]["contamination_ratio"], 1.0)
        self.assertEqual(len(op.run(rewritten)), 0)

        normalized_equivalent = dict(rescored[0])
        normalized_equivalent["text"] = "  " + BENCH_EN.upper() + "  "
        with patch.object(op, "_ngram_hashes", side_effect=AssertionError("unexpected recomputation")):
            op.compute_stats_single(normalized_equivalent)


if __name__ == "__main__":
    unittest.main()
