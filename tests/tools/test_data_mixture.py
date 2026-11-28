import importlib.util
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from data_juicer.utils.data_mixture import (
    quality_weighted_rows,
    shuffle_rows,
    token_budget_shards,
    write_jsonl_shards,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class DataMixtureTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        # tests/run.py exposes tests/tools as "tools" during discovery.
        script = Path(__file__).resolve().parents[2] / "tools/postprocess/data_mixture.py"
        spec = importlib.util.spec_from_file_location("data_mixture", script)
        self.mixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mixture)

    def _check_export(self, budget, expected):
        for prefix in ("source", "source with spaces"):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sources = []
                for source in ("a", "b"):
                    path = root / f"{prefix}_{source}.jsonl"
                    path.write_text(
                        "".join(json.dumps({"text": f"{source}-{i}", "source": source}) + "\n" for i in range(12)),
                        encoding="utf-8",
                    )
                    sources.append(path.as_posix())
                output = root / "mixed.jsonl"
                argv = [
                    "data_mixture.py",
                    "--data_path",
                    "1",
                    sources[0],
                    "3",
                    sources[1],
                    "--export_path",
                    str(output),
                    "--num_proc",
                    "1",
                ]
                if budget is not None:
                    argv.extend(["--max_samples", str(budget)])
                with patch.object(sys, "argv", argv):
                    self.mixture.run_mixture()
                rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(Counter(row["source"] for row in rows), expected)
                self.assertTrue(all(row["text"] in {f'{row["source"]}-{i}' for i in range(12)} for row in rows))

    def test_export_without_budget(self):
        self._check_export(None, {"a": 12, "b": 12})

    def test_export_weighted_undersampling(self):
        self._check_export(8, {"a": 2, "b": 6})

    def test_export_weighted_oversampling(self):
        self._check_export(20, {"a": 5, "b": 15})

    def test_reject_nonpositive_sample_budget(self):
        for budget in ("0", "-1"):
            with self.subTest(budget=budget):
                stderr = io.StringIO()
                with patch.object(
                    sys, "argv", ["data_mixture.py", "--data_path", "unused.jsonl", "--max_samples", budget]
                ):
                    with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                        self.mixture.parse_args()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--max_samples must be a positive integer", stderr.getvalue())

    def test_reject_token_shard_without_budget(self):
        argv = ["data_mixture.py", "--data_path", "unused.jsonl", "--shard_by_tokens"]
        with patch.object(sys, "argv", argv):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.mixture.parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_token_budget_shards_respect_budget(self):
        rows = [{"text": f"r{i}"} for i in range(4)]
        shards = token_budget_shards(rows, [3, 3, 3, 3], budget=6)
        self.assertEqual([len(shard) for shard in shards], [2, 2])

    def test_token_budget_shards_allow_oversized_row(self):
        rows = [{"text": "big"}, {"text": "small"}]
        shards = token_budget_shards(rows, [100, 1], budget=10, tail_merge_ratio=0.0)
        self.assertEqual([len(shard) for shard in shards], [1, 1])

    def test_token_budget_shards_merge_small_tail(self):
        rows = [{"text": "a"}, {"text": "b"}]
        shards = token_budget_shards(rows, [6, 1], budget=10, tail_merge_ratio=0.5)
        self.assertEqual(len(shards), 1)

    def test_shuffle_rows_is_deterministic(self):
        rows = [{"i": i} for i in range(30)]
        first = shuffle_rows(rows, seed=42)
        second = shuffle_rows(rows, seed=42)
        other = shuffle_rows(rows, seed=43)
        self.assertEqual(first, second)
        self.assertNotEqual([r["i"] for r in first], [r["i"] for r in other])

    def test_quality_weighted_rows_prefers_high_quality(self):
        rows = [{"q": 0.0, "i": i} for i in range(100)] + [{"q": 1.0, "i": 100 + i} for i in range(100)]
        ordered = quality_weighted_rows(rows, "q", alpha=2.0, seed=7)
        high_in_top50 = sum(1 for row in ordered[:50] if row["q"] == 1.0)
        self.assertGreater(high_in_top50, 35)

    def test_write_jsonl_shards_names(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mixed.jsonl"
            written = write_jsonl_shards([[{"a": 1}], [{"a": 2}]], str(output))
            self.assertEqual([Path(p).name for p in written], ["mixed-00000.jsonl", "mixed-00001.jsonl"])
            self.assertEqual(json.loads(Path(written[1]).read_text(encoding="utf-8")), {"a": 2})

    def test_export_token_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for source in ("a", "b"):
                path = root / f"{source}.jsonl"
                path.write_text(
                    "".join(json.dumps({"text": "w w w", "source": source}) + "\n" for _ in range(10)),
                    encoding="utf-8",
                )
                sources.append(path.as_posix())
            output = root / "mixed.jsonl"
            argv = [
                "data_mixture.py",
                "--data_path",
                "1",
                sources[0],
                "1",
                sources[1],
                "--export_path",
                str(output),
                "--num_proc",
                "1",
                "--shard_by_tokens",
                "--token_budget",
                "10",
            ]
            with patch.object(sys, "argv", argv):
                self.mixture.run_mixture()
            shards = sorted(root.glob("mixed-*.jsonl"))
            self.assertGreater(len(shards), 1)
            rows = []
            for shard in shards:
                shard_rows = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
                self.assertLessEqual(sum(len(row["text"].split()) for row in shard_rows), 12)
                rows.extend(shard_rows)
            self.assertEqual(len(rows), 20)


if __name__ == "__main__":
    unittest.main()
