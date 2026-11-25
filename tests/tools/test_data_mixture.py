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

    def test_require_data_paths(self):
        for args in ([], ["--data_path"]):
            with self.subTest(args=args):
                with patch.object(sys, "argv", ["data_mixture.py", *args]):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        self.mixture.parse_args()
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
